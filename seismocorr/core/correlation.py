# seismocorr/core/correlation.py

"""
Cross-Correlation Core Module

提供灵活高效的互相关计算接口，支持：
- 时域 / 频域算法选择
- 多种归一化与滤波预处理
- 单道对或多道批量输入
- 返回标准 CCF 结构（lags, ccf）

不包含文件 I/O 或任务调度 —— 这些由 pipeline 层管理。
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# 优化库导入
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(func=None, **kwargs):
        """Numba不可用时的回退装饰器"""
        if func is None:
            return lambda f: f
        return func

    def prange(n):
        """Numba不可用时的回退prange"""
        return range(n)


from scipy.fftpack import fft, ifft

from seismocorr.preprocessing.freq_norm import get_freq_normalizer
from seismocorr.preprocessing.normal_func import bandpass
from seismocorr.preprocessing.time_norm import get_time_normalizer

# -----------------------------
# 类型定义
# -----------------------------
ArrayLike = Union[np.ndarray, List[float]]
LagsAndCCF = Tuple[np.ndarray, np.ndarray]
BatchResult = Dict[str, LagsAndCCF]  # {channel_pair: (lags, ccf)}


# -----------------------------
# 核心算法枚举
# -----------------------------
SUPPORTED_METHODS = ["time-domain", "freq-domain", "deconv", "coherency"]
NORMALIZATION_OPTIONS = ["zscore", "one-bit", "rms", "no"]

# -----------------------------
# 计算结果缓存
# -----------------------------
class CCF_Cache:
    """
    互相关计算结果缓存类
    使用LRU策略管理缓存大小，避免内存溢出
    """

    def __init__(self, max_size: int = 10000):
        """
        初始化缓存

        Args:
            max_size: 缓存最大容量，超过则删除最旧的条目
        """
        self.max_size = max_size
        self.cache = {}  # {cache_key: (timestamp, result)}
        self.access_order = []  # 记录访问顺序，用于LRU

    def _generate_key(self, a: str, b: str, sampling_rate: float, **kwargs) -> str:
        """
        生成缓存键

        Args:
            a, b: 通道名称
            sampling_rate: 采样率
            **kwargs: 计算参数

        Returns:
            唯一的缓存键
        """
        # 排序通道名称，确保(a,b)和(b,a)生成相同的键
        sorted_chans = tuple(sorted([a, b]))
        # 提取关键参数
        method = kwargs.get("method", "time-domain")
        time_normalize = kwargs.get("time_normalize", "one-bit")
        freq_normalize = kwargs.get("freq_normalize", "no")
        freq_band = kwargs.get("freq_band", None)
        max_lag = kwargs.get("max_lag", None)

        # 生成键
        key_parts = [
            f"chans:{sorted_chans}",
            f"sr:{sampling_rate}",
            f"method:{method}",
            f"time_norm:{time_normalize}",
            f"freq_norm:{freq_normalize}",
            f"freq_band:{freq_band}",
            f"max_lag:{max_lag}",
        ]
        return "|".join(key_parts)

    def get(
        self, a: str, b: str, sampling_rate: float, **kwargs
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        获取缓存结果

        Args:
            a, b: 通道名称
            sampling_rate: 采样率
            **kwargs: 计算参数

        Returns:
            缓存的结果，如果不存在则返回None
        """
        key = self._generate_key(a, b, sampling_rate, **kwargs)
        if key in self.cache:
            # 更新访问顺序
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key][1]
        return None

    def set(
        self,
        a: str,
        b: str,
        sampling_rate: float,
        result: Tuple[np.ndarray, np.ndarray],
        **kwargs,
    ) -> None:
        """
        设置缓存结果

        Args:
            a, b: 通道名称
            sampling_rate: 采样率
            result: 计算结果
            **kwargs: 计算参数
        """
        key = self._generate_key(a, b, sampling_rate, **kwargs)

        # 添加到缓存
        self.cache[key] = (0, result)  # 简化实现，不使用真实时间戳
        self.access_order.append(key)

        # 如果超过最大容量，删除最旧的条目
        if len(self.cache) > self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]


# 创建全局缓存实例
ccf_cache = CCF_Cache(max_size=10000)


# -----------------------------
# 主要函数：compute_cross_correlation
# -----------------------------


@njit(cache=True, fastmath=True, nogil=True)
def _apply_preprocessing(data: np.ndarray, window: int) -> np.ndarray:
    """
    使用Numba优化的预处理函数
    """
    # 去趋势（简化版本，仅去线性趋势）
    n = len(data)
    x = np.arange(n)
    mean_x = np.mean(x)
    mean_y = np.mean(data)

    # 计算斜率
    slope = np.sum((x - mean_x) * (data - mean_y)) / np.sum((x - mean_x) ** 2)
    intercept = mean_y - slope * mean_x

    # 去趋势
    detrended = data - (slope * x + intercept)

    # 去均值
    demeaned = detrended - np.mean(detrended)

    # 加窗
    if window > 0:
        # 创建汉宁窗
        han_window = np.hanning(2 * window)
        # 应用窗函数到数据两端
        demeaned[:window] *= han_window[:window]
        demeaned[-window:] *= han_window[window:]

    return demeaned


def compute_cross_correlation(
    x: ArrayLike,
    y: ArrayLike,
    sampling_rate: float,
    method: str = "time-domain",
    time_normalize: str = "one-bit",
    freq_normalize: str = "no",
    freq_band: Optional[Tuple[float, float]] = None,
    max_lag: Optional[Union[float, int]] = None,
    nfft: Optional[int] = None,
    time_norm_kwargs: Optional[Dict[str, Any]] = None,
    freq_norm_kwargs: Optional[Dict[str, Any]] = None,
) -> LagsAndCCF:
    """
    计算两个时间序列的互相关函数（CCF）

    Args:
        x, y: 时间序列数据
        sampling_rate: 采样率 (Hz)
        method: 计算方法 ('time-domain', 'freq-domain', 'deconv', 'coherency')
        normalize: 归一化方式
        freq_band: 带通滤波范围 (fmin, fmax)，单位 Hz
        max_lag: 最大滞后时间（秒）；若为 None，则使用 min(len(x), len(y))
        nfft: FFT 长度，自动补零到 next_fast_len

    Returns:
        lags: 时间滞后数组 (单位：秒)
        ccf: 互相关函数值
    """
    # 转换为浮点数组
    x = _as_float_array(x)
    y = _as_float_array(y)

    if len(x) == 0 or len(y) == 0:
        return np.array([]), np.array([])

    # 确定最大滞后
    if not max_lag:
        max_lag = min(len(x), len(y)) / sampling_rate

    # 初始化参数字典（避免不必要的复制）
    time_norm_kwargs = time_norm_kwargs or {}
    freq_norm_kwargs = freq_norm_kwargs or {}

    # 对x和y进行预处理，使用Numba优化的函数
    # 跳过非常短的信号的预处理，避免除以零错误
    if len(x) > 2 and len(y) > 2:
        window = max(1, int(len(x) * 0.05))
        x = _apply_preprocessing(x, window)
        y = _apply_preprocessing(y, window)

    # 滤波（如果需要）
    if freq_band is not None:
        x = bandpass(x, freq_band[0], freq_band[1], sr=sampling_rate)
        y = bandpass(y, freq_band[0], freq_band[1], sr=sampling_rate)

    # 时域归一化
    time_norm_kwargs_with_fs = {**time_norm_kwargs, "Fs": sampling_rate, "npts": len(x)}
    normalizer = get_time_normalizer(time_normalize, **time_norm_kwargs_with_fs)
    x = normalizer.apply(x)
    y = normalizer.apply(y)

    # 频域归一化
    freq_norm_kwargs_with_fs = {**freq_norm_kwargs, "Fs": sampling_rate}
    normalizer = get_freq_normalizer(freq_normalize, **freq_norm_kwargs_with_fs)
    x = normalizer.apply(x)
    y = normalizer.apply(y)

    # 截断到相同长度（避免后续处理中的不匹配）
    min_len = min(len(x), len(y))
    if len(x) > min_len:
        x = x[:min_len]
    if len(y) > min_len:
        y = y[:min_len]

    # 选择方法
    if method == "time-domain":
        lags, ccf = _xcorr_time_domain(x, y, sampling_rate, max_lag)
    elif method in ["freq-domain", "deconv"]:
        lags, ccf = _xcorr_freq_domain(
            x, y, sampling_rate, max_lag, nfft, deconv=method == "deconv"
        )
    elif method == "coherency":
        lags, ccf = _coherency(x, y, sampling_rate, max_lag, nfft)
    else:
        raise ValueError(
            f"Unsupported method: {method}. Choose from {SUPPORTED_METHODS}"
        )

    return lags, ccf


# -----------------------------
# 批量计算多个道对
# -----------------------------


def batch_cross_correlation_sequential(
    traces: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str]],
    sampling_rate: float,
    use_cache: bool = False,
    **kwargs,
) -> BatchResult:
    """
    批量计算多个通道对之间的互相关

    Example:
        result = batch_cross_correlation(
            traces={'STA1.CHZ': data1, 'STA2.CHZ': data2},
            pairs=[('STA1.CHZ', 'STA2.CHZ')],
            sampling_rate=100,
            method='freq-domain'
        )

    Args:
        traces: 通道数据字典
        pairs: 通道对列表
        sampling_rate: 采样率
        use_cache: 是否使用缓存机制
        **kwargs: 计算参数

    Returns:
        dict: { "STA1.CHZ--STA2.CHZ": (lags, ccf), ... }
    """
    result = {}
    # 从kwargs中移除use_cache，避免传递给compute_cross_correlation
    compute_kwargs = kwargs.copy()

    for a, b in pairs:
        if a not in traces or b not in traces:
            continue

        try:
            # 尝试从缓存获取结果
            if use_cache:
                cached_result = ccf_cache.get(a, b, sampling_rate, **compute_kwargs)
                if cached_result is not None:
                    key = f"{a}--{b}"
                    result[key] = cached_result
                    continue

            # 缓存中没有，计算新结果
            lags, ccf = compute_cross_correlation(
                traces[a], traces[b], sampling_rate, **compute_kwargs
            )
            key = f"{a}--{b}"
            result[key] = (lags, ccf)

            # 将结果存入缓存
            if use_cache:
                ccf_cache.set(a, b, sampling_rate, (lags, ccf), **compute_kwargs)

        except Exception as e:
            print(f"Failed on pair {a}-{b}: {e}")
    return result


def _process_single_pair(args):
    """处理单个通道对的辅助函数"""
    a, b, trace_a, trace_b, sampling_rate, kwargs, use_cache = args
    compute_kwargs = kwargs.copy()

    try:
        # 尝试从缓存获取结果
        if use_cache:
            from seismocorr.core.correlation import ccf_cache

            cached_result = ccf_cache.get(a, b, sampling_rate, **compute_kwargs)
            if cached_result is not None:
                key = f"{a}--{b}"
                return key, cached_result

        # 缓存中没有，计算新结果
        lags, ccf = compute_cross_correlation(
            trace_a, trace_b, sampling_rate, **compute_kwargs
        )
        key = f"{a}--{b}"

        # 将结果存入缓存
        if use_cache:
            from seismocorr.core.correlation import ccf_cache

            ccf_cache.set(a, b, sampling_rate, (lags, ccf), **compute_kwargs)

        return key, (lags, ccf)
    except Exception as e:
        print(f"Failed on pair {a}-{b}: {e}")
        return None, None


def batch_cross_correlation(
    traces: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str]],
    sampling_rate: float,
    n_jobs: int = -1,
    parallel_backend: str = "auto",  # "auto", "process", "thread"
    use_cache: bool = True,
    **kwargs,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    批量计算多个通道对之间的互相关（并行版本）

    Args:
        n_jobs: 并行工作数，-1 表示使用所有CPU核心
        parallel_backend: 并行后端 ("auto", "process", "thread")
        use_cache: 是否使用缓存机制
    """
    # 优化1：小批量直接顺序执行，避免并行开销
    if n_jobs == 0 or n_jobs == 1 or len(pairs) <= 4:
        return batch_cross_correlation_sequential(
            traces, pairs, sampling_rate, use_cache=use_cache, **kwargs
        )

    # 优化2：动态调整并行度，根据任务规模和硬件情况优化
    num_cores = mp.cpu_count()
    num_pairs = len(pairs)

    if n_jobs == -1:
        # 根据任务规模动态调整核心数
        if num_pairs < num_cores:
            # 任务数少于核心数，使用任务数作为并行度
            n_jobs = num_pairs
        elif num_pairs < num_cores * 2:
            # 任务数适中，使用所有核心
            n_jobs = num_cores
        else:
            # 任务数较多，使用核心数的2倍，充分利用硬件资源
            n_jobs = min(int(num_cores * 2), num_pairs)

    # 限制最大并行度，避免过多的线程/进程切换开销
    n_jobs = min(n_jobs, num_pairs, num_cores * 4)

    # 从kwargs中移除use_cache，避免传递给compute_cross_correlation
    compute_kwargs = kwargs.copy()

    # 优化3：提前验证所有通道对，避免运行时错误
    valid_tasks = []
    for a, b in pairs:
        if a not in traces:
            print(
                f"Warning: Trace for channel '{a}' not found, skipping pair ({a}, {b})"
            )
            continue
        if b not in traces:
            print(
                f"Warning: Trace for channel '{b}' not found, skipping pair ({a}, {b})"
            )
            continue
        valid_tasks.append(
            (a, b, traces[a], traces[b], sampling_rate, compute_kwargs, use_cache)
        )

    if not valid_tasks:
        return {}

    # 智能选择最佳并行后端
    avg_data_size = 0
    for a, b, trace_a, trace_b, _, _, _ in valid_tasks:
        avg_data_size += len(trace_a) + len(trace_b)
    avg_data_size /= len(valid_tasks)

    # 根据方法类型判断任务特性
    method = kwargs.get("method", "time-domain")
    is_cpu_intensive = method in ["time-domain", "freq-domain"]

    # 自动选择最佳并行后端
    if parallel_backend == "auto":
        if is_cpu_intensive and avg_data_size < 5e5:
            # 计算密集型且数据量较小，优先使用进程池
            parallel_backend = "process"
        else:
            # 数据量较大或IO密集型，优先使用线程池
            parallel_backend = "thread"

    # 选择执行器
    if parallel_backend == "process":
        Executor = ProcessPoolExecutor
    else:
        Executor = ThreadPoolExecutor

    results = {}

    with Executor(max_workers=n_jobs) as executor:
        # 使用map代替submit，减少任务创建和管理开销
        for key, result_val in executor.map(_process_single_pair, valid_tasks):
            if key is not None and result_val is not None:
                results[key] = result_val

    return results


# -----------------------------
# 内部实现函数
# -----------------------------


@njit(cache=True, fastmath=True, nogil=True)
def _as_float_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).flatten()


@njit(cache=True, fastmath=True, nogil=True)
def _xcorr_time_domain(
    x: np.ndarray, y: np.ndarray, sr: float, max_lag: float
) -> LagsAndCCF:
    """
    时域互相关计算

    Args:
        x, y: 输入信号
        sr: 采样率 (Hz)
        max_lag: 最大滞后时间（秒）

    Returns:
        lags: 时间滞后数组 (单位：秒)
        ccf: 互相关函数值
    """
    # 确保输入信号长度相同
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    # 将秒转换为样本数
    max_lag_samples = int(max_lag * sr)

    # 限制最大滞后不超过信号长度
    max_lag_samples = min(max_lag_samples, min_len - 1)

    n = len(x)
    ccf_len = 2 * n - 1
    ccf_full = np.zeros(ccf_len)

    for i in range(ccf_len):
        # 计算当前滞后的起始和结束索引
        shift = i - (n - 1)
        if shift < 0:
            # 正lag（x领先y）
            start = -shift
            end = n
            x_segment = x[start:end]
            y_segment = y[: end - start]
        else:
            # 负lag（y领先x）
            start = 0
            end = n - shift
            x_segment = x[:end]
            y_segment = y[shift : shift + end]

        # 计算点积
        ccf_full[i] = np.sum(x_segment * y_segment)

    # 计算滞后对应的索引范围
    center = ccf_len // 2  # 零滞后对应的索引

    # 截取从 -max_lag_samples 到 +max_lag_samples 的部分
    start_idx = center - max_lag_samples
    end_idx = center + max_lag_samples + 1  # +1 确保包含max_lag_samples

    # 确保索引不越界
    start_idx = max(0, start_idx)
    end_idx = min(ccf_len, end_idx)

    # 提取互相关值
    ccf = ccf_full[start_idx:end_idx]

    # 计算对应的滞后时间（秒）
    lags_samples = np.arange(-max_lag_samples, max_lag_samples + 1)
    lags = lags_samples / sr

    # 归一化互相关
    norm_factor = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if norm_factor > 0:
        ccf = ccf / norm_factor

    return lags, ccf


def _xcorr_freq_domain(
    x: np.ndarray,
    y: np.ndarray,
    sr: float,
    max_lag: float,
    nfft: Optional[int],
    deconv=False,
) -> LagsAndCCF:
    """
    频域互相关/去卷积计算

    Args:
        x, y: 输入信号
        sr: 采样率 (Hz)
        max_lag: 最大滞后时间（秒）
        nfft: FFT长度
        deconv: 如果为True，执行去卷积；如果为False，执行标准互相关

    Returns:
        lags: 时间滞后数组（秒）
        ccf: 互相关/去卷积结果

    注意：
    - 当 deconv=False: 计算标准互相关，用于信号相似性分析
    - 当 deconv=True: 计算去卷积，用于系统辨识或反卷积
    """
    length = len(x)
    if nfft is None:
        from scipy.fftpack import next_fast_len

        nfft = next_fast_len(length)

    X = fft(x, n=nfft)
    Y = fft(y, n=nfft)

    if deconv:
        # Deconvolution: Y/X
        eps = np.median(np.abs(X)) * 1e-6
        Sxy = Y / (X + eps)
    else:
        # Cross-spectrum
        Sxy = np.conj(X) * Y

    ccf_full = np.real(ifft(Sxy))
    # 因为是循环互相关，需要移位
    ccf_shifted = np.fft.ifftshift(ccf_full)

    # 提取 ±max_lag 范围
    center = nfft // 2
    lag_in_samples = int(max_lag * sr)
    start = center - lag_in_samples
    end = center + lag_in_samples + 1
    lags = np.arange(-lag_in_samples, lag_in_samples + 1) / sr
    return lags, ccf_shifted[start:end]


def _coherency(
    x: np.ndarray, y: np.ndarray, sr: float, max_lag: int, nfft: Optional[int]
) -> LagsAndCCF:
    """使用相干性作为权重的互相关（类似 PWS 的频域版本），提高互相关结果的可靠性"""
    lags, ccf_raw = _xcorr_freq_domain(x, y, sr, max_lag, nfft, deconv=False)

    # 在频域计算相位一致性
    Cxy = fft(ccf_raw)
    phase = np.exp(1j * np.angle(Cxy))
    coh = np.abs(np.mean(phase)) ** 4  # 权重
    return lags, ccf_raw * coh
