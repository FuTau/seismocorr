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

import numpy as np
from typing import Union, Tuple, Optional, Dict, List
from scipy.signal import correlate, butter, filtfilt
from scipy.fftpack import fft, ifft

from seismocorr.preprocessing.time_norm import get_time_normalizer
from seismocorr.preprocessing.freq_norm import get_freq_normalizer
from seismocorr.core.stacking import stack_ccfs


# -----------------------------
# 类型定义
# -----------------------------
ArrayLike = Union[np.ndarray, List[float]]
LagsAndCCF = Tuple[np.ndarray, np.ndarray]
BatchResult = Dict[str, LagsAndCCF]  # {channel_pair: (lags, ccf)}


# -----------------------------
# 核心算法枚举
# -----------------------------
SUPPORTED_METHODS = ['time-domain', 'freq-domain', 'deconv', 'coherency']
NORMALIZATION_OPTIONS = ['zscore', 'one-bit', 'rms', 'no']


# -----------------------------
# 主要函数：compute_cross_correlation
# -----------------------------

def compute_cross_correlation(
    x: ArrayLike,
    y: ArrayLike,
    sampling_rate: float,
    method: str = 'time-domain',
    time_normalize: str = 'one-bit',
    freq_normalize: str = 'no',
    freq_band: Optional[Tuple[float, float]] = None,
    max_lag: Optional[Union[float, int]] = None,
    nfft: Optional[int] = None,
) -> LagsAndCCF:
    """
    计算两个时间序列的互相关函数（CCF）

    Args:
        x, y: 时间序列数据
        sampling_rate: 采样率 (Hz)
        method: 计算方法 ('time-domain', 'freq-domain', 'deconv', 'coherency')
        normalize: 归一化方式
        freq_band: 带通滤波范围 (fmin, fmax)，单位 Hz
        max_lag: 最大滞后时间（秒）或样本数；若为 None，则使用 min(len(x), len(y))
        nfft: FFT 长度，自动补零到 next_fast_len

    Returns:
        lags: 时间滞后数组 (单位：秒)
        ccf: 互相关函数值
    """
    x = _as_float_array(x)
    y = _as_float_array(y)

    # 参数预处理
    if max_lag is not None:
        if isinstance(max_lag, float):
            max_lag_samples = int(max_lag * sampling_rate)
        else:
            max_lag_samples = int(max_lag)
    else:
        max_lag_samples = min(len(x), len(y)) // 2

    # 滤波
    if freq_band is not None:
        x = _bandpass_filter(x, freq_band, sampling_rate)
        y = _bandpass_filter(y, freq_band, sampling_rate)

    # 归一化
    normalizer = get_time_normalizer(time_normalize)
    x = normalizer.apply(x)
    y = normalizer.apply(y)

    normalizer = get_freq_normalizer(freq_normalize)
    x = normalizer.apply(x)
    y = normalizer.apply(y)

    # 截断到相同长度
    min_len = min(len(x), len(y))
    x, y = x[:min_len], y[:min_len]

    # 选择方法
    if method == 'time-domain':
        lags, ccf = _xcorr_time_domain(x, y, sampling_rate, max_lag_samples)
    elif method in ['freq-domain', 'deconv']:
        lags, ccf = _xcorr_freq_domain(x, y, sampling_rate, max_lag_samples, nfft, deconv=method=='deconv')
    elif method == 'coherency':
        lags, ccf = _coherency(x, y, sampling_rate, max_lag_samples, nfft)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose from {SUPPORTED_METHODS}")

    return lags, ccf


# -----------------------------
# 批量计算多个道对
# -----------------------------

def batch_cross_correlation(
    traces: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str]],
    sampling_rate: float,
    **kwargs
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

    Returns:
        dict: { "STA1.CHZ--STA2.CHZ": (lags, ccf), ... }
    """
    result = {}
    for a, b in pairs:
        if a not in traces or b not in traces:
            continue
        try:
            lags, ccf = compute_cross_correlation(traces[a], traces[b], sampling_rate, **kwargs)
            key = f"{a}--{b}"
            result[key] = (lags, ccf)
        except Exception as e:
            print(f"Failed on pair {a}-{b}: {e}")
    return result


# -----------------------------
# 内部实现函数
# -----------------------------

def _as_float_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).flatten()

def _bandpass_filter(data: np.ndarray, band: Tuple[float, float], sr: float) -> np.ndarray:
    fmin, fmax = band
    nyq = sr / 2.0
    if fmax >= nyq:
        fmax = 0.95 * nyq
    b, a = butter(4, [fmin / nyq, fmax / nyq], btype='band')
    return filtfilt(b, a, data)

def _xcorr_time_domain(x: np.ndarray, y: np.ndarray, sr: float, max_lag: int) -> LagsAndCCF:
    ccf = correlate(x, y, mode='same')
    n = len(ccf)
    half = n // 2
    start = max(half - max_lag, 0)
    end = min(half + max_lag, n)
    lags = np.arange(start - half, end - half) / sr
    return lags, ccf[start:end]

def _xcorr_freq_domain(x: np.ndarray, y: np.ndarray, sr: float, max_lag: int, nfft: Optional[int], deconv=False) -> LagsAndCCF:
    length = len(x)
    if nfft is None:
        from scipy.fftpack.helper import next_fast_len
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
    lag_in_samples = int(max_lag)
    start = center - lag_in_samples
    end = center + lag_in_samples + 1
    lags = np.arange(-lag_in_samples, lag_in_samples + 1) / sr
    return lags, ccf_shifted[start:end]

def _coherency(x: np.ndarray, y: np.ndarray, sr: float, max_lag: int, nfft: Optional[int]) -> LagsAndCCF:
    """使用相干性作为权重的互相关（类似 PWS 的频域版本）"""
    lags, ccf_raw = _xcorr_freq_domain(x, y, sr, max_lag, nfft, deconv=False)
    
    # 在频域计算相位一致性
    Cxy = fft(ccf_raw)
    phase = np.exp(1j * np.angle(Cxy))
    coh = np.abs(np.mean(phase)) ** 4  # 权重
    return lags, ccf_raw * coh


# -----------------------------
# 工具函数：多段叠加（Sub-stacking）
# -----------------------------

def sub_stack_ccfs(
    trace_pairs: List[Tuple[ArrayLike, ArrayLike]],
    sampling_rate: float,
    method: str = 'linear',
    segment_length: float = 3600.0,
    step: float = 1800.0,
    **kwargs
) -> LagsAndCCF:
    """
    对长时间连续数据分段做互相关后进行子叠加（sub-stacking）

    Args:
        trace_pairs: 多组 (x, y) 数据对（如每小时一段）
        sampling_rate: 采样率
        method: 叠加方法（见 stacking.stack_ccfs）
        segment_length: 每段长度（秒）
        step: 步长（用于非整除情况）
        **kwargs: 传递给 compute_cross_correlation 的参数

    Returns:
        最终叠加的 CCF
    """
    ccfs = []
    duration = len(trace_pairs[0][0]) / sampling_rate
    n_per_segment = int(segment_length * sampling_rate)
    n_step = int(step * sampling_rate)

    for i in range(0, len(trace_pairs[0][0]) - n_per_segment + 1, n_step):
        seg_x = [x[i:i+n_per_segment] for x, _ in trace_pairs]
        seg_y = [y[i:i+n_per_segment] for _, y in trace_pairs]

        for sx, sy in zip(seg_x, seg_y):
            if len(sx) < 10: continue
            _, ccf = compute_cross_correlation(sx, sy, sampling_rate, **kwargs)
            ccfs.append(ccf)

    if not ccfs:
        raise ValueError("No valid segments generated for stacking")

    # 使用 stacking 模块统一叠加
    final_ccf = stack_ccfs(ccfs, method=method)
    # 假设所有 CCF 具有相同的 lags
    return np.linspace(-len(final_ccf)//2, len(final_ccf)//2, len(final_ccf)) / sampling_rate, final_ccf
