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
from typing import Any, Union, Tuple, Optional, Dict, List
from scipy.signal import correlate, butter, filtfilt
from scipy.fftpack import fft, ifft

from seismocorr.preprocessing.time_norm import get_time_normalizer
from seismocorr.preprocessing.freq_norm import get_freq_normalizer
from seismocorr.preprocessing.normal_func import bandpass, detrend, demean, taper
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
    x = _as_float_array(x)
    y = _as_float_array(y)
    if len(x) == 0 or len(y) == 0:
        return np.array([]), np.array([])
    if not max_lag:
        max_lag = min(len(x), len(y)) / sampling_rate
    # 初始化参数字典
    time_norm_kwargs = time_norm_kwargs or {}
    freq_norm_kwargs = freq_norm_kwargs or {}
    x = detrend(x, type='linear')
    x = demean(x)
    x = taper(x, width=0.05)
    y = detrend(y, type='linear')
    y = demean(y)
    y = taper(y, width=0.05)
    # 参数预处理

    # 滤波
    if freq_band is not None:
        x = bandpass(x, freq_band[0],freq_band[1], sr=sampling_rate)
        y = bandpass(y, freq_band[0],freq_band[1], sr=sampling_rate)
    # 时域归一化 - 确保传递必要的参数
    time_norm_kwargs_with_fs = time_norm_kwargs.copy()
    if 'Fs' not in time_norm_kwargs_with_fs:
        time_norm_kwargs_with_fs['Fs'] = sampling_rate
    if 'npts' not in time_norm_kwargs_with_fs:
        time_norm_kwargs_with_fs['npts'] = len(x)  # 使用x的长度作为默认
    # 归一化
    normalizer = get_time_normalizer(time_normalize,**time_norm_kwargs_with_fs)
    x = normalizer.apply(x)
    y = normalizer.apply(y)
    # 频域归一化 - 确保传递必要的参数
    freq_norm_kwargs_with_fs = freq_norm_kwargs.copy()
    if 'Fs' not in freq_norm_kwargs_with_fs:
        freq_norm_kwargs_with_fs['Fs'] = sampling_rate
    normalizer = get_freq_normalizer(freq_normalize,**freq_norm_kwargs_with_fs)
    x = normalizer.apply(x)
    y = normalizer.apply(y)

    # 截断到相同长度
    min_len = min(len(x), len(y))
    x, y = x[:min_len], y[:min_len]

    # 选择方法
    if method == 'time-domain':
        lags, ccf = _xcorr_time_domain(x, y, sampling_rate, max_lag)
    elif method in ['freq-domain', 'deconv']:
        lags, ccf = _xcorr_freq_domain(x, y, sampling_rate, max_lag, nfft, deconv=method=='deconv')
    elif method == 'coherency':
        lags, ccf = _coherency(x, y, sampling_rate, max_lag, nfft)
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
            # print(a)
            # print(b)
            # print(traces.keys())
            continue
        try:
            lags, ccf = compute_cross_correlation(traces[a], traces[b], sampling_rate, **kwargs)
            # print(lags,ccf)
            # print(f"Computed CCF for pair: {a} - {b}")
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

def _xcorr_time_domain(x: np.ndarray, y: np.ndarray, sr: float, max_lag: float) -> LagsAndCCF:
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
    
    # 计算互相关
    ccf_full = correlate(x, y, mode='full')
    
    # 计算滞后对应的索引范围
    n = len(ccf_full)
    center = n // 2  # 零滞后对应的索引
    
    # 截取从 -max_lag_samples 到 +max_lag_samples 的部分
    start_idx = center - max_lag_samples
    end_idx = center + max_lag_samples + 1  # +1 确保包含max_lag_samples
    
    # 确保索引不越界
    start_idx = max(0, start_idx)
    end_idx = min(n, end_idx)
    
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

def _xcorr_freq_domain(x: np.ndarray, y: np.ndarray, sr: float, max_lag: float, nfft: Optional[int], deconv=False) -> LagsAndCCF:
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
    lag_in_samples = int(max_lag*sr)
    start = center - lag_in_samples
    end = center + lag_in_samples + 1
    lags = np.arange(-lag_in_samples, lag_in_samples + 1) / sr
    return lags, ccf_shifted[start:end]

def _coherency(x: np.ndarray, y: np.ndarray, sr: float, max_lag: int, nfft: Optional[int]) -> LagsAndCCF:
    """使用相干性作为权重的互相关（类似 PWS 的频域版本），提高互相关结果的可靠性"""
    lags, ccf_raw = _xcorr_freq_domain(x, y, sr, max_lag, nfft, deconv=False)
    
    # 在频域计算相位一致性
    Cxy = fft(ccf_raw)
    phase = np.exp(1j * np.angle(Cxy))
    coh = np.abs(np.mean(phase)) ** 4  # 权重
    return lags, ccf_raw * coh
