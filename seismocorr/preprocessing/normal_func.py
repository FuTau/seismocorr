# seismocorr/preprocessing/norm_func.py

"""
Unified Preprocessing Toolkit

提供完整的信号预处理功能，适用于地震背景噪声互相关分析。
支持：
- 趋势移除（detrend, demean）
- 滤波（带通、低通、高通）
- 时域 / 频域归一化
- 分段 + FFT 流水线
- 批量处理接口

设计原则：
    - 函数式接口为主，便于组合
    - 支持配置驱动（config['filter'] = 'bandpass'）
    - 内存友好，支持 chunked 处理
"""

import numpy as np
from typing import Dict, Any, Union, Optional, List
from scipy.signal import butter, filtfilt, detrend as scipy_detrend

from seismocorr.preprocessing.freq_norm import get_freq_normalizer
from seismocorr.preprocessing.time_norm import get_time_normalizer


# =============================================================================
# 🛠 基础预处理函数
# =============================================================================

def demean(x: np.ndarray) -> np.ndarray:
    """去除均值"""
    return x - np.mean(x)


def detrend(x: np.ndarray, type: str = 'linear') -> np.ndarray:
    """
    去除趋势

    Args:
        x: 输入数组
        type: 'constant'（去均值）、'linear'（去线性趋势）

    Returns:
        去趋势后的数组
    """
    return scipy_detrend(x, type=type)


def taper(x: np.ndarray, width: float = 0.05) -> np.ndarray:
    """
    对信号加窗（汉宁窗），减少边缘效应

    Args:
        x: 输入数组
        width: 窗口比例（默认首尾 5% 加窗）

    Returns:
        加窗后的数组
    """
    window = int(len(x) * width)
    if window == 0:
        return x.copy()
    y = x.copy()
    y[:window] *= np.hanning(2 * window)[:window]
    y[-window:] *= np.hanning(2 * window)[window:]
    return y


# =============================================================================
# 🔧 滤波函数
# =============================================================================

def _butter_filter(
    data: np.ndarray,
    sampling_rate: float,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """
    通用 Butterworth 滤波器

    Args:
        data: 输入时间序列
        sampling_rate: 采样率 (Hz)
        freq_min: 高通频率（Hz）
        freq_max: 低通频率（Hz）
        order: 滤波阶数
        zero_phase: 是否零相位滤波（前后各一次）

    Returns:
        滤波后的时间序列
    """
    nyquist = sampling_rate / 2.0

    # 设计滤波器
    if freq_min and freq_max:
        btype = 'band'
        critical = [freq_min / nyquist, freq_max / nyquist]
    elif freq_min:
        btype = 'high'
        critical = [freq_min / nyquist]
    elif freq_max:
        btype = 'low'
        critical = [freq_max / nyquist]
    else:
        return data.copy()  # 无滤波要求

    # 防止超 Nyquist
    critical = [c for c in critical if c < 1.0]
    if not critical:
        return data.copy()

    b, a = butter(order, critical, btype=btype)

    filtered = filtfilt(b, a, data) if zero_phase else np.apply_along_axis(lambda x: np.convolve(x, b, mode='same'), 0, data)
    return filtered


def bandpass(
    x: np.ndarray,
    fmin: float,
    fmax: float,
    sr: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """带通滤波"""
    return _butter_filter(x, sr, freq_min=fmin, freq_max=fmax, order=order, zero_phase=zero_phase)


def lowpass(
    x: np.ndarray,
    fmax: float,
    sr: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """低通滤波"""
    return _butter_filter(x, sr, freq_max=fmax, order=order, zero_phase=zero_phase)


def highpass(
    x: np.ndarray,
    fmin: float,
    sr: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """高通滤波"""
    return _butter_filter(x, sr, freq_min=fmin, order=order, zero_phase=zero_phase)


# =============================================================================
# 🏭 归一化方法注册表（工厂模式）
# =============================================================================

# --- 延迟导入 ---
def _get_time_norm_registry():
    from seismocorr.preprocessing.time_norm import (
        ZScoreNormalizer,
        OneBitNormalizer,
        RMSNormalizer,
        NoTimeNorm,
    )
    return {
        'zscore': ZScoreNormalizer,
        'one-bit': OneBitNormalizer,
        'rms': RMSNormalizer,
        'no': NoTimeNorm,
    }


def _get_freq_norm_registry():
    from seismocorr.preprocessing.freq_norm import (
        SpectralWhitening,
        RmaFreqNorm,
        NoFreqNorm,
    )
    return {
        'whiten': lambda win=20: SpectralWhitening(smooth_win=win),
        'rma': lambda alpha=0.9: RmaFreqNorm(alpha=alpha),
        'no': NoFreqNorm,
    }





# =============================================================================
# 🧰 高级工具函数
# =============================================================================

def apply_preprocessing(
    x: np.ndarray,
    sampling_rate: float,
    detrend_type: Optional[str] = 'linear',
    taper_width: Optional[float] = 0.05,
    filter_type: Optional[str] = None,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    time_norm: Optional[str] = None,
    freq_norm: Optional[str] = None,
    **kwargs
) -> np.ndarray:
    """
    统一预处理流水线（推荐用于单道数据）

    Example:
        processed = apply_preprocessing(
            x=data,
            sampling_rate=100,
            detrend_type='linear',
            taper_width=0.05,
            filter_type='bandpass',
            freq_min=0.1,
            freq_max=1.0,
            time_norm='one-bit'
        )

    Returns:
        处理后的时间序列
    """
    y = x.astype(np.float64).copy()

    # 1. 去趋势
    if detrend_type:
        y = detrend(y, type=detrend_type)

    # 2. 加窗
    if taper_width and taper_width > 0:
        y = taper(y, width=taper_width)

    # 3. 滤波
    if filter_type == 'bandpass' and freq_min and freq_max:
        y = bandpass(y, freq_min, freq_max, sampling_rate)
    elif filter_type == 'lowpass' and freq_max:
        y = lowpass(y, freq_max, sampling_rate)
    elif filter_type == 'highpass' and freq_min:
        y = highpass(y, freq_min, sampling_rate)

    # 4. 归一化
    if time_norm:
        normalizer = get_time_normalizer(time_norm)
        y = normalizer(y)
    
    if freq_norm:
        normalizer = get_time_normalizer(freq_norm)
        y = normalizer(y)

    return y


def batch_preprocess_traces(
    traces: Dict[str, np.ndarray],
    sampling_rate: float,
    **config
) -> Dict[str, np.ndarray]:
    """
    批量预处理多个通道数据

    Args:
        traces: {channel_name: data_array}
        sampling_rate: 全局采样率
        **config: 同 apply_preprocessing 参数

    Returns:
        处理后的字典
    """
    return {
        name: apply_preprocessing(data, sampling_rate=sampling_rate, **config)
        for name, data in traces.items()
    }


def prepare_fft_segments(
    x: np.ndarray,
    segment_length: float,
    step: float,
    sampling_rate: float,
    max_lag_seconds: Optional[float] = None,
    **preprocessing_kwargs
) -> np.ndarray:
    """
    完整分段预处理流程：用于准备 FFT 输入

    Args:
        x: 原始时间序列
        segment_length: 段长（秒）
        step: 步长（秒）
        sampling_rate: 采样率
        max_lag_seconds: 可选，限制最大滞后以控制 nfft
        **preprocessing_kwargs: 传递给 apply_preprocessing 的参数

    Returns:
        shape=(n_windows, n_freqs//2) 的复数数组
    """
    from scipy.fftpack import fft

    seg_len_samp = int(segment_length * sampling_rate)
    step_samp = int(step * sampling_rate)

    segments = []
    for start in range(0, len(x) - seg_len_samp + 1, step_samp):
        seg = x[start:start + seg_len_samp]
        # 应用完整预处理
        processed_seg = apply_preprocessing(seg, sampling_rate=sampling_rate, **preprocessing_kwargs)
        segments.append(processed_seg)

    if not segments:
        return np.empty((0, 0), dtype=complex)

    # 转为数组并进行 FFT
    arr = np.array(segments)
    N = arr.shape[-1]
    X = fft(arr, axis=-1)[..., :N // 2]

    # 频域归一化
    freq_norm_name = preprocessing_kwargs.pop("freq_norm", "no")
    if freq_norm_name != "no":
        freq_norm = get_freq_normalizer(freq_norm_name, **preprocessing_kwargs)
        X = freq_norm(X)

    return X


# =============================================================================
# 🔍 查询接口
# =============================================================================

def list_supported_operations() -> Dict[str, List[str]]:
    """列出所有支持的操作类型"""
    return {
        "detrend": ["linear", "constant"],
        "taper": ["hanning"],
        "filter": ["bandpass", "lowpass", "highpass"],
        "time_norm": list(_get_time_norm_registry().keys()),
        "freq_norm": list(_get_freq_norm_registry().keys()),
    }
