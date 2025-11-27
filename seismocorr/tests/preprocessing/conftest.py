import pytest
import numpy as np
from typing import Tuple, Dict, Any

@pytest.fixture
def sample_signal() -> np.ndarray:
    """生成测试用的示例信号"""
    t = np.linspace(0, 1, 1000)
    # 正弦波 + 噪声
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    signal += 0.1 * np.random.randn(len(t))
    return signal

@pytest.fixture
def constant_signal() -> np.ndarray:
    """生成常数信号（用于测试边界情况）"""
    return np.ones(100)

@pytest.fixture
def zero_signal() -> np.ndarray:
    """生成全零信号（用于测试边界情况）"""
    return np.zeros(100)

@pytest.fixture
def outlier_signal() -> np.ndarray:
    """生成包含异常值的信号"""
    signal = np.random.randn(100)
    signal[10] = 100.0  # 添加异常值
    signal[20] = -50.0  # 添加异常值
    return signal

@pytest.fixture
def ram_normalizer_params() -> Dict[str, Any]:
    """RAMNormalizer 参数"""
    return {
        'fmin': 1.0,
        'Fs': 100.0,
        'npts': 1000,
        'norm_win': 0.5
    }

@pytest.fixture
def multi_freq_signal() -> np.ndarray:
    """生成多频率信号（用于频域归一化测试）"""
    t = np.linspace(0, 2, 2000)
    # 包含多个频率成分的信号
    signal = (np.sin(2 * np.pi * 5 * t) + 
              0.5 * np.sin(2 * np.pi * 20 * t) + 
              0.2 * np.sin(2 * np.pi * 50 * t))
    return signal

@pytest.fixture
def bandwhiten_params() -> Dict[str, Any]:
    """BandWhitening 参数"""
    return {
        'freq_min': 10.0,
        'freq_max': 30.0,
        'Fs': 1000.0
    }

@pytest.fixture
def spectral_whiten_params() -> Dict[str, Any]:
    """SpectralWhitening 参数"""
    return {
        'smooth_win': 15
    }

@pytest.fixture
def rma_params() -> Dict[str, Any]:
    """RmaFreqNorm 参数"""
    return {
        'alpha': 0.95
    }

@pytest.fixture
def trend_signal() -> np.ndarray:
    """生成带趋势的信号"""
    t = np.linspace(0, 1, 1000)
    # 线性趋势 + 正弦波
    signal = 2.0 * t + np.sin(2 * np.pi * 5 * t)
    return signal

@pytest.fixture
def edge_heavy_signal() -> np.ndarray:
    """生成边缘效应明显的信号（用于加窗测试）"""
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * t)
    # 在边缘添加尖峰
    signal[0] = 10.0
    signal[-1] = -10.0
    return signal

@pytest.fixture
def bandpass_params() -> Dict[str, Any]:
    """带通滤波参数"""
    return {
        'fmin': 1.0,
        'fmax': 20.0,
        'sr': 100.0
    }

@pytest.fixture
def lowpass_params() -> Dict[str, Any]:
    """低通滤波参数"""
    return {
        'fmax': 15.0,
        'sr': 100.0
    }

@pytest.fixture
def highpass_params() -> Dict[str, Any]:
    """高通滤波参数"""
    return {
        'fmin': 5.0,
        'sr': 100.0
    }

@pytest.fixture
def taper_params() -> Dict[str, Any]:
    """加窗参数"""
    return {
        'width': 0.1
    }