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


@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def das_data_ct(rng):
    """(channels, time) : 32x2000, 含一个共模分量 + 随机噪声 + 一个相干波场"""
    n_ch, n_t = 32, 2000
    t = np.arange(n_t)

    cm = 2.0 * np.sin(2 * np.pi * 0.01 * t)  # 共模（随时间变化）
    noise = 0.2 * rng.standard_normal((n_ch, n_t))

    # 一个简单相干信号：沿通道有线性相位差（像传播波）
    phase = np.linspace(0, 2 * np.pi, n_ch)[:, None]
    wave = 0.5 * np.sin(2 * np.pi * 0.03 * t)[None, :] * np.cos(phase)

    x = noise + wave + cm[None, :]
    return x.astype(np.float64)


@pytest.fixture
def das_data_tc(das_data_ct):
    """(time, channels)"""
    return das_data_ct.T.copy()


@pytest.fixture
def das_with_naninf(das_data_ct):
    x = das_data_ct.copy()
    x[0, 10] = np.nan
    x[1, 20] = np.inf
    return x


@pytest.fixture
def dt_dx():
    return 0.01, 10.0  # dt=0.01s, dx=10m

@pytest.fixture
def waterlevel_params():
    # Fs=100Hz, win_length=1s -> 100 samples per window
    return dict(Fs=100.0, win_length=1.0, water_level_factor=1.0, n_iter=2, eps=1e-10)


@pytest.fixture
def waterlevel_signal():
    """
    构造一个“局部能量异常高”的信号：
    - 第1窗：小噪声
    - 第2窗：强能量（更大幅度）
    - 第3窗：小噪声
    """
    Fs = 100
    win_n = 100
    rng = np.random.default_rng(0)

    w1 = 0.2 * rng.standard_normal(win_n)
    w2 = 5.0 * rng.standard_normal(win_n)   # 强能量窗口
    w3 = 0.2 * rng.standard_normal(win_n)

    x = np.concatenate([w1, w2, w3]).astype(float)
    return x


@pytest.fixture
def cwt_params():
    """
    CWTSoftThreshold1D 需要 fs 和 noise_idx。
    noise_idx 设为前 0.5 秒（纯噪声段）。
    """
    Fs = 100.0
    n = 200
    noise_n = 200  # 1s=200 samples, 这里取 0.5s=100 也行；取 200 稍稳一些
    return dict(
        Fs=Fs,
        n=n,
        noise_idx=slice(0, noise_n),
        wavelet="cmor1.5-1.0",
        voices_per_octave=8,   # 降低一点，避免测试太慢
        quantile=0.99,
        f_min=1.0,
        f_max=Fs / 2,
        normalize=True,
        eps=1e-12,
    )


@pytest.fixture
def cwt_signal(cwt_params):
    """
    构造信号：
    - 前段：纯噪声（作为 noise_idx）
    - 后段：噪声 + 正弦信号 + 一个尖峰（用于 designal 测试）
    """
    fs = cwt_params["Fs"]
    n = cwt_params["n"]
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)

    noise = 0.3 * rng.standard_normal(n)
    sine = 1.0 * np.sin(2 * np.pi * 5.0 * t)  # 5Hz 正弦
    x = noise.copy()
    x[n // 2 :] += sine[n // 2 :]

    # 加一个尖峰（瞬态）
    x[int(0.75 * n)] += 10.0
    return x.astype(float)

@pytest.fixture
def powerlaw_params():
    return {"alpha": 0.5}

@pytest.fixture
def clipwhiten_params():
    return {"smooth_win": 20, "min_weight": 0.1, "max_weight": 10.0}

@pytest.fixture
def bandwise_params():
    # 覆盖两个频带：低频 + 中高频
    return {
        "bands": [(5.0, 30.0), (80.0, 150.0)],
        "Fs": 1000.0,
        "method": "rms",
    }

@pytest.fixture
def refspectrum_params(multi_freq_signal):
    # 用“观测谱本身”作为参考谱：理论上应近似恢复原信号（权重≈1）
    ref = np.abs(np.fft.fft(multi_freq_signal))
    return {"ref_spectrum": ref}