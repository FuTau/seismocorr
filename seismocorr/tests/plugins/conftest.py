import pytest
import numpy as np
from typing import Dict, Any, Tuple

from seismocorr.plugins.disper import DispersionConfig

# 添加现有的fixture
@pytest.fixture
def sample_signal() -> np.ndarray:
    """生成测试用的示例信号"""
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    signal += 0.1 * np.random.randn(len(t))
    return signal

# 添加频散成像专用的fixture
@pytest.fixture
def dispersion_test_data() -> Dict[str, Any]:
    """生成频散成像测试数据"""
    np.random.seed(42)
    
    # 频率数组
    n_freq = 100
    f = np.linspace(0.1, 10.0, n_freq)
    
    # 台站距离数组
    n_stations = 10
    dist = np.linspace(100, 1000, n_stations)
    
    # 模拟频域互相关数据
    # 创建具有频散特性的数据
    cc_array_f = np.zeros((n_stations, n_freq), dtype=complex)
    
    for i, freq in enumerate(f):
        if freq == 0:
            continue
            
        # 模拟相速度频散：频率越高，速度越快
        phase_velocity = 1000 + 500 * (freq / 10.0)  # 1000-1500 m/s
        
        for j, distance in enumerate(dist):
            # 相位延迟
            phase = 2 * np.pi * freq * distance / phase_velocity
            # 振幅随距离衰减
            amplitude = np.exp(-distance / 2000)
            # 添加随机相位噪声
            phase_noise = 0.1 * np.random.randn()
            
            cc_array_f[j, i] = amplitude * np.exp(1j * (phase + phase_noise))
    
    return {
        'frequencies': f,
        'distances': dist,
        'cc_array_f': cc_array_f
    }

@pytest.fixture
def realistic_dispersion_data() -> Dict[str, Any]:
    """生成更真实的频散成像测试数据（模拟实际面波）"""
    np.random.seed(42)
    
    n_freq = 80
    n_stations = 12
    
    # 频率范围：0.5-8 Hz（典型面波频率）
    f = np.linspace(0.5, 8.0, n_freq)
    
    # 台站距离：50-600米
    dist = np.linspace(50, 600, n_stations)
    
    # 创建具有明显频散特性的数据
    cc_array_f = np.zeros((n_stations, n_freq), dtype=complex)
    
    for i, freq in enumerate(f):
        # 瑞利波频散曲线：低频低速，高频高速
        if freq < 2.0:
            phase_velocity = 800 + 200 * (freq / 2.0)  # 800-1000 m/s
        else:
            phase_velocity = 1000 + 500 * ((freq - 2.0) / 6.0)  # 1000-1500 m/s
        
        for j, distance in enumerate(dist):
            # 几何扩散衰减
            geometric_spreading = 1.0 / np.sqrt(distance)
            # 内在衰减
            intrinsic_attenuation = np.exp(-0.001 * distance * freq)
            # 总振幅
            amplitude = geometric_spreading * intrinsic_attenuation
            
            # 相位项
            phase = 2 * np.pi * freq * distance / phase_velocity
            # 随机噪声
            amplitude_noise = 0.05 * np.random.randn()
            phase_noise = 0.1 * np.random.randn()
            
            cc_array_f[j, i] = (amplitude + amplitude_noise) * np.exp(1j * (phase + phase_noise))
    
    return {
        'frequencies': f,
        'distances': dist,
        'cc_array_f': cc_array_f
    }

@pytest.fixture
def dispersion_configurations() -> Dict[str, DispersionConfig]:
    """生成不同的频散成像配置"""
    return {
        'default': DispersionConfig(),
        'high_freq': DispersionConfig(freqmin=1.0, freqmax=20.0, vmin=500.0, vmax=3000.0),
        'low_velocity': DispersionConfig(vmin=200.0, vmax=1000.0, vnum=80),
        'high_resolution': DispersionConfig(vnum=200)
    }