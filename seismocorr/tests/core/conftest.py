import pytest
import numpy as np
from typing import Dict, Any, List
import tempfile
import os

# 添加现有的fixture
@pytest.fixture
def sample_signal() -> np.ndarray:
    """生成测试用的示例信号"""
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    signal += 0.1 * np.random.randn(len(t))
    return signal

# 添加叠加测试专用的fixture
@pytest.fixture
def ccf_list() -> List[np.ndarray]:
    """生成测试用的CCF列表"""
    np.random.seed(42)  # 确保可重复性
    n_ccfs = 10
    ccf_length = 100
    
    # 创建基于相同信号的CCF，添加不同噪声
    t = np.linspace(0, 1, ccf_length)
    base_signal = np.exp(-0.5 * (t - 0.5) ** 2 / 0.1 ** 2)  # 高斯脉冲
    
    ccf_list = []
    for i in range(n_ccfs):
        noise_level = 0.1 + 0.05 * i  # 递增的噪声水平
        noisy_signal = base_signal + noise_level * np.random.randn(ccf_length)
        ccf_list.append(noisy_signal)
    
    return ccf_list

@pytest.fixture
def single_ccf() -> np.ndarray:
    """生成单个CCF"""
    t = np.linspace(0, 1, 100)
    return np.exp(-0.5 * (t - 0.5) ** 2 / 0.1 ** 2)

@pytest.fixture
def ccf_list_with_outliers() -> List[np.ndarray]:
    """生成包含异常值的CCF列表"""
    np.random.seed(42)
    n_ccfs = 10
    ccf_length = 100
    
    t = np.linspace(0, 1, ccf_length)
    base_signal = np.exp(-0.5 * (t - 0.5) ** 2 / 0.1 ** 2)
    
    ccf_list = []
    for i in range(n_ccfs):
        if i == 5:  # 第6个CCF添加异常值
            noisy_signal = base_signal + 10.0 * np.random.randn(ccf_length)  # 强噪声
        else:
            noisy_signal = base_signal + 0.1 * np.random.randn(ccf_length)
        ccf_list.append(noisy_signal)
    
    return ccf_list

@pytest.fixture
def identical_ccfs() -> List[np.ndarray]:
    """生成完全相同的CCF列表"""
    t = np.linspace(0, 1, 100)
    base_signal = np.sin(2 * np.pi * 5 * t)
    
    return [base_signal.copy() for _ in range(5)]

@pytest.fixture
def ccf_list_with_negatives() -> List[np.ndarray]:
    """生成包含负值的CCF列表"""
    np.random.seed(42)
    n_ccfs = 5
    ccf_length = 100
    
    t = np.linspace(0, 1, ccf_length)
    base_signal = np.sin(2 * np.pi * 5 * t)  # 包含负值的信号
    
    ccf_list = []
    for i in range(n_ccfs):
        noisy_signal = base_signal + 0.1 * np.random.randn(ccf_length)
        ccf_list.append(noisy_signal)
    
    return ccf_list

@pytest.fixture
def coherent_ccfs() -> List[np.ndarray]:
    """生成高相干性的CCF列表"""
    np.random.seed(42)
    n_ccfs = 10
    ccf_length = 100
    
    t = np.linspace(0, 1, ccf_length)
    base_signal = np.exp(-0.5 * (t - 0.5) ** 2 / 0.1 ** 2)
    
    ccf_list = []
    for i in range(n_ccfs):
        # 添加很小的噪声，保持高相干性
        noisy_signal = base_signal + 0.01 * np.random.randn(ccf_length)
        ccf_list.append(noisy_signal)
    
    return ccf_list

@pytest.fixture
def variable_length_ccfs() -> List[np.ndarray]:
    """生成不同长度的CCF列表"""
    np.random.seed(42)
    lengths = [80, 100, 120, 90, 110]
    
    ccf_list = []
    for length in lengths:
        t = np.linspace(0, 1, length)
        signal = np.exp(-0.5 * (t - 0.5) ** 2 / 0.1 ** 2)
        ccf_list.append(signal)
    
    return ccf_list

@pytest.fixture
def short_ccfs() -> List[np.ndarray]:
    """生成短CCF列表"""
    np.random.seed(42)
    n_ccfs = 5
    ccf_length = 20  # 很短的信号
    
    t = np.linspace(0, 1, ccf_length)
    base_signal = np.sin(2 * np.pi * 5 * t)
    
    ccf_list = []
    for i in range(n_ccfs):
        noisy_signal = base_signal + 0.1 * np.random.randn(ccf_length)
        ccf_list.append(noisy_signal)
    
    return ccf_list

@pytest.fixture
def ccf_list_with_nan() -> List[np.ndarray]:
    """生成包含NaN的CCF列表"""
    np.random.seed(42)
    n_ccfs = 5
    ccf_length = 100
    
    t = np.linspace(0, 1, ccf_length)
    base_signal = np.exp(-0.5 * (t - 0.5) ** 2 / 0.1 ** 2)
    
    ccf_list = []
    for i in range(n_ccfs):
        noisy_signal = base_signal + 0.1 * np.random.randn(ccf_length)
        if i == 2:  # 第3个CCF包含NaN
            noisy_signal[50:60] = np.nan
        ccf_list.append(noisy_signal)
    
    return ccf_list

@pytest.fixture
def ccf_list_with_inf() -> List[np.ndarray]:
    """生成包含无穷大的CCF列表"""
    np.random.seed(42)
    n_ccfs = 5
    ccf_length = 100
    
    t = np.linspace(0, 1, ccf_length)
    base_signal = np.exp(-0.5 * (t - 0.5) ** 2 / 0.1 ** 2)
    
    ccf_list = []
    for i in range(n_ccfs):
        noisy_signal = base_signal + 0.1 * np.random.randn(ccf_length)
        if i == 2:  # 第3个CCF包含无穷大
            noisy_signal[50] = np.inf
            noisy_signal[51] = -np.inf
        ccf_list.append(noisy_signal)
    
    return ccf_list

@pytest.fixture
def noisy_ccfs() -> List[np.ndarray]:
    """生成高噪声的CCF列表（用于测试信噪比改善）"""
    np.random.seed(42)
    n_ccfs = 20
    ccf_length = 200
    
    # 创建信号部分（前100个点）和噪声部分（后100个点）
    t_signal = np.linspace(0, 0.5, 100)
    signal = np.exp(-0.5 * (t_signal - 0.25) ** 2 / 0.05 ** 2)
    
    ccf_list = []
    for i in range(n_ccfs):
        # 前100个点：信号+噪声
        signal_part = signal + 0.5 * np.random.randn(100)
        # 后100个点：只有噪声
        noise_part = 0.5 * np.random.randn(100)
        full_signal = np.concatenate([signal_part, noise_part])
        ccf_list.append(full_signal)
    
    return ccf_list