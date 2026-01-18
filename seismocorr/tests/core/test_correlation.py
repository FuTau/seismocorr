import pytest
import numpy as np
import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from seismocorr.core.correlation import (
    compute_cross_correlation,
    batch_cross_correlation,
    _xcorr_time_domain,
    _xcorr_freq_domain,
    _coherency,
    _as_float_array,
    SUPPORTED_METHODS
)


class TestAsFloatArray:
    """测试数组转换函数"""
    
    def test_as_float_array_list_input(self):
        """测试列表输入"""
        input_list = [1.0, 2.0, 3.0]
        result = _as_float_array(input_list)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert np.array_equal(result, expected)
        assert result.dtype == np.float64
    
    def test_as_float_array_numpy_input(self):
        """测试numpy数组输入"""
        input_array = np.array([1, 2, 3], dtype=np.int32)
        result = _as_float_array(input_array)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert np.array_equal(result, expected)
        assert result.dtype == np.float64
    
    def test_as_float_array_2d_input(self):
        """测试二维数组输入（应该被展平）"""
        input_2d = np.array([[1, 2], [3, 4]])
        result = _as_float_array(input_2d)
        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        assert np.array_equal(result, expected)
    
    def test_as_float_array_empty(self):
        """测试空数组输入"""
        empty_array = np.array([])
        result = _as_float_array(empty_array)
        assert len(result) == 0
        assert result.dtype == np.float64


class TestXCorrTimeDomain:
    """测试时域互相关函数"""
    
    def test_xcorr_time_domain_basic(self, sample_signal):
        """测试基本时域互相关"""
        x = sample_signal[:500]
        y = sample_signal[100:600]  # 有延迟的版本
        
        sampling_rate = 100.0
        max_lag_seconds = 1.0  # 1秒，单位是秒
        
        lags, ccf = _xcorr_time_domain(x, y, sr=sampling_rate, max_lag=max_lag_seconds)
        
        # 检查输出形状
        assert len(lags) == len(ccf)
        expected_length = 2 * int(max_lag_seconds * sampling_rate) + 1
        assert len(lags) == expected_length
        
        # 检查滞后时间正确
        assert lags[0] == -max_lag_seconds  # 应该是-1.0秒
        assert lags[-1] == max_lag_seconds   # 应该是+1.0秒
        assert np.isclose(lags[len(lags)//2], 0.0)  # 零滞后在中间
    
    def test_xcorr_time_domain_identical_signals(self, sample_signal):
        """测试相同信号的互相关"""
        x = sample_signal[:500]
        y = x.copy()  # 完全相同
        
        sampling_rate = 100.0
        max_lag_seconds = 0.5  # 0.5秒，单位是秒
        
        lags, ccf = _xcorr_time_domain(x, y, sr=sampling_rate, max_lag=max_lag_seconds)
        
        # 相同信号应该在零滞后处有最大值
        zero_lag_idx = len(ccf) // 2
        assert np.argmax(ccf) == zero_lag_idx
        assert ccf[zero_lag_idx] > 0.9
    
    def test_xcorr_time_domain_shifted_signals(self):
        """测试有延迟的信号互相关"""
        t = np.linspace(0, 1, 1000)
        x = np.sin(2 * np.pi * 5 * t)
        
        # 创建有延迟的信号
        delay_seconds = 0.5  # 0.5秒延迟
        delay_samples = int(delay_seconds * 100)  # 100Hz采样率
        y = np.roll(x, delay_samples)
        
        # 使用秒作为max_lag单位
        lags, ccf = _xcorr_time_domain(x, y, sr=100.0, max_lag=1.0)  # max_lag=1秒
        
        # 最大值应该在延迟处
        max_idx = np.argmax(ccf)
        expected_lag_seconds = delay_seconds
        actual_lag_seconds = lags[max_idx]  # 直接使用秒数
        
        # 允许少量误差（由于边界效应）
        # 注意：这里比较的是秒数，不是样本数
        assert abs(actual_lag_seconds - expected_lag_seconds) <= 0.02  # 20毫秒误差


    def test_xcorr_time_domain_small_max_lag(self, sample_signal):
        """测试小max_lag值"""
        x = sample_signal[:200]
        y = sample_signal[:200]
        
        # 使用秒作为max_lag单位
        lags, ccf = _xcorr_time_domain(x, y, sr=100.0, max_lag=0.1)  # max_lag=0.1秒
        
        # 计算期望的长度：0.1秒 * 100Hz = 10个样本
        # 从-10到+10，总共21个点
        expected_length = 2 * int(0.1 * 100) + 1  # 21
        
        assert len(lags) == expected_length
        assert lags[0] == -0.1  # -max_lag秒
        assert lags[-1] == 0.1   # +max_lag秒


class TestXCorrFreqDomain:
    """测试频域互相关函数"""
    
    def test_xcorr_freq_domain_basic(self, sample_signal):
        """测试基本频域互相关"""
        x = sample_signal[:500]
        y = sample_signal[:500]
        
        lags, ccf = _xcorr_freq_domain(x, y, sr=100.0, max_lag=0.5, nfft=None)
        
        # 检查输出形状
        assert len(lags) == len(ccf)
        assert len(lags) == 101  # 从 -50 到 +50
        
        # 检查滞后时间正确
        assert lags[0] == -0.5  # -max_lag/sr
        assert lags[-1] == 0.5  # +max_lag/sr
    
    def test_xcorr_freq_domain_with_nfft(self, sample_signal):
        """测试指定nfft的频域互相关"""
        x = sample_signal[:500]
        y = sample_signal[:500]
        
        # 测试不同的nfft值
        for nfft in [512, 1024, None]:
            lags, ccf = _xcorr_freq_domain(x, y, sr=100.0, max_lag=0.5, nfft=nfft)
            assert len(lags) == 101
            assert len(ccf) == 101
    
    def test_xcorr_freq_domain_deconvolution(self, sample_signal):
        """测试反卷积模式的频域互相关"""
        x = sample_signal[:500]
        y = sample_signal[:500]
        
        lags, ccf_deconv = _xcorr_freq_domain(x, y, sr=100.0, max_lag=0.5, nfft=None, deconv=True)
        lags, ccf_normal = _xcorr_freq_domain(x, y, sr=100.0, max_lag=0.5, nfft=None, deconv=False)
        
        # 反卷积结果应该与普通互相关不同
        assert not np.allclose(ccf_deconv, ccf_normal)


class TestCoherency:
    """测试相干性加权互相关"""
    
    def test_coherency_basic(self, sample_signal):
        """测试基本相干性计算"""
        x = sample_signal[:500]
        y = sample_signal[:500]
        
        lags, ccf = _coherency(x, y, sr=100.0, max_lag=0.5, nfft=None)
        
        assert len(lags) == 101
        assert len(ccf) == 101
    
    def test_coherency_vs_normal(self, sample_signal):
        """测试相干性加权与普通互相关的区别"""
        x = sample_signal[:500]
        y = sample_signal[50:550]  # 有延迟的版本
        
        lags_coherency, ccf_coherency = _coherency(x, y, sr=100.0, max_lag=0.5, nfft=None)
        lags_normal, ccf_normal = _xcorr_freq_domain(x, y, sr=100.0, max_lag=0.5, nfft=None)
        
        # 相干性加权结果应该与普通互相关不同
        assert not np.array_equal(ccf_coherency, ccf_normal)


class TestComputeCrossCorrelation:
    """测试主互相关计算函数"""
    
    def test_compute_cross_correlation_basic(self, sample_signal):
        """测试基本互相关计算"""
        x = sample_signal[:500]
        y = sample_signal[:500]
        
        lags, ccf = compute_cross_correlation(
            x, y, sampling_rate=100.0,max_lag=1,
            method='time-domain',
            time_normalize='one-bit',
            freq_normalize='no'
        )
        
        assert len(lags) == len(ccf)
        assert lags[0] < 0 < lags[-1]  # 滞后应该对称
    
    def test_all_supported_methods(self, sample_signal):
        """测试所有支持的互相关方法"""
        x = sample_signal[:500]
        y = sample_signal[:500]
        
        for method in SUPPORTED_METHODS:
            lags, ccf = compute_cross_correlation(
                x, y, sampling_rate=100.0,max_lag=1,
                method=method,
                time_normalize='no',
                freq_normalize='no'
            )
            
            assert len(lags) == len(ccf)
            assert not np.any(np.isnan(ccf))
            assert not np.any(np.isinf(ccf))
    
    def test_bandpass_filtering(self, multi_freq_signal):
        """测试带通滤波"""
        x = multi_freq_signal[:1000]
        y = multi_freq_signal[:1000]
        
        lags, ccf = compute_cross_correlation(
            x, y, sampling_rate=100.0,max_lag=1,
            method='time-domain',
            freq_band=(5.0, 15.0)  # 5-15Hz带通
        )
        
        assert len(ccf) > 0
    
    
    def test_edge_cases(self, edge_case_signals):
        """测试边界情况"""
        # 空信号
        empty = edge_case_signals['empty']
        lags, ccf = compute_cross_correlation(empty, empty, sampling_rate=100.0,max_lag=1)
        assert len(ccf) == 0
        
        # 单点信号
        single = edge_case_signals['single_point']
        lags, ccf = compute_cross_correlation(single, single, sampling_rate=100.0,max_lag=1)
        assert len(ccf) > 0
        
        # 小信号
        small = edge_case_signals['small_signal']
        lags, ccf = compute_cross_correlation(small, small, sampling_rate=100.0,max_lag=1)
        assert len(ccf) > 0
    
    def test_different_length_signals(self, sample_signal):
        """测试不同长度信号"""
        x = sample_signal[:300]
        y = sample_signal[:500]  # 不同长度
        
        lags, ccf = compute_cross_correlation(x, y, sampling_rate=100.0,max_lag=1)
        
        # 应该自动截断到相同长度
        assert len(ccf) > 0


class TestBatchCrossCorrelation:
    """测试批量互相关计算"""
    
    def test_batch_cross_correlation_basic(self, batch_traces, station_pairs):
        """测试基本批量互相关"""
        result = batch_cross_correlation(
            traces=batch_traces,
            pairs=station_pairs,
            sampling_rate=100.0,
            method='time-domain'
        )
        
        # 检查结果字典
        assert len(result) == len(station_pairs)
        
        for key, (lags, ccf) in result.items():
            assert len(lags) == len(ccf)
            assert len(ccf) > 0
            assert '--' in key  # 键名应该包含分隔符
    
    def test_batch_cross_correlation_missing_traces(self, batch_traces):
        """测试缺失轨迹的情况"""
        # 包含不存在的轨迹对
        pairs = [
            ('NET.STA00.CHZ', 'NET.STA01.CHZ'),
            ('NET.STA00.CHZ', 'MISSING.CHZ')  # 不存在的轨迹
        ]
        
        result = batch_cross_correlation(batch_traces, pairs, sampling_rate=100.0)
        
        # 应该只包含有效的对
        assert len(result) == 1
        assert 'NET.STA00.CHZ--NET.STA01.CHZ' in result
        assert 'NET.STA00.CHZ--MISSING.CHZ' not in result
    
    def test_batch_cross_correlation_empty_input(self):
        """测试空输入"""
        traces = {}
        pairs = []
        
        result = batch_cross_correlation(traces, pairs, sampling_rate=100.0)
        assert len(result) == 0


class TestIntegration:
    """测试集成功能"""
    
    def test_delay_detection_integration(self, correlation_test_signals):
        """测试延迟检测集成"""
        signal1 = correlation_test_signals['signal1']
        signal2 = correlation_test_signals['signal2']
        sr = correlation_test_signals['sampling_rate']
        known_delay = correlation_test_signals['known_delay']
        
        lags, ccf = compute_cross_correlation(
            signal1, signal2, sampling_rate=sr,
            method='time-domain',
            max_lag=0.2
        )
        
        # 找到最大相关的位置
        max_idx = np.argmax(ccf)
        detected_delay = lags[max_idx]
        
        # 检测到的延迟应该接近已知延迟
        assert abs(detected_delay + known_delay) < 0.02  # 允许20毫秒误差
    
    def test_signal_to_noise_ratio(self):
        """测试信噪比情况下的互相关"""
        t = np.linspace(0, 2, 2000)
        clean_signal = np.sin(2 * np.pi * 5 * t)
        
        # 添加噪声
        np.random.seed(42)
        noise = 0.05 * np.random.randn(len(clean_signal))
        
        x = clean_signal[:1000] + noise[:1000]
        y = clean_signal[100:1100] + noise[100:1100]  # 有延迟的带噪版本
        
        lags, ccf = compute_cross_correlation(
            x, y, sampling_rate=100.0,
            method='time-domain',
            time_normalize='zscore',  # 归一化有助于噪声抑制
            max_lag=1.0
        )
        
        # 即使有噪声，也应该能检测到延迟
        max_idx = np.argmax(ccf)
        detected_delay = lags[max_idx]
        
        # 期望延迟为0.1秒（100个样本@1000Hz中的100个样本延迟）
        expected_delay = 1
        assert abs(detected_delay + expected_delay) < 0.2  # 允许较大误差


if __name__ == "__main__":
    pytest.main([__file__, "-v"])