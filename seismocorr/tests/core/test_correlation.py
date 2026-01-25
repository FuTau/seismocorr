import pytest
import numpy as np
import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from seismocorr.core.correlation.correlation import (
    CorrelationEngine,
    BatchCorrelator,
    CorrelationConfig,
    SUPPORTED_METHODS
)


class TestCorrelationEngine:
    """测试主互相关计算函数"""
    
    def test_compute_cross_correlation_basic(self, sample_signal):
        """测试基本互相关计算"""
        x = sample_signal[:500]
        y = sample_signal[:500]
        
        # 创建配置
        config = CorrelationConfig(
            method='time-domain',
            time_normalize='one-bit',
            freq_normalize='no',
            max_lag=1
        )
        
        # 创建引擎并计算
        engine = CorrelationEngine(config)
        lags, ccf = engine.compute_cross_correlation(x, y, sampling_rate=100.0)
        
        assert len(lags) == len(ccf)
        assert lags[0] < 0 < lags[-1]  # 滞后应该对称
    
    def test_all_supported_methods(self, sample_signal):
        """测试所有支持的互相关方法"""
        x = sample_signal[:500]
        y = sample_signal[:500]
        
        for method in SUPPORTED_METHODS:
            # 创建配置
            config = CorrelationConfig(
                method=method,
                time_normalize='no',
                freq_normalize='no',
                max_lag=1
            )
            
            # 创建引擎并计算
            engine = CorrelationEngine(config)
            lags, ccf = engine.compute_cross_correlation(x, y, sampling_rate=100.0)
            
            assert len(lags) == len(ccf)
            assert not np.any(np.isnan(ccf))
            assert not np.any(np.isinf(ccf))
    
    def test_bandpass_filtering(self, multi_freq_signal):
        """测试带通滤波"""
        x = multi_freq_signal[:1000]
        y = multi_freq_signal[:1000]
        
        # 创建配置
        config = CorrelationConfig(
            method='time-domain',
            freq_band=(5.0, 15.0)  # 5-15Hz带通
        )
        
        # 创建引擎并计算
        engine = CorrelationEngine(config)
        lags, ccf = engine.compute_cross_correlation(x, y, sampling_rate=100.0)
        
        assert len(ccf) > 0
    
    
    def test_edge_cases(self, edge_case_signals):
        """测试边界情况"""
        # 空信号
        empty = edge_case_signals['empty']
        
        engine = CorrelationEngine()
        lags, ccf = engine.compute_cross_correlation(empty, empty, sampling_rate=100.0)
        assert len(ccf) == 0
        
        # 单点信号
        single = edge_case_signals['single_point']
        lags, ccf = engine.compute_cross_correlation(single, single, sampling_rate=100.0)
        assert len(ccf) > 0
        
        # 小信号
        small = edge_case_signals['small_signal']
        lags, ccf = engine.compute_cross_correlation(small, small, sampling_rate=100.0)
        assert len(ccf) > 0
    
    def test_different_length_signals(self, sample_signal):
        """测试不同长度信号"""
        x = sample_signal[:300]
        y = sample_signal[:500]  # 不同长度
        
        engine = CorrelationEngine()
        lags, ccf = engine.compute_cross_correlation(x, y, sampling_rate=100.0)
        
        # 应该自动截断到相同长度
        assert len(ccf) > 0


class TestBatchCorrelator:
    """测试批量互相关计算"""
    
    def test_batch_cross_correlation_basic(self, batch_traces, station_pairs):
        """测试基本批量互相关"""
        # 创建配置
        config = CorrelationConfig(method='time-domain')
        
        # 创建批量计算器并计算
        batch_correlator = BatchCorrelator()
        lags, ccfs, keys = batch_correlator.batch_cross_correlation(
            traces=batch_traces,
            pairs=station_pairs,
            sampling_rate=100.0,
            config=config
        )
        
        # 检查结果
        assert len(keys) == len(station_pairs)
        assert ccfs.shape[0] == len(keys)
        assert ccfs.shape[1] == len(lags)
        
        for i, key in enumerate(keys):
            assert len(lags) == len(ccfs[i])
            assert len(ccfs[i]) > 0
            assert '--' in key  # 键名应该包含分隔符
    
    def test_batch_cross_correlation_missing_traces(self, batch_traces):
        """测试缺失轨迹的情况"""
        # 包含不存在的轨迹对
        pairs = [
            ('NET.STA00.CHZ', 'NET.STA01.CHZ'),
            ('NET.STA00.CHZ', 'MISSING.CHZ')  # 不存在的轨迹
        ]
        
        # 创建批量计算器并计算
        batch_correlator = BatchCorrelator()
        lags, ccfs, keys = batch_correlator.batch_cross_correlation(
            traces=batch_traces,
            pairs=pairs,
            sampling_rate=100.0
        )
        
        # 应该只包含有效的对
        assert len(keys) == 1
        assert ccfs.shape[0] == 1
        assert 'NET.STA00.CHZ--NET.STA01.CHZ' in keys
        assert 'NET.STA00.CHZ--MISSING.CHZ' not in keys
    
    def test_batch_cross_correlation_empty_input(self):
        """测试空输入"""
        traces = {}
        pairs = []
        
        # 创建批量计算器并计算
        batch_correlator = BatchCorrelator()
        lags, ccfs, keys = batch_correlator.batch_cross_correlation(
            traces=traces,
            pairs=pairs,
            sampling_rate=100.0
        )
        
        assert len(keys) == 0
        assert ccfs.shape == (0,)


class TestIntegration:
    """测试集成功能"""
    
    def test_delay_detection_integration(self, correlation_test_signals):
        """测试延迟检测集成"""
        signal1 = correlation_test_signals['signal1']
        signal2 = correlation_test_signals['signal2']
        sr = correlation_test_signals['sampling_rate']
        known_delay = correlation_test_signals['known_delay']
        
        # 创建配置
        config = CorrelationConfig(
            method='time-domain',
            max_lag=0.2
        )
        
        # 创建引擎并计算
        engine = CorrelationEngine(config)
        lags, ccf = engine.compute_cross_correlation(signal1, signal2, sampling_rate=sr)
        
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
        
        # 创建配置
        config = CorrelationConfig(
            method='time-domain',
            time_normalize='zscore',  # 归一化有助于噪声抑制
            max_lag=1.0
        )
        
        # 创建引擎并计算
        engine = CorrelationEngine(config)
        lags, ccf = engine.compute_cross_correlation(x, y, sampling_rate=100.0)
        
        # 即使有噪声，也应该能检测到延迟
        max_idx = np.argmax(ccf)
        detected_delay = lags[max_idx]
        
        # 期望延迟为0.1秒（100个样本@1000Hz中的100个样本延迟）
        expected_delay = 1
        assert abs(detected_delay + expected_delay) < 0.2  # 允许较大误差


if __name__ == "__main__":
    pytest.main([__file__, "-v"])