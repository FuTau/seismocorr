import pytest
import numpy as np
import sys
import os

from seismocorr.preprocessing.time_norm import (
    moving_ave,
    ZScoreNormalizer,
    OneBitNormalizer,
    RMSNormalizer,
    ClipNormalizer,
    NoTimeNorm,
    RAMNormalizer,
    get_time_normalizer,
    _TIME_NORM_MAP
)


class TestMovingAve:
    """测试移动平均函数"""
    
    def test_moving_ave_basic(self):
        """测试基本移动平均功能"""
        data = np.array([1, 2, 3, 4, 5])
        window = 3
        result = moving_ave(data, window)
        expected = np.array([4/3,2.0, 3.0, 4.0, 14/3])  # 简单移动平均
        assert result.shape == (5,)
        assert np.allclose(result, expected)
    
    def test_moving_ave_single_point(self):
        """测试单点输入"""
        data = np.array([5.0])
        result = moving_ave(data, 1)
        assert result.shape == (1,)
        assert result[0] == 5.0
    
    def test_moving_ave_window_larger_than_data(self):
        """测试窗口大于数据长度的情况"""
        data = np.array([1, 2, 3])
        result = moving_ave(data, 5)
        # 应该返回单个值或适当处理
        assert len(result) == len(data)


class TestZScoreNormalizer:
    """测试ZScore归一化"""
    
    def test_zscore_normalization(self, sample_signal):
        """测试ZScore归一化结果"""
        normalizer = ZScoreNormalizer()
        normalized = normalizer.apply(sample_signal)
        
        # 检查均值为0，标准差为1
        assert np.isclose(normalized.mean(), 0.0, atol=1e-10)
        assert np.isclose(normalized.std(), 1.0, atol=1e-10)
    
    def test_zscore_constant_signal(self, constant_signal):
        """测试常数信号的ZScore归一化"""
        normalizer = ZScoreNormalizer()
        normalized = normalizer.apply(constant_signal)
        
        # 常数信号归一化后应为全零
        assert np.allclose(normalized, 0.0)
    
    def test_zscore_zero_signal(self, zero_signal):
        """测试全零信号的ZScore归一化"""
        normalizer = ZScoreNormalizer()
        normalized = normalizer.apply(zero_signal)
        
        # 全零信号归一化后仍为全零
        assert np.allclose(normalized, 0.0)


class TestOneBitNormalizer:
    """测试1-bit归一化"""
    
    def test_onebit_normalization(self, sample_signal):
        """测试1-bit归一化结果"""
        normalizer = OneBitNormalizer()
        normalized = normalizer.apply(sample_signal)
        
        # 检查所有值为±1
        assert np.all(np.abs(normalized) == 1.0)
        assert set(np.unique(normalized)).issubset({-1.0, 1.0})
    
    def test_onebit_preserves_sign(self, sample_signal):
        """测试1-bit归一化保持原始符号"""
        normalizer = OneBitNormalizer()
        normalized = normalizer.apply(sample_signal)
        
        original_sign = np.sign(sample_signal)
        result_sign = np.sign(normalized)
        
        # 符号应该保持一致（除了零值）
        nonzero_mask = sample_signal != 0
        if np.any(nonzero_mask):
            assert np.array_equal(
                original_sign[nonzero_mask], 
                result_sign[nonzero_mask]
            )


class TestRMSNormalizer:
    """测试RMS归一化"""
    
    def test_rms_normalization(self, sample_signal):
        """测试RMS归一化结果"""
        normalizer = RMSNormalizer()
        normalized = normalizer.apply(sample_signal)
        
        # 检查RMS值为1
        rms = np.sqrt(np.mean(normalized ** 2))
        assert np.isclose(rms, 1.0, atol=1e-10)
    
    def test_rms_zero_signal(self, zero_signal):
        """测试全零信号的RMS归一化"""
        normalizer = RMSNormalizer()
        normalized = normalizer.apply(zero_signal)
        
        # 全零信号归一化后仍为全零
        assert np.allclose(normalized, 0.0)


class TestClipNormalizer:
    """测试截幅归一化"""
    
    def test_clip_normalization(self, outlier_signal):
        """测试截幅归一化结果"""
        clip_val = 3.0
        normalizer = ClipNormalizer(clip_val=clip_val)
        normalized = normalizer.apply(outlier_signal)
        
        # 检查所有值在[-clip_val, clip_val]范围内
        assert np.all(normalized >= -clip_val)
        assert np.all(normalized <= clip_val)
        
        # 检查异常值被正确截断
        assert np.max(normalized) <= clip_val
        assert np.min(normalized) >= -clip_val
    
    def test_clip_custom_value(self, outlier_signal):
        """测试自定义截断值"""
        custom_clip = 2.0
        normalizer = ClipNormalizer(clip_val=custom_clip)
        normalized = normalizer.apply(outlier_signal)
        
        assert np.all(normalized >= -custom_clip)
        assert np.all(normalized <= custom_clip)


class TestNoTimeNorm:
    """测试无操作归一化"""
    
    def test_no_normalization(self, sample_signal):
        """测试无操作归一化"""
        normalizer = NoTimeNorm()
        normalized = normalizer.apply(sample_signal)
        
        # 应该返回原始信号的副本
        assert np.array_equal(normalized, sample_signal)
        assert normalized is not sample_signal  # 应该是副本


class TestRAMNormalizer:
    """测试RAM归一化"""
    
    def test_ram_normalization_basic(self, sample_signal, ram_normalizer_params):
        """测试RAM归一化基本功能"""
        normalizer = RAMNormalizer(**ram_normalizer_params)
        normalized = normalizer.apply(sample_signal.copy())

        # 基本检查：输出应与输入同形状
        assert normalized.shape == sample_signal.shape
        assert not np.allclose(normalized, sample_signal)  # 应该有所改变
    
    def test_ram_normalizer_parameters(self):
        """测试RAM归一化器参数"""
        fmin, Fs, npts, norm_win = 2.0, 200.0, 1000, 0.5
        normalizer = RAMNormalizer(fmin, Fs, npts, norm_win)
        
        assert normalizer.fmin == fmin
        assert normalizer.Fs == Fs
        assert normalizer.npts == npts
        assert normalizer.norm_win == norm_win


class TestGetTimeNormalizer:
    """测试归一化器工厂函数"""
    
    def test_get_all_normalizers(self, ram_normalizer_params):
        """测试获取所有支持的归一化器"""
        for name in _TIME_NORM_MAP.keys():
            if name == 'ramn':
                normalizer = get_time_normalizer(name, **ram_normalizer_params)
            else:
                normalizer = get_time_normalizer(name)
            assert normalizer is not None
            assert hasattr(normalizer, 'apply')
    
    def test_get_normalizer_with_params(self):
        """测试带参数的归一化器获取"""
        # 测试ClipNormalizer带参数
        clip_normalizer = get_time_normalizer('clip', clip_val=2.5)
        assert isinstance(clip_normalizer, ClipNormalizer)
        assert clip_normalizer.clip_val == 2.5
        
        # 测试RAMNormalizer带参数
        ram_normalizer = get_time_normalizer('ramn', fmin=1.0, Fs=100.0, npts=1000)
        assert isinstance(ram_normalizer, RAMNormalizer)
        assert ram_normalizer.fmin == 1.0
    
    def test_get_normalizer_invalid_name(self):
        """测试无效归一化器名称"""
        with pytest.raises(ValueError, match="Unknown time normalization method"):
            get_time_normalizer('invalid_method')
    
    def test_ram_normalizer_missing_params(self):
        """测试RAM归一化器缺少必需参数"""
        with pytest.raises(ValueError, match="RAMNormalizer requires"):
            get_time_normalizer('ramn')  # 缺少必需参数
    
    def test_normalizer_callable_interface(self, sample_signal):
        """测试归一化器的可调用接口"""
        normalizer = get_time_normalizer('zscore')
        
        # 测试apply方法和__call__方法应该一致
        result1 = normalizer.apply(sample_signal)
        result2 = normalizer(sample_signal)
        
        assert np.array_equal(result1, result2)


class TestEdgeCases:
    """测试边界情况"""
    
    def test_empty_signal(self,ram_normalizer_params):
        """测试空信号"""
        empty_signal = np.array([])
        
        for name in ['zscore', 'one-bit', 'rms', 'clip', 'no', 'ramn']:
            if name == 'ramn':
                normalizer = get_time_normalizer(name, **ram_normalizer_params)
            else:
                normalizer = get_time_normalizer(name)
            result = normalizer.apply(empty_signal)
            assert len(result) == 0
    
    def test_single_point_signal(self,ram_normalizer_params):
        """测试单点信号"""
        single_point = np.array([5.0])
        
        for name in ['zscore', 'one-bit', 'rms', 'clip', 'no', 'ramn']:
            if name == 'ramn':
                normalizer = get_time_normalizer(name, **ram_normalizer_params)
            else:
                normalizer = get_time_normalizer(name)
            result = normalizer.apply(single_point)
            assert len(result) == 1
    
    def test_large_signal(self,ram_normalizer_params):
        """测试大信号（性能测试）"""
        large_signal = np.random.randn(100000)  # 10万个点
        
        for name in ['zscore', 'one-bit', 'rms', 'clip', 'no', 'ramn']:
            if name == 'ramn':
                normalizer = get_time_normalizer(name, **ram_normalizer_params)
            else:
                normalizer = get_time_normalizer(name)
            
            # 时间性能测试（可选）
            import time
            start_time = time.time()
            result = normalizer.apply(large_signal)
            end_time = time.time()
            
            assert len(result) == len(large_signal)
            assert end_time - start_time < 1.0  # 应该在1秒内完成


def test_normalizer_immutability(sample_signal,ram_normalizer_params):
    """测试归一化器不会修改原始信号"""
    original_copy = sample_signal.copy()
    
    for name in _TIME_NORM_MAP.keys():
        if name == 'ramn':  # 跳过需要参数的RAM归一化
            normalizer = get_time_normalizer(name, **ram_normalizer_params)
        else:   
            normalizer = get_time_normalizer(name)
        normalized = normalizer.apply(sample_signal)
        
        # 原始信号不应被修改
        assert np.array_equal(sample_signal, original_copy)
        assert normalized is not sample_signal  # 应该是新数组


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])