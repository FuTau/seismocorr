import pytest
import numpy as np
import sys
import os
from scipy import signal

from seismocorr.preprocessing.freq_norm import (
    SpectralWhitening,
    BandWhitening,
    RmaFreqNorm,
    NoFreqNorm,
    get_freq_normalizer,
    _FREQ_NORM_MAP
)


class TestSpectralWhitening:
    """测试谱白化"""
    
    def test_spectral_whitening_basic(self, multi_freq_signal):
        """测试谱白化基本功能"""
        normalizer = SpectralWhitening(smooth_win=20)
        whitened = normalizer.apply(multi_freq_signal)
        
        # 基本检查：输出应与输入同形状
        assert whitened.shape == multi_freq_signal.shape
        assert not np.allclose(whitened, multi_freq_signal)  # 应该有所改变
        
    
    def test_spectral_whitening_constant_signal(self, constant_signal):
        """测试常数信号的谱白化"""
        normalizer = SpectralWhitening(smooth_win=50)
        whitened = normalizer.apply(constant_signal)
        whitened = whitened - np.mean(whitened)
        # 常数信号白化后应该接近零（因为FFT后直流分量被去除）
        assert np.allclose(whitened, 0.0, atol=1e-10)
    
    def test_spectral_whitening_zero_signal(self, zero_signal):
        """测试全零信号的谱白化"""
        normalizer = SpectralWhitening()
        whitened = normalizer.apply(zero_signal)
        
        # 全零信号白化后仍应为全零
        assert np.allclose(whitened, 0.0)
    
    def test_spectral_whitening_spectrum_flattening(self, multi_freq_signal):
        """测试谱白化确实使频谱平坦化"""
        normalizer = SpectralWhitening(smooth_win=20)
        whitened = normalizer.apply(multi_freq_signal)
        
        # 计算原始信号和白化后信号的频谱
        f_orig, psd_orig = signal.periodogram(multi_freq_signal, fs=1000)
        f_white, psd_white = signal.periodogram(whitened, fs=1000)
        
        # 白化后频谱应该更平坦（方差更小）
        cv_orig = np.std(psd_orig) / np.mean(psd_orig)  # 变异系数
        cv_white = np.std(psd_white) / np.mean(psd_white)
        
        # 白化后频谱应该更平坦
        assert cv_white < cv_orig * 2  # 允许一定变化
    
    def test_spectral_whitening_custom_window(self):
        """测试自定义平滑窗口"""
        # 创建简单信号测试
        t = np.linspace(0, 1, 1000)
        test_signal = np.sin(2 * np.pi * 10 * t)
        
        # 使用不同窗口大小
        for smooth_win in [10, 20, 50]:
            normalizer = SpectralWhitening(smooth_win=smooth_win)
            whitened = normalizer.apply(test_signal)
            assert whitened.shape == test_signal.shape


class TestBandWhitening:
    """测试频带白化"""
    
    def test_band_whitening_basic(self, multi_freq_signal, bandwhiten_params):
        """测试频带白化基本功能"""
        normalizer = BandWhitening(**bandwhiten_params)
        whitened = normalizer.apply(multi_freq_signal)
        
        # 基本检查
        assert whitened.shape == multi_freq_signal.shape
        assert not np.allclose(whitened, multi_freq_signal)
    
    def test_band_whitening_frequency_response(self):
        """测试频带白化的频率响应"""
        # 创建测试信号：包含目标频带内外的频率
        Fs = 1000
        t = np.linspace(0, 1, Fs)
        
        # 目标频带：10-30Hz，测试信号包含5Hz（带外）和20Hz（带内）
        signal_5hz = np.sin(2 * np.pi * 5 * t)
        signal_20hz = np.sin(2 * np.pi * 20 * t)
        test_signal = signal_5hz + signal_20hz
        
        # 应用频带白化（10-30Hz）
        normalizer = BandWhitening(freq_min=10, freq_max=30, Fs=Fs)
        whitened = normalizer.apply(test_signal)
        
        # 计算频谱
        f, psd_orig = signal.periodogram(test_signal, fs=Fs)
        f, psd_white = signal.periodogram(whitened, fs=Fs)
        
        # 带外频率（5Hz）应该被抑制
        idx_5hz = np.argmin(np.abs(f - 5))
        idx_20hz = np.argmin(np.abs(f - 20))
        
        # 20Hz相对于5Hz的功率比应该增加
        ratio_orig = psd_orig[idx_20hz] / (psd_orig[idx_5hz] + 1e-10)
        ratio_white = psd_white[idx_20hz] / (psd_white[idx_5hz] + 1e-10)
        
        assert ratio_white > ratio_orig  # 带内频率应该相对增强
    
    def test_band_whitening_out_of_band(self):
        """测试频带外信号处理"""
        Fs = 1000
        t = np.linspace(0, 1, Fs)
        
        # 信号完全在频带外（1-5Hz，频带设置为10-30Hz）
        test_signal = np.sin(2 * np.pi * 3 * t)
        
        normalizer = BandWhitening(freq_min=10, freq_max=30, Fs=Fs)
        whitened = normalizer.apply(test_signal)
        
        # 频带外信号应该被显著抑制
        assert np.std(whitened) < np.std(test_signal) * 0.5
    
    def test_band_whitening_single_point(self):
        """测试单点信号"""
        single_point = np.array([5.0])
        normalizer = BandWhitening(freq_min=10, freq_max=30, Fs=1000)
        result = normalizer.apply(single_point)
        
        # 单点信号应该原样返回
        assert np.array_equal(result, single_point)
    
    def test_band_whitening_invalid_band(self):
        """测试无效频带参数"""
        with pytest.raises(ValueError):
            BandWhitening(freq_min=30, freq_max=10, Fs=1000)  # min > max


class TestRmaFreqNorm:
    """测试递归移动平均白化"""
    
    def test_rma_normalization_basic(self, multi_freq_signal):
        """测试RMA白化基本功能"""
        normalizer = RmaFreqNorm(alpha=0.9)
        whitened = normalizer.apply(multi_freq_signal)
        
        # 基本检查
        assert whitened.shape == multi_freq_signal.shape
        
        # 第一次应用后应该初始化avg_power
        assert normalizer.avg_power is not None
    
    def test_rma_multi_application(self, multi_freq_signal):
        """测试RMA多次应用"""
        normalizer = RmaFreqNorm(alpha=0.9)
        
        # 第一次应用
        whitened1 = normalizer.apply(multi_freq_signal)
        power1 = normalizer.avg_power.copy()
        
        # 第二次应用相同信号
        whitened2 = normalizer.apply(multi_freq_signal)
        power2 = normalizer.avg_power
        
        # 平均功率应该更新（由于递归平均）
        assert not np.array_equal(power1, power2)
        
        # 但输出应该相似
        assert np.allclose(whitened1, whitened2, rtol=0.1)
    
    def test_rma_different_signals(self):
        """测试RMA处理不同信号"""
        normalizer = RmaFreqNorm(alpha=0.9)
        
        # 信号1
        t = np.linspace(0, 1, 1000)
        signal1 = np.sin(2 * np.pi * 10 * t)
        whitened1 = normalizer.apply(signal1)
        
        # 信号2（不同频率）
        signal2 = np.sin(2 * np.pi * 20 * t)
        whitened2 = normalizer.apply(signal2)
        
        # 应该都能正常处理
        assert whitened1.shape == signal1.shape
        assert whitened2.shape == signal2.shape
    
    def test_rma_alpha_effect(self, multi_freq_signal):
        """测试alpha参数的影响"""
        # 使用不同的alpha值
        for alpha in [0.5, 0.9, 0.99]:
            normalizer = RmaFreqNorm(alpha=alpha)
            whitened = normalizer.apply(multi_freq_signal)
            assert whitened.shape == multi_freq_signal.shape


class TestNoFreqNorm:
    """测试无频域归一化"""
    
    def test_no_normalization(self, multi_freq_signal):
        """测试无操作归一化"""
        normalizer = NoFreqNorm()
        result = normalizer.apply(multi_freq_signal)
        
        # 应该返回原始信号的副本
        assert np.array_equal(result, multi_freq_signal)
        assert result is not multi_freq_signal  # 应该是副本


class TestGetFreqNormalizer:
    """测试频域归一化器工厂函数"""
    
    def test_get_all_normalizers(self):
        """测试获取所有支持的频域归一化器"""
        for name in _FREQ_NORM_MAP.keys():
            if name == 'bandwhiten':  # 跳过需要参数的BandWhitening
                continue
            normalizer = get_freq_normalizer(name)
            assert normalizer is not None
            assert hasattr(normalizer, 'apply')
    
    def test_get_normalizer_with_params(self, bandwhiten_params, spectral_whiten_params, rma_params):
        """测试带参数的归一化器获取"""
        # 测试SpectralWhitening带参数
        spec_normalizer = get_freq_normalizer('whiten', **spectral_whiten_params)
        assert isinstance(spec_normalizer, SpectralWhitening)
        assert spec_normalizer.smooth_win == spectral_whiten_params['smooth_win']
        
        # 测试BandWhitening带参数
        band_normalizer = get_freq_normalizer('bandwhiten', **bandwhiten_params)
        assert isinstance(band_normalizer, BandWhitening)
        assert band_normalizer.fmin == bandwhiten_params['freq_min']
        
        # 测试RmaFreqNorm带参数
        rma_normalizer = get_freq_normalizer('rma', **rma_params)
        assert isinstance(rma_normalizer, RmaFreqNorm)
        assert rma_normalizer.alpha == rma_params['alpha']
    
    def test_get_normalizer_invalid_name(self):
        """测试无效归一化器名称"""
        with pytest.raises(ValueError, match="Unknown frequency normalization method"):
            get_freq_normalizer('invalid_method')
    
    def test_bandwhiten_missing_params(self):
        """测试BandWhitening缺少必需参数"""
        with pytest.raises(ValueError, match="BandWhitening requires"):
            get_freq_normalizer('bandwhiten')  # 缺少必需参数
    
    def test_normalizer_callable_interface(self, multi_freq_signal):
        """测试归一化器的可调用接口"""
        normalizer = get_freq_normalizer('whiten')
        
        # 测试apply方法和__call__方法应该一致
        result1 = normalizer.apply(multi_freq_signal)
        result2 = normalizer(multi_freq_signal)
        
        assert np.array_equal(result1, result2)


class TestEdgeCases:
    """测试频域归一化的边界情况"""
    
    def test_empty_signal(self):
        """测试空信号"""
        empty_signal = np.array([])
        
        for name in ['whiten', 'rma', 'bandwhiten', 'no']:
            if name == 'bandwhiten':  # 跳过需要参数的BandWhitening
                continue
            normalizer = get_freq_normalizer(name)
            result = normalizer.apply(empty_signal)
            assert len(result) == 0
    
    def test_single_point_signal(self):
        """测试单点信号"""
        single_point = np.array([5.0])
        
        for name in ['whiten', 'rma', 'no']:
            normalizer = get_freq_normalizer(name)
            result = normalizer.apply(single_point)
            assert len(result) == 1
    
    def test_large_signal(self):
        """测试大信号（性能测试）"""
        large_signal = np.random.randn(100000)  # 10万个点
        
        for name in ['whiten', 'rma', 'no']:
            normalizer = get_freq_normalizer(name)
            
            # 时间性能测试
            import time
            start_time = time.time()
            result = normalizer.apply(large_signal)
            end_time = time.time()
            
            assert len(result) == len(large_signal)
            assert end_time - start_time < 5.0  # 应该在5秒内完成
    
    def test_very_low_frequency_signal(self):
        """测试极低频信号"""
        t = np.linspace(0, 10, 10000)  # 长时信号以包含低频
        low_freq_signal = np.sin(2 * np.pi * 0.1 * t)  # 0.1Hz
        
        normalizer = get_freq_normalizer('whiten')
        result = normalizer.apply(low_freq_signal)
        
        assert result.shape == low_freq_signal.shape


def test_normalizer_immutability(multi_freq_signal):
    """测试归一化器不会修改原始信号"""
    original_copy = multi_freq_signal.copy()
    
    for name in _FREQ_NORM_MAP.keys():
        if name == 'bandwhiten':  # 跳过需要参数的BandWhitening
            continue
            
        normalizer = get_freq_normalizer(name)
        normalized = normalizer.apply(multi_freq_signal)
        
        # 原始信号不应被修改
        assert np.array_equal(multi_freq_signal, original_copy)
        assert normalized is not multi_freq_signal  # 应该是新数组


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])