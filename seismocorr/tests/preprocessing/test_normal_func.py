import pytest
import numpy as np
import sys
import os
from scipy.signal import detrend as scipy_detrend

from seismocorr.preprocessing.normal_func import (
    demean, detrend, taper,
    bandpass, lowpass, highpass,
    _butter_filter
)


class TestDemean:
    """测试去均值函数"""
    
    def test_demean_basic(self, sample_signal):
        """测试基本去均值功能"""
        demeaned = demean(sample_signal)
        
        # 检查均值是否接近零
        assert np.isclose(demeaned.mean(), 0.0, atol=1e-10)
        # 检查标准差是否保持不变
        assert np.isclose(demeaned.std(), sample_signal.std(), rtol=1e-5)
    
    def test_demean_constant_signal(self, constant_signal):
        """测试常数信号的去均值"""
        demeaned = demean(constant_signal)
        # 常数信号去均值后应为全零
        assert np.allclose(demeaned, 0.0)
    
    def test_demean_zero_signal(self, zero_signal):
        """测试全零信号的去均值"""
        demeaned = demean(zero_signal)
        # 全零信号去均值后仍为全零
        assert np.allclose(demeaned, 0.0)
    
    def test_demean_empty_signal(self):
        """测试空信号"""
        empty_signal = np.array([])
        demeaned = demean(empty_signal)
        assert len(demeaned) == 0


class TestDetrend:
    """测试去趋势函数"""
    
    def test_detrend_linear(self, trend_signal):
        """测试线性去趋势"""
        detrended = detrend(trend_signal, type='linear')
        
        # 检查趋势是否被移除
        # 线性趋势信号的斜率应该接近零
        t = np.linspace(0, 1, len(trend_signal))
        slope = np.polyfit(t, detrended, 1)[0]
        assert abs(slope) < 0.1  # 斜率应该很小
    
    def test_detrend_constant(self, trend_signal):
        """测试常数去趋势（等同于去均值）"""
        detrended_constant = detrend(trend_signal, type='constant')
        demeaned = demean(trend_signal)
        
        # 常数去趋势应该等同于去均值
        assert np.allclose(detrended_constant, demeaned)
    
    def test_detrend_no_trend_signal(self, sample_signal):
        """测试无趋势信号的去趋势"""
        detrended = detrend(sample_signal, type='linear')
        
        # 无趋势信号去趋势后应该变化不大
        correlation = np.corrcoef(sample_signal, detrended)[0, 1]
        assert correlation > 0.98  # 高度相关
    
    def test_detrend_constant_signal(self, constant_signal):
        """测试常数信号的去趋势"""
        detrended = detrend(constant_signal, type='linear')
        # 常数信号去趋势后应为全零
        assert np.allclose(detrended, 0.0)
    
    def test_detrend_invalid_type(self, sample_signal):
        """测试无效的去趋势类型"""
        with pytest.raises(ValueError):
            detrend(sample_signal, type='invalid_type')


class TestTaper:
    """测试加窗函数"""
    
    def test_taper_basic(self, edge_heavy_signal):
        """测试基本加窗功能"""
        tapered = taper(edge_heavy_signal, width=0.1)
        
        # 检查信号长度不变
        assert len(tapered) == len(edge_heavy_signal)
        # 检查边缘值被衰减
        assert abs(tapered[0]) < abs(edge_heavy_signal[0])
        assert abs(tapered[-1]) < abs(edge_heavy_signal[-1])
        # 检查中间部分基本不变
        mid_idx = len(edge_heavy_signal) // 2
        assert np.isclose(tapered[mid_idx], edge_heavy_signal[mid_idx])
    
    def test_taper_different_widths(self, edge_heavy_signal):
        """测试不同窗宽"""
        for width in [0.05, 0.1, 0.2]:
            tapered = taper(edge_heavy_signal, width=width)
            assert len(tapered) == len(edge_heavy_signal)
    
    def test_taper_zero_width(self, edge_heavy_signal):
        """测试零窗宽（应该返回原信号）"""
        tapered = taper(edge_heavy_signal, width=0.0)
        assert np.array_equal(tapered, edge_heavy_signal)
    
    def test_taper_small_signal(self):
        """测试小信号加窗"""
        small_signal = np.array([1.0, 2.0, 3.0])
        tapered = taper(small_signal, width=0.5)
        assert len(tapered) == 3


class TestButterFilter:
    """测试Butterworth滤波器"""
    
    def test_butter_filter_bandpass(self, multi_freq_signal, bandpass_params):
        """测试带通滤波"""
        filtered = _butter_filter(
            multi_freq_signal,
            sampling_rate=100.0,
            freq_min=bandpass_params['fmin'],
            freq_max=bandpass_params['fmax']
        )
        
        # 基本检查
        assert len(filtered) == len(multi_freq_signal)
        assert not np.allclose(filtered, multi_freq_signal)
    
    def test_butter_filter_lowpass(self, multi_freq_signal, lowpass_params):
        """测试低通滤波"""
        filtered = _butter_filter(
            multi_freq_signal,
            sampling_rate=100.0,
            freq_max=lowpass_params['fmax']
        )
        
        assert len(filtered) == len(multi_freq_signal)
    
    def test_butter_filter_highpass(self, multi_freq_signal, highpass_params):
        """测试高通滤波"""
        filtered = _butter_filter(
            multi_freq_signal,
            sampling_rate=100.0,
            freq_min=highpass_params['fmin']
        )
        
        assert len(filtered) == len(multi_freq_signal)
    
    def test_butter_filter_no_filter(self, sample_signal):
        """测试无滤波情况（应该返回原信号）"""
        filtered = _butter_filter(sample_signal, sampling_rate=100.0)
        assert np.array_equal(filtered, sample_signal)
    
    def test_butter_filter_zero_phase(self, multi_freq_signal):
        """测试零相位滤波"""
        # 测试零相位和非零相位滤波
        filtered_zero = _butter_filter(
            multi_freq_signal,
            sampling_rate=100.0,
            freq_min=1.0,
            freq_max=20.0,
            zero_phase=True
        )
        
        filtered_nonzero = _butter_filter(
            multi_freq_signal,
            sampling_rate=100.0,
            freq_min=1.0,
            freq_max=20.0,
            zero_phase=False
        )
        
        # 两者应该不同
        assert not np.allclose(filtered_zero, filtered_nonzero)
    
    def test_butter_filter_frequency_response(self):
        """测试滤波器的频率响应"""
        # 创建包含特定频率的信号
        Fs = 100.0
        t = np.linspace(0, 1, int(Fs))
        
        # 包含低频和高频成分
        low_freq = 5.0  # Hz
        high_freq = 40.0  # Hz
        signal = np.sin(2 * np.pi * low_freq * t) + 0.5 * np.sin(2 * np.pi * high_freq * t)
        
        # 应用低通滤波（截止频率20Hz）
        filtered = _butter_filter(signal, Fs, freq_max=20.0)
        
        # 计算频谱
        from scipy import signal as sp_signal
        f_orig, psd_orig = sp_signal.periodogram(signal, Fs)
        f_filt, psd_filt = sp_signal.periodogram(filtered, Fs)
        
        # 找到高频成分的索引
        idx_high = np.argmin(np.abs(f_orig - high_freq))
        
        # 高频成分应该被抑制
        attenuation = psd_filt[idx_high] / psd_orig[idx_high]
        assert attenuation < 0.1  # 衰减至少90%
    
    def test_butter_filter_edge_cases(self):
        """测试边界情况"""
        # 空信号
        empty_signal = np.array([])
        filtered = _butter_filter(empty_signal, sampling_rate=100.0)
        assert len(filtered) == 0
        
        # 单点信号
        single_point = np.array([1.0])
        filtered = _butter_filter(single_point, sampling_rate=100.0)
        assert len(filtered) == 1
        
        # 超出Nyquist频率
        signal = np.random.randn(100)
        filtered = _butter_filter(signal, sampling_rate=100.0, freq_min=60.0, freq_max=80.0)
        # 应该返回原始信号（因为频率超出Nyquist）
        assert np.array_equal(filtered, signal)


class TestBandpass:
    """测试带通滤波函数"""
    
    def test_bandpass_basic(self, multi_freq_signal, bandpass_params):
        """测试基本带通滤波"""
        filtered = bandpass(
            multi_freq_signal,
            fmin=bandpass_params['fmin'],
            fmax=bandpass_params['fmax'],
            sr=bandpass_params['sr']
        )
        
        assert len(filtered) == len(multi_freq_signal)
        assert not np.allclose(filtered, multi_freq_signal)
    
    def test_bandpass_frequency_range(self):
        """测试带通滤波的频率范围"""
        Fs = 100.0
        t = np.linspace(0, 1, int(Fs))
        
        # 创建包含带内和带外频率的信号
        in_band = 10.0  # Hz（在1-20Hz带内）
        out_band = 30.0  # Hz（在1-20Hz带外）
        signal = np.sin(2 * np.pi * in_band * t) + 0.5 * np.sin(2 * np.pi * out_band * t)
        
        # 应用带通滤波
        filtered = bandpass(signal, fmin=1.0, fmax=20.0, sr=Fs)
        
        # 计算频谱
        from scipy import signal as sp_signal
        f_orig, psd_orig = sp_signal.periodogram(signal, Fs)
        f_filt, psd_filt = sp_signal.periodogram(filtered, Fs)
        
        # 找到带外频率的索引
        idx_out = np.argmin(np.abs(f_orig - out_band))
        
        # 带外频率应该被显著抑制
        attenuation = psd_filt[idx_out] / psd_orig[idx_out]
        assert attenuation < 0.1  # 衰减至少90%


class TestLowpass:
    """测试低通滤波函数"""
    
    def test_lowpass_basic(self, multi_freq_signal, lowpass_params):
        """测试基本低通滤波"""
        filtered = lowpass(
            multi_freq_signal,
            fmax=lowpass_params['fmax'],
            sr=lowpass_params['sr']
        )
        
        assert len(filtered) == len(multi_freq_signal)
    
    def test_lowpass_high_frequency_attenuation(self):
        """测试低通滤波对高频的抑制"""
        Fs = 100.0
        t = np.linspace(0, 1, int(Fs))
        
        # 创建包含低频和高频的信号
        low_freq = 5.0  # Hz
        high_freq = 30.0  # Hz（高于截止频率15Hz）
        signal = np.sin(2 * np.pi * low_freq * t) + 0.5 * np.sin(2 * np.pi * high_freq * t)
        
        # 应用低通滤波（截止频率15Hz）
        filtered = lowpass(signal, fmax=15.0, sr=Fs)
        
        # 计算频谱
        from scipy import signal as sp_signal
        f_orig, psd_orig = sp_signal.periodogram(signal, Fs)
        f_filt, psd_filt = sp_signal.periodogram(filtered, Fs)
        
        # 找到高频成分的索引
        idx_high = np.argmin(np.abs(f_orig - high_freq))
        
        # 高频成分应该被显著抑制
        attenuation = psd_filt[idx_high] / psd_orig[idx_high]
        assert attenuation < 0.1  # 衰减至少90%


class TestHighpass:
    """测试高通滤波函数"""
    
    def test_highpass_basic(self, multi_freq_signal, highpass_params):
        """测试基本高通滤波"""
        filtered = highpass(
            multi_freq_signal,
            fmin=highpass_params['fmin'],
            sr=highpass_params['sr']
        )
        
        assert len(filtered) == len(multi_freq_signal)
    
    def test_highpass_low_frequency_attenuation(self):
        """测试高通滤波对低频的抑制"""
        Fs = 100.0
        t = np.linspace(0, 1, int(Fs))
        
        # 创建包含低频和高频的信号
        low_freq = 2.0  # Hz（低于截止频率5Hz）
        high_freq = 15.0  # Hz（高于截止频率5Hz）
        signal = np.sin(2 * np.pi * low_freq * t) + 0.5 * np.sin(2 * np.pi * high_freq * t)
        
        # 应用高通滤波（截止频率5Hz）
        filtered = highpass(signal, fmin=5.0, sr=Fs)
        
        # 计算频谱
        from scipy import signal as sp_signal
        f_orig, psd_orig = sp_signal.periodogram(signal, Fs)
        f_filt, psd_filt = sp_signal.periodogram(filtered, Fs)
        
        # 找到低频成分的索引
        idx_low = np.argmin(np.abs(f_orig - low_freq))
        
        # 低频成分应该被显著抑制
        attenuation = psd_filt[idx_low] / psd_orig[idx_low]
        assert attenuation < 0.1  # 衰减至少90%


class TestIntegration:
    """测试预处理函数的组合使用"""
    
    def test_preprocessing_pipeline(self, sample_signal):
        """测试完整的预处理流水线"""
        # 1. 去均值
        demeaned = demean(sample_signal)
        
        # 2. 去趋势
        detrended = detrend(demeaned, type='linear')
        
        # 3. 加窗
        tapered = taper(detrended, width=0.05)
        
        # 4. 带通滤波
        filtered = bandpass(tapered, fmin=1.0, fmax=20.0, sr=100.0)
        
        # 检查最终结果
        assert len(filtered) == len(sample_signal)
        assert not np.allclose(filtered, sample_signal)
    
    def test_filter_chain(self, multi_freq_signal):
        """测试滤波器链"""
        # 先低通再高通应该近似于带通
        lowpassed = lowpass(multi_freq_signal, fmax=20.0, sr=100.0)
        highpassed = highpass(lowpassed, fmin=1.0, sr=100.0)
        bandpassed = bandpass(multi_freq_signal, fmin=1.0, fmax=20.0, sr=100.0)
        
        # 两者应该相似但不完全相同
        correlation = np.corrcoef(highpassed, bandpassed)[0, 1]
        assert correlation > 0.7  # 应该有一定相关性


if __name__ == "__main__":
    pytest.main([__file__, "-v"])