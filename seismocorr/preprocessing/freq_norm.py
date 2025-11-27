# seismocorr/preprocessing/freq_norm.py

"""
频域归一化方法（Frequency-domain Normalization）

接收时域信号，进行频域白化处理，返回时域信号。
参考：Bensen et al., 2007; Denolle et al., 2013
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional

ArrayLike = Union[np.ndarray, list]


class FreqNormalizer(ABC):
    """频域归一化抽象基类"""
    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Args:
            data: 时域信号
            
        Returns:
            归一化后的时域数据
        """
        pass

    def __call__(self, data):
        return self.apply(data)
    def _check_empty_input(self, data: np.ndarray) -> bool:
        """检查输入是否为空数组"""
        return len(data) == 0


class SpectralWhitening(FreqNormalizer):
    """谱白化（Spectral Whitening）"""
    def __init__(self, smooth_win: int = 20):
        self.smooth_win = smooth_win

    def _smooth(self, x: np.ndarray) -> np.ndarray:
        """移动平均平滑，确保输出长度与输入相同"""
        if len(x) < self.smooth_win:
            return x
        
        # 使用 'same' 模式进行卷积，保持输出长度与输入相同
        kernel = np.ones(self.smooth_win) / self.smooth_win
        smoothed = np.convolve(x, kernel, mode='same')
        
        return smoothed

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        对时域信号进行谱白化
        
        Args:
            data: 时域信号
            
        Returns:
            白化后的时域信号
        """
        if self._check_empty_input(data):
            return data.copy()
        # 计算FFT
        n = len(data)
        FFT = np.fft.fft(data)
        
        # 获取幅度谱和相位谱
        amplitude = np.abs(FFT)
        phase = np.angle(FFT)
        
        # 对幅度谱进行平滑
        smoothed_amp = self._smooth(amplitude)
        
        # 避免除以零
        with np.errstate(divide='ignore', invalid='ignore'):
            weight = np.where(smoothed_amp > 0, 1.0 / smoothed_amp, 0.0)
        
        # 应用白化权重
        whitened_FFT = FFT * weight
        
        # 计算IFFT返回时域信号
        whitened_data = np.fft.ifft(whitened_FFT).real
        
        return whitened_data


class BandWhitening(FreqNormalizer):
    """频带白化（Band Whitening）"""
    def __init__(self, freq_min: float, freq_max: float, Fs: float):
        self.fmin = freq_min
        self.fmax = freq_max
        self.Fs = Fs
        if self.fmin >= self.fmax:
            raise ValueError("freq_min must be less than freq_max")

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        对时域信号进行频带白化
        
        Args:
            data: 时域信号
            
        Returns:
            白化后的时域信号
        """
        if self._check_empty_input(data):
            return data.copy()
        n = len(data)
        if n == 1:
            return data
        
        # 计算频率范围
        frange = float(self.fmax) - float(self.fmin)
        nsmo = int(np.fix(min(0.01, 0.5 * frange) * float(n) / self.Fs))
        
        # 计算频率轴
        f = np.arange(n) * self.Fs / (n - 1.)
        
        # 找到目标频带
        JJ = ((f > float(self.fmin)) & (f < float(self.fmax))).nonzero()[0]
        
        if len(JJ) == 0:
            # 如果没有找到目标频带，返回原始信号
            return data.copy()
        
        # 计算FFT
        FFTs = np.fft.fft(data)
        FFTsW = np.zeros(n, dtype=complex)
        
        # 创建白化滤波器
        # 频带外设置为0
        FFTsW[:] = 0.0
        
        # 频带内设置过渡带
        if nsmo > 0 and len(JJ) > 2 * nsmo:
            # 左过渡带（余弦锥）
            smo1 = (np.cos(np.linspace(np.pi / 2, np.pi, nsmo + 1)) ** 2)
            FFTsW[JJ[0]:JJ[0] + nsmo + 1] = smo1 * np.exp(1j * np.angle(FFTs[JJ[0]:JJ[0] + nsmo + 1]))
            
            # 频带内完全白化
            FFTsW[JJ[0] + nsmo + 1:JJ[-1] - nsmo] = np.ones(len(JJ) - 2 * (nsmo + 1)) * \
                                                   np.exp(1j * np.angle(FFTs[JJ[0] + nsmo + 1:JJ[-1] - nsmo]))
            
            # 右过渡带（余弦锥）
            smo2 = (np.cos(np.linspace(0., np.pi / 2., nsmo + 1)) ** 2)
            espo = np.exp(1j * np.angle(FFTs[JJ[-1] - nsmo:JJ[-1] + 1]))
            FFTsW[JJ[-1] - nsmo:JJ[-1] + 1] = smo2 * espo
        else:
            # 如果频带太窄，直接在整个频带内白化
            FFTsW[JJ] = np.exp(1j * np.angle(FFTs[JJ]))
        
        # 计算IFFT
        whitedata = 2.0 * np.fft.ifft(FFTsW).real
        
        return whitedata


class RmaFreqNorm(FreqNormalizer):
    """递归移动平均白化 (Recursive Moving Average Whitening)"""
    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha  # 平滑系数
        self.avg_power = None

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        对时域信号进行RMA白化
        
        Args:
            data: 时域信号
            
        Returns:
            白化后的时域信号
        """
        if self._check_empty_input(data):
            return data.copy()
        # 计算FFT
        n = len(data)
        FFT = np.fft.fft(data)
        
        # 计算功率谱
        power = np.abs(FFT) ** 2
        
        # 更新平均功率（递归移动平均）
        if self.avg_power is None or len(self.avg_power) != len(power):
            self.avg_power = power
        else:
            self.avg_power = self.alpha * self.avg_power + (1 - self.alpha) * power
        
        # 避免除以零
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_factor = np.where(self.avg_power > 0, 1.0 / np.sqrt(self.avg_power), 0.0)
        
        # 应用归一化
        whitened_FFT = FFT * norm_factor
        
        # 计算IFFT
        whitened_data = np.fft.ifft(whitened_FFT).real
        
        return whitened_data


class NoFreqNorm(FreqNormalizer):
    """无频域归一化"""
    def apply(self, data: np.ndarray) -> np.ndarray:
        return data.copy()


# 更新映射表
_FREQ_NORM_MAP = {
    'whiten': SpectralWhitening,
    'rma': RmaFreqNorm,
    'bandwhiten': BandWhitening,
    'no': NoFreqNorm,
}


def get_freq_normalizer(name: str, **kwargs) -> FreqNormalizer:
    """
    获取频域归一化器实例

    Args:
        name: 方法名 ('whiten', 'rma', 'bandwhiten', 'no')
        **kwargs: 如 smooth_win, alpha, freq_min, freq_max, Fs

    Returns:
        FreqNormalizer 实例
    """
    cls = _FREQ_NORM_MAP.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown frequency normalization method: '{name}'. "
                       f"Choose from {list(_FREQ_NORM_MAP.keys())}")

    # 根据方法名传递特定参数
    if name.lower() == 'whiten':
        smooth_win = kwargs.get('smooth_win', 20)
        return cls(smooth_win=smooth_win)
    elif name.lower() == 'rma':
        alpha = kwargs.get('alpha', 0.9)
        return cls(alpha=alpha)
    elif name.lower() == 'bandwhiten':
        # BandWhitening 需要特定参数
        freq_min = kwargs.get('freq_min')
        freq_max = kwargs.get('freq_max')
        Fs = kwargs.get('Fs')
        
        if freq_min is None or freq_max is None or Fs is None:
            raise ValueError("BandWhitening requires freq_min, freq_max, and Fs parameters")
            
        return cls(freq_min=freq_min, freq_max=freq_max, Fs=Fs)
    else:
        # 其他归一化方法使用默认构造函数
        return cls()