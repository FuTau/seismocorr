# seismocorr/preprocessing/freq_norm.py

"""
频域归一化方法（Frequency-domain Normalization）

通常在计算 FFT 后、互相关前使用。
参考：Bensen et al., 2007; Denolle et al., 2013
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union

ArrayLike = Union[np.ndarray, list]


class FreqNormalizer(ABC):
    """频域归一化抽象基类"""
    @abstractmethod
    def apply(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: 复数频域信号，shape=(n_windows, n_freqs//2)

        Returns:
            归一化后的频域数据
        """
        pass

    def __call__(self, X):
        return self.apply(X)


class SpectralWhitening(FreqNormalizer):
    """谱白化（Spectral Whitening）"""
    def __init__(self, smooth_win: int = 20):
        self.smooth_win = smooth_win

    def _smooth(self, x: np.ndarray) -> np.ndarray:
        """移动平均平滑"""
        if len(x) < self.smooth_win:
            return x
        pad_len = self.smooth_win // 2
        x_padded = np.pad(x, pad_len, mode='edge')
        kernel = np.ones(self.smooth_win) / self.smooth_win
        return np.convolve(x_padded, kernel, mode='valid')

    def apply(self, X: np.ndarray) -> np.ndarray:
        # 对每个窗口做白化
        Y = np.zeros_like(X, dtype=complex)
        for i in range(X.shape[0]):
            spec = X[i]
            amplitude = np.abs(spec)
            if np.max(amplitude) == 0:
                Y[i] = spec
            else:
                smoothed_amp = self._smooth(amplitude)
                # 白化：除以平滑后的幅度谱
                with np.errstate(divide='ignore', invalid='ignore'):
                    weight = np.where(smoothed_amp > 0, 1.0 / smoothed_amp, 0.0)
                Y[i] = spec * weight
        return Y

class BandWhitening(FreqNormalizer):
    """谱白化（Spectral Whitening）"""
    def __init__(self, freq_min: float, freq_max: float, Fs):
        self.fmin = freq_min
        self.fmax = freq_max
        self.Fs = Fs

    def apply(self, X: np.ndarray) -> np.ndarray:
        # 对每个窗口做白化
        nsamp = self.Fs
        n = len(data)
        if n == 1:
            return data
        else:
            frange = float(self.fmax) - float(self.fmin)
            nsmo = int(np.fix(min(0.01, 0.5 * (frange))* float(n)/nsamp))
            f = np.arange(n) * nsamp /(n -1.)
            JJ = ((f > float(self.fmin)) & (f < float(self.fmax))).nonzero()[0]

            # 信号的傅里叶变换
            FFTs = np.fft.fft(data)
            FFTsW = np.zeros(n) + 1j * np.zeros(n)

            # 
            smo1 = (np.cos(np.linspace(np.pi / 2, np.pi, nsmo+1))**2)
            FFTsW[JJ[0]:JJ[0]+nsmo+1] = smo1 * np.exp(1j * np.angle(FFTs[JJ[0]:JJ[0]+nsmo+1]))

            FFTsW[JJ[0]+nsmo+1:JJ[-1]-nsmo] = np.ones(len(JJ) - 2 * (nsmo+1))\
            * np.exp(1j * np.angle(FFTs[JJ[0]+nsmo+1:JJ[-1]-nsmo]))

            smo2 = (np.cos(np.linspace(0., np.pi/2., nsmo+1))**2.)
            espo = np.exp(1j * np.angle(FFTs[JJ[-1]-nsmo:JJ[-1]+1]))
            FFTsW[JJ[-1]-nsmo:JJ[-1]+1] = smo2 * espo

            whitedata = 2. * np.fft.ifft(FFTsW).real
            
            data = np.require(whitedata, dtype="float32")

            return data

class RmaFreqNorm(FreqNormalizer):
    """Recursive Moving Average (RMA) 白化"""
    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha  # 平滑系数

    def apply(self, X: np.ndarray) -> np.ndarray:
        Y = np.zeros_like(X, dtype=complex)
        avg_power = None

        for i in range(X.shape[0]):
            power = np.abs(X[i]) ** 2
            if avg_power is None:
                avg_power = power
            else:
                avg_power = self.alpha * avg_power + (1 - self.alpha) * power

            with np.errstate(divide='ignore', invalid='ignore'):
                norm_factor = np.where(avg_power > 0, 1.0 / np.sqrt(avg_power), 0.0)
            Y[i] = X[i] * norm_factor

        return Y


class NoFreqNorm(FreqNormalizer):
    """无频域归一化"""
    def apply(self, X: np.ndarray) -> np.ndarray:
        return X.copy()

_FREQ_NORM_MAP = {
    'whiten': SpectralWhitening,
    'rma': RmaFreqNorm,
    'no': NoFreqNorm,
}

def get_freq_normalizer(name: str, **kwargs) -> FreqNormalizer:
    """
    获取频域归一化器实例

    Args:
        name: 方法名 ('whiten', 'rma', 'no')
        **kwargs: 如 smooth_win, alpha

    Returns:
        FreqNormalizer 实例
    """
    cls = _FREQ_NORM_MAP.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown frequency normalization method: '{name}'. "
                       f"Choose from {list(_FREQ_NORM_MAP.keys())}")

    if name.lower() == 'whiten':
        return SpectralWhitening(smooth_win=kwargs.get('smooth_win', 20))
    elif name.lower() == 'rma':
        return RmaFreqNorm(alpha=kwargs.get('alpha', 0.9))
    elif name.lower() == 'bandwhiten':
        return BandWhitening(freq_min=kwargs['freq_min'], freq_max=kwargs['freq_max'], Fs=kwargs['Fs'])
    else:
        return cls()