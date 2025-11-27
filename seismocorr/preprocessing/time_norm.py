# seismocorr/preprocessing/time_norm.py

"""
时域归一化方法（Time-domain Normalization）

在 FFT 前对时间序列进行预处理。
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union

ArrayLike = Union[np.ndarray, list]

def moving_ave(A, N): ## change the moving average calculation to take as input N the full window length to smooth
    '''
    Alternative function for moving average for an array.
    PARAMETERS:
    ---------------------
    A: 1-D array of data to be smoothed
    N: integer, it defines the full!! window length to smooth
    RETURNS:
    ---------------------
    B: 1-D array with smoothed data
    '''
    # defines an array with N extra samples at either side
    temp = np.zeros(len(A) + 2 * N)
    # set the central portion of the array to A
    temp[N: -N] = A
    # leading samples: equal to first sample of actual array
    temp[0: N] = temp[N]
    # trailing samples: Equal to last sample of actual array
    temp[-N:] = temp[-N-1]
    # convolve with a boxcar and normalize, and use only central portion of the result
    # with length equal to the original array, discarding the added leading and trailing samples
    B = np.convolve(temp, np.ones(N)/N, mode='same')[N: -N]
    return B

class TimeNormalizer(ABC):
    """时域归一化抽象基类"""
    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x):
        return self.apply(x)


class ZScoreNormalizer(TimeNormalizer):
    """Z-Score 标准化: (x - μ) / σ"""
    def apply(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean()
        std = x.std()
        if std == 0:
            return x - mean
        return (x - mean) / std


class OneBitNormalizer(TimeNormalizer):
    """1-bit 归一化: sign(x)"""
    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x)


class RMSNormalizer(TimeNormalizer):
    """RMS 归一化: x / sqrt(mean(x²))"""
    def apply(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2))
        if rms == 0:
            return x.copy()
        return x / rms


class ClipNormalizer(TimeNormalizer):
    """截幅归一化：限制最大值"""
    def __init__(self, clip_val: float = 3.0):
        self.clip_val = clip_val

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, -self.clip_val, self.clip_val)


class NoTimeNorm(TimeNormalizer):
    """无操作"""
    def apply(self, x: np.ndarray) -> np.ndarray:
        return x.copy()
    

class RAMNormalizer(TimeNormalizer):
    """RAM 归一化: x / mean(|x|)"""
    def __init__(self,fmin,Fs,norm_win=0.5):
        self.fmin = fmin
        self.Fs = Fs
        self.norm_win = norm_win

    def apply(self, x: np.ndarray) -> np.ndarray:
        period = 1 / self.fmin
        lwin = int(period * self.Fs * self.norm_win)
        N = 2*lwin+1
        x = x/moving_ave(np.abs(x),N)
        return x.copy()

_TIME_NORM_MAP = {
    'zscore': ZScoreNormalizer,
    'one-bit': OneBitNormalizer,
    'rms': RMSNormalizer,
    'clip': ClipNormalizer,  # 直接使用类，而不是lambda
    'no': NoTimeNorm,
    'ramn': RAMNormalizer
}

def get_time_normalizer(name: str, **kwargs) -> TimeNormalizer:
    """
    获取时域归一化器实例

    Args:
        name: 方法名 ('zscore', 'one-bit', 'rms', 'clip', 'ramn', 'no')
        **kwargs: 传递给特定类的参数

    Returns:
        TimeNormalizer 实例
    """
    cls = _TIME_NORM_MAP.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown time normalization method: '{name}'. "
                       f"Choose from {list(_TIME_NORM_MAP.keys())}")
    
    # 根据方法名传递特定参数
    if name.lower() == 'clip':
        clip_val = kwargs.get('clip_val', 3.0)
        return cls(clip_val=clip_val)
    elif name.lower() == 'ramn':
        # RAMNormalizer 需要特定参数
        fmin = kwargs.get('fmin')
        Fs = kwargs.get('Fs')
        norm_win = kwargs.get('norm_win', 0.5)
        
        if fmin is None or Fs is None:
            raise ValueError("RAMNormalizer requires fmin and Fs parameters")
            
        return cls(fmin=fmin, Fs=Fs, norm_win=norm_win)
    else:
        # 其他归一化方法使用默认构造函数
        return cls()