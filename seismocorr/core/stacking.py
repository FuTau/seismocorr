# seismocorr/core/stacking.py

"""
Staking Strategies for Cross-Correlation Functions (CCFs)

支持多种叠加方法，用于提升信噪比（SNR）。
所有方法均接受一个 CCF 列表（List[np.ndarray]），返回一个叠加后的 CCF。
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Callable

from scipy.fftpack import hilbert, next_fast_len

# 类型别名
ArrayLike = Union[np.ndarray, List[float], List[np.ndarray]]


class StackingStrategy(ABC):
    """
    叠加策略抽象基类
    所有具体策略需继承并实现 stack 方法
    """

    @abstractmethod
    def stack(self, ccf_list: List[np.ndarray]) -> np.ndarray:
        """
        对一组 CCF 进行叠加

        Args:
            ccf_list: 多个互相关函数，形状应相同 [n_lags]

        Returns:
            stacked_ccf: 叠加后的互相关函数
        """
        pass

    def __call__(self, ccf_list: List[np.ndarray]) -> np.ndarray:
        return self.stack(ccf_list)


class LinearStack(StackingStrategy):
    """线性叠加：最简单的平均"""
    def stack(self, ccf_list: List[np.ndarray]) -> np.ndarray:
        arr = np.array(ccf_list)
        return np.mean(arr, axis=0)


class SelectiveStack(StackingStrategy):
    """选择叠加：将与平均值相关性低的剔除后再叠加"""
    def stack(self, ccf_list: List[np.ndarray]) -> np.ndarray:
        arr = np.array(ccf_list)
        cc = np.ones(arr.shape[0])
        new_stack = np.mean(arr, axis=0)
        for i in range(arr.shape[0]):
            cc[i] = np.corrcoef(new_stack, arr[i])[0, 1]
        epsilon = np.median(cc)
        ik = np.where(cc>=epsilon)[0]
        new_stack = np.mean(arr[ik,:], axis=0)
        return new_stack

class NrootStack(StackingStrategy):
    """N次根叠加"""
    def __init__(self):
        self.power = 2

    def stack(self, ccf_list: List[np.ndarray]) -> np.ndarray:
        arr = np.array(ccf_list)
        N, M = arr.shape
        dout = np.zeros(M,dtype=np.float32)

        for i in range(N):
            dat = arr[i,:]
            dout += np.sign(dat) * (np.abs(dat))**(1.0/self.power)
        dout /= N
        nstack = dout * np.abs(dout)**(self.power - 1.0)
        return nstack
        
    
class PhaseWeightedStack(StackingStrategy):
    """
    相位加权叠加（PWS）
    Ref: Schimmel and Palssen, 1997
    使用相位一致性作为权重：一致性越高，权重越大
    """
    def __init__(self,power):
        self.power = power

    def stack(self, ccf_list: List[np.ndarray]) -> np.ndarray:
        ccfs = np.array(ccf_list)
        N,M = ccfs.shape
        analytic = hilbert(ccfs, axis=1, N=next_fast_len(M))
        phase = np.angle(analytic)
        phase_stack = np.mean(np.exp(1j*phase), axis=0)
        phase_stack = np.abs(phase_stack)**self.power
        weighted = np.multiply(ccfs, phase_stack)
        return np.mean(weighted, axis=0)


class RobustStack(StackingStrategy):
    """
    鲁棒叠加（Robust Stack）
    REF: Palvis and Vernon, 2010
    """
    def __init__(self,epsilon: float = 1e-8):
        self.epsilon = epsilon  # 防止除零
    def stack(self, ccf_list: List[np.ndarray]) -> np.ndarray:
        res = 9e9
        ccfs = np.array(ccf_list)
        w = np.ones(ccfs.shape[0])
        nstep = 0
        newstack = np.median(ccfs, axis=0)
        while res > self.epsilon:
            stack = newstack
            for i in range(ccfs.shape[0]):
                crap = np.multiply(stack, ccfs[i,:].T)
                crap_dot = np.sum(crap)
                di_norm = np.linalg.norm(ccfs[i,:])
                ri = ccfs[i,:] - crap_dot * stack
                ri_norm = np.linalg.norm(ri)
                w[i] = np.abs(crap_dot) / di_norm / ri_norm
            w = w / np.sum(w)
            newstack = np.sum((w*ccfs.T).T, axis=0)
            res = np.linalg.norm(newstack - stack,ord = 1) / np.linalg.norm(newstack) / len(ccfs[:,1])
            nstep += 1
            if nstep > 10:
                return newstack, w, nstep
        return newstack



class TimeFrequencyStack(StackingStrategy):
    """
    时频加权叠加（TF-PWS）
    在时频域进行相位加权，进一步提升分辨率
    Ref: Bensen et al., 2007; Trad et al., 2003
    """
    def __init__(self, window_length: int = 20, alpha: float = 2.0):
        self.win_len = window_length
        self.alpha = alpha

    def stack(self, ccf_list: List[np.ndarray]) -> np.ndarray:
        from scipy.signal import stft, istft

        stacked = np.zeros_like(ccf_list[0])
        weights = np.ones_like(stacked)

        for ccf in ccf_list:
            # 短时傅里叶变换
            f, t, Z = stft(ccf, nperseg=self.win_len)
            phase = np.exp(1j * np.angle(Z))
            phase_coh = np.abs(np.mean(phase, axis=0))  # 时间点上的平均相位一致性
            tf_weights = phase_coh ** self.alpha

            # 逆变换回时域
            _, reconstructed = istft(Z * tf_weights, nperseg=self.win_len)
            stacked += reconstructed[:len(stacked)]
            weights += tf_weights.mean(axis=0)[:len(stacked)]

        return stacked / len(ccf_list)


# =====================================================================
# 🏭 工厂函数：根据名称创建策略实例
# =====================================================================

_STRATEGY_REGISTRY = {
    'linear': LinearStack,
    'pws': PhaseWeightedStack,
    'robust': RobustStack,
    'selective': SelectiveStack,
    'tf-pws': TimeFrequencyStack,
}

def get_stacker(name: str, **kwargs) -> StackingStrategy:
    """
    工厂函数：根据名称获取叠加器实例

    Args:
        name: 叠加方法名，如 'linear', 'pws', 'robust'
        **kwargs: 传递给具体策略的参数（如 alpha, threshold）

    Returns:
        StackingStrategy 实例

    Raises:
        ValueError: 如果方法名不支持
    """
    cls = _STRATEGY_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown stacking method: {name}. "
                       f"Choose from {list(_STRATEGY_REGISTRY.keys())}")
    
    # 特殊处理带参数的类
    if name.lower() == 'pws':
        return cls(alpha=kwargs.get('power', 2.0))
    elif name.lower() == 'robust':
        return cls(
            threshold=kwargs.get('epsilon', 1.0*1e-8)
        )
    elif name.lower() == 'tf-pws':
        return cls(
            window_length=kwargs.get('window_length', 20),
            alpha=kwargs.get('alpha', 2.0)
        )
    else:
        return cls()


# =====================================================================
# 🔧 辅助函数：直接对数组列表进行叠加（简化接口）
# =====================================================================

def stack_ccfs(ccf_list: List[np.ndarray], method: str = 'linear', **kwargs) -> np.ndarray:
    """
    快捷函数：直接对一组 CCF 执行叠加

    Example:
        stacked = stack_ccfs([ccf1, ccf2, ccf3], method='pws', alpha=4)

    Args:
        ccf_list: CCF 数组列表
        method: 叠加方法名
        **kwargs: 方法参数

    Returns:
        叠加后的 CCF
    """
    stacker = get_stacker(method, **kwargs)
    return stacker(ccf_list)
