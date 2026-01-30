from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
from .fj_helper_func import fj, fj_rr, mfj_rr
from seismocorr.utils.io import save_dispersion_for_picker
from scipy.special import j0, j1, jn_zeros

@dataclass
class DispersionConfig:
    """频散成像配置参数"""
    freqmin: float = 0.1
    freqmax: float = 10.0
    vmin: float = 100.0
    vmax: float = 5000.0
    vnum: int = 100
    sampling_rate: float = 100.0

@dataclass
class PlotConfig:
    """绘图配置参数"""
    fig_width: int = 10
    fig_height: int = 6
    font_size: int = 12
    cmap: str = 'jet'
    vmin: float = 0.0
    vmax: float = 0.8

class DispersionMethod(Enum):
    """频散成像方法枚举"""
    FJ = "fj"
    FJ_RR = "fj_rr"
    MFJ_RR = "mfj_rr"
    SLANT_STACK = "slant_stack"
    MASW = "masw"

class DispersionStrategy(ABC):
    """频散成像策略基类"""
    
    @abstractmethod
    def compute(self, cc_array_f: np.ndarray, f: np.ndarray, 
                dist: np.ndarray, config: DispersionConfig) -> np.ndarray:
        """计算频散谱"""
        pass
    
    def _calculate_weights(self, dist: np.ndarray) -> np.ndarray:
        """计算权重（FJ系列方法共用）"""
        rn = np.zeros(len(dist) + 2)
        rn[1:-1] = dist
        rn[-1] = rn[-2]
        return (rn[2:] ** 2 + 2 * rn[1:-1] * (rn[2:] - rn[0:-2]) - rn[0:-2] ** 2) / 8.

class FJ(DispersionStrategy):
    """FJ方法策略"""
    
    def compute(self, cc_array_f: np.ndarray, f: np.ndarray, 
                dist: np.ndarray, config: DispersionConfig) -> np.ndarray:
        v = np.linspace(config.vmin, config.vmax, config.vnum)
        
        # 准备fj_python函数所需的参数
        U_f = np.real(cc_array_f).astype(np.float32).flatten()  # 转换为一维数组，与helper_func.py中的实现一致
        r = dist
        c = v.astype(np.float32)
        nc = len(v)
        nr = len(r)
        nf = len(f)
        
        # 使用helper_func.py中的fj_python函数计算频散谱
        spec = fj(U_f, r, f, c, nc, nr, nf)

        return spec

class FJ_RR(DispersionStrategy):
    """FJ_RR方法策略"""
    
    def compute(self, cc_array_f: np.ndarray, f: np.ndarray, 
                dist: np.ndarray, config: DispersionConfig) -> np.ndarray:
        v = np.linspace(config.vmin, config.vmax, config.vnum)
        
        # 准备fj_rr函数所需的参数
        U_f = np.real(cc_array_f).flatten()  # 转换为一维数组
        r = dist
        c = v
        nc = len(v)
        nr = len(r)
        nf = len(f)
        
        # 使用fj_helper_func.py中的fj_rr函数计算频散谱
        spec = fj_rr(U_f, r, f, c, nc, nr, nf)
        
        return spec

class MFJ_RR(DispersionStrategy):
    """MFJ_RR方法策略"""
    
    def compute(self, cc_array_f: np.ndarray, f: np.ndarray, 
                dist: np.ndarray, config: DispersionConfig) -> np.ndarray:
        v = np.linspace(config.vmin, config.vmax, config.vnum)
        
        # 准备mfj_rr函数所需的参数
        U_f = np.real(cc_array_f).flatten()  # 转换为一维数组
        r = dist
        c = v
        nc = len(v)
        nr = len(r)
        nf = len(f)
        
        # 使用fj_helper_func.py中的mfj_rr函数计算频散谱
        spec = mfj_rr(U_f, r, f, c, nc, nr, nf)
        
        return spec

class SlantStack(DispersionStrategy):
    """Slant_stack方法策略"""
    
    def compute(self, cc_array_f: np.ndarray, freq: np.ndarray, 
                dist: np.ndarray, config: DispersionConfig) -> np.ndarray:
        
        # 计算最大慢度
        p_max = 1.0 / config.vmin
        
        # 试算慢度值
        q = np.linspace(0, p_max, config.vnum + 1)
        nq = len(q)
        
        # 初始化功率谱
        pnorm = np.zeros((len(freq), len(q)))
        
        # 遍历每个频率
        for n in range(len(freq)):
            omega = 2 * np.pi * freq[n]
            
            # 构建线性Radon变换矩阵
            L = np.exp(-1j * omega * np.outer(dist, q))
            
            # 执行Radon变换
            y = cc_array_f[:, n]
            x = np.dot(L.conj().T, y)
            
            # 计算功率
            pnorm[n, :] = np.abs(x) ** 2
            
            # 归一化功率
            max_power = np.nanmax(pnorm[n, :])
            if max_power > 0:
                pnorm[n, :] /= max_power
        
        # 找到峰值慢度
        p_peak = np.zeros(len(freq))
        for n in range(len(freq)):
            max_id = np.nanargmax(pnorm[n, :])
            p_peak[n] = np.max([q[max_id], 1e-10])  # 避免零慢度
        
        # 转换为速度
        v_peak = 1.0 / p_peak
        
        # 移除慢度为0的分量
        q = q[1:]
        v_vals = 1.0 / q
        pnorm = pnorm[:, 1:].T  # 转置为(vnum, nf)

        return pnorm

class MASW(DispersionStrategy):
    """MASW方法策略"""
    
    def compute(self, cc_array_f: np.ndarray, freq: np.ndarray,
         dist: np.ndarray, config: DispersionConfig) -> np.ndarray:
        """
        相移法频散提取
        参考: Park et al. (1998), MASWaves_dispersion_imaging
        
        Args:
            cc_array_f: 频域互相关数据或时域数据
            freq: 频率数组，如果为None且输入为时域数据则自动计算
            dist: 台站距离数组
            config: 频散配置
            
        Returns:
            频散谱
        """
        # 试算速度
        v_vals = np.linspace(config.vmin, config.vmax, config.vnum)
        
        # 如果输入是时域数据且freq为None，则计算频率
        if freq is None:
            # 假设输入是时域数据，计算FFT
            n_samples, n_stations = cc_array_f.shape
            
            # 计算频率数组
            freq = np.fft.fftfreq(n_samples, d=1/config.sampling_rate)
            freq = freq[:n_samples//2]  # 只保留正频率
            
            # 将时域数据转换为频域数据
            cc_array_f_fft = np.fft.fft(cc_array_f, axis=0)[:n_samples//2, :]
            n_stations = cc_array_f_fft.shape[1]
        else:
            # 输入是频域数据，检查形状
            if cc_array_f.shape[0] == len(dist):
                # 形状为(n_stations, n_freq)，转置为(n_freq, n_stations)
                cc_array_f_fft = cc_array_f.T
                n_stations = cc_array_f.shape[0]
            else:
                # 形状为(n_freq, n_stations)，直接使用
                cc_array_f_fft = cc_array_f
                n_stations = cc_array_f.shape[1]
        
        # 初始化功率谱
        power = np.zeros((len(v_vals), len(freq)))
        
        # 接收道数
        N = len(dist)
        
        # 遍历每个频率
        for c in range(len(freq)):
            # 跳过频率为0的情况
            if freq[c] <= 0:
                continue
            
            # 遍历每个试算速度
            for r in range(len(v_vals)):
                delta = 2 * np.pi * freq[c] / v_vals[r]
                
                # 计算相位移位因子
                phase_term = np.exp(-1j * delta * dist)
                
                # 避免除以零
                cc_amp = np.abs(cc_array_f_fft[c, :])
                if np.all(cc_amp == 0):
                    continue
                    
                normalized_cc = cc_array_f_fft[c, :] / cc_amp
                conj_normalized_cc = np.conj(normalized_cc)
                temp = np.sum(phase_term * conj_normalized_cc)
                
                # 计算绝对振幅并归一化
                power[r, c] = np.abs(temp) / N
        
        return power


class DispersionFactory:
    """频散成像工厂类"""
    
    @staticmethod
    def create_strategy(method: DispersionMethod) -> DispersionStrategy:
        """创建策略实例"""
        strategies = {
            DispersionMethod.FJ: FJ,
            DispersionMethod.FJ_RR: FJ_RR,
            DispersionMethod.MFJ_RR: MFJ_RR,
            DispersionMethod.SLANT_STACK: SlantStack,
            DispersionMethod.MASW: MASW,
        }
        
        if method not in strategies:
            raise ValueError(f"不支持的频散成像方法: {method}")
        
        return strategies[method]()

class DispersionAnalyzer:
    """频散分析器主类"""
    
    def __init__(self, method: DispersionMethod, config: Optional[DispersionConfig] = None):
        self.method = method
        self.config = config or DispersionConfig()
        self.strategy = DispersionFactory.create_strategy(method)
    
    def analyze(self, *args, **kwargs) -> np.ndarray:
        """执行频散分析"""
        result = self.strategy.compute(*args, **kwargs)
        
        # MASW方法直接返回power数组，不需要特殊处理
        return result
    
    def plot_spectrum(self, f: np.ndarray, c: np.ndarray, A: np.ndarray, 
                     plot_config: Optional[PlotConfig] = None, 
                     saved_figname: Optional[str] = None) -> None:
        """绘制频散谱图"""
        config = plot_config or PlotConfig()
        
        # 频率范围选择
        fmin, fmax = self.config.freqmin, self.config.freqmax
        no_fmin = np.argmin(np.abs(f - fmin))
        no_fmax = np.argmin(np.abs(f - fmax))
        
        # 处理不同形状的A数组
        if len(A.shape) == 3:
            # MASW方法返回的A可能是3D数组，取第一维
            A = A[:, :, 0]
        elif A.shape[0] != len(f) or A.shape[1] != len(c):
            # 如果A的形状不匹配，尝试转置
            A = A.T
        
        # 确保A的形状是[nf, nc]
        if A.shape[0] != len(f):
            # 如果频率维度不匹配，可能需要调整
            Aplot = A[:, no_fmin:no_fmax]
            fplot = f[no_fmin:no_fmax]
            # 转置Aplot以匹配预期形状
            Aplot = Aplot.T
        else:
            # 正常情况
            Aplot = A[no_fmin:no_fmax, :]
            fplot = f[no_fmin:no_fmax]
        
        # 绘图
        max_val = np.nanmax(np.abs(Aplot))
        
        
        # 使用shading='nearest'来避免维度不匹配的问题
        plt.pcolormesh(fplot, c, Aplot.T/max_val, 
                      cmap=config.cmap, vmin=0, vmax=0.5, shading='nearest')
        plt.grid(True)
        
        # 坐标轴设置
        plt.xticks(np.linspace(0, fmax + 0.01, 11))
        plt.xlabel('Frequency [Hz]', fontsize=config.font_size)
        plt.ylabel('Phase velocity [m/s]', fontsize=config.font_size)
        plt.xlim([fmin, fmax])
        
        # 图形设置
        plt.gcf().set_size_inches(config.fig_width, config.fig_height)
        plt.gca().tick_params(direction='out', which='both')
        plt.tick_params(axis='both', which='major', labelsize=config.font_size)
        
        # 颜色条
        cbar = plt.colorbar(location='top', pad=0.05)
        cbar.ax.tick_params(labelsize=config.font_size)
        cbar.set_label('Normalized amplitude', fontsize=config.font_size, labelpad=10)
        
        if saved_figname:
            plt.savefig(saved_figname, dpi=100)
        plt.show()


# 使用示例
def example_usage():
    """使用示例"""
    # 创建配置
    config = DispersionConfig(
        freqmin=0.5, freqmax=5.0, vmin=500, vmax=3000, vnum=200
    )
    
    plot_config = PlotConfig(fig_width=12, fig_height=8, font_size=14)
    
    # 模拟数据
    n_freq = 100
    n_stations = 10
    f = np.linspace(0.1, 10, n_freq)
    dist = np.linspace(100, 1000, n_stations)
    cc_array_f = np.random.rand(n_stations, n_freq) + 1j * np.random.rand(n_stations, n_freq)
    
    # 使用FJ方法分析
    analyzer = DispersionAnalyzer(DispersionMethod.FJ, config)
    spectrum = analyzer.analyze(cc_array_f, f, dist, config)
    
    # 绘制结果
    v = np.linspace(config.vmin, config.vmax, config.vnum)
    analyzer.plot_spectrum(f, v, spectrum, plot_config, "fj_spectrum.png")

    outpath = save_dispersion_for_picker(
        A=spectrum,
        f=f,
        v=v,
        method=analyzer.method.value,
        out_dir="outputs",
        freqmin=config.freqmin,
        freqmax=config.freqmax,
        normalize=True,
        tag=None  # 自定义文件名标识
    )

    print("Saved picker file:", outpath)

if __name__ == "__main__":
    example_usage()