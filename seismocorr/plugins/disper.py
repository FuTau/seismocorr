from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from scipy.special import hankel2, j0, j1, jn_zeros
from matplotlib import pyplot as plt
from numba import jit
from dataclasses import dataclass
from typing import Tuple, Optional, Callable

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
    SPAC = "spac"
    MASW = "masw"
    ZERO_CROSSING = "zero_crossing"

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
        w = self._calculate_weights(dist)
        spec = np.zeros((len(f), len(v)))
        
        for i, fi in enumerate(f):
            if fi == 0:
                continue
            FJ_array = np.zeros(len(v))
            y0 = cc_array_f[:, i]
            
            for j, vj in enumerate(v):
                k1 = 2 * np.pi * fi / vj
                m1 = j0(k1 * dist)
                FJ_array[j] = m1 @ (y0 * w) * (2 * np.pi * fi) ** 2 / vj**3
                
            spec[i, :] = FJ_array
            
        return spec

class FJ_RR(DispersionStrategy):
    """FJ_RR方法策略"""
    
    def compute(self, cc_array_f: np.ndarray, f: np.ndarray, 
                dist: np.ndarray, config: DispersionConfig) -> np.ndarray:
        v = np.linspace(config.vmin, config.vmax, config.vnum)
        w = self._calculate_weights(dist)
        spec = np.zeros((len(f), len(v)))
        
        for i, fi in enumerate(f):
            if fi == 0:
                continue
            FJ_array = np.zeros(len(v))
            y0 = cc_array_f[:, i]
            
            for j, vj in enumerate(v):
                k1 = 2 * np.pi * fi / vj
                m1 = j1(k1 * dist)
                FJ_array[j] = m1 @ (y0 * w) * (2 * np.pi * fi) ** 2 / vj**3
                
            spec[i, :] = FJ_array
            
        return spec

class MFJ_RR(DispersionStrategy):
    """MFJ_RR方法策略"""
    
    def compute(self, cc_array_f: np.ndarray, f: np.ndarray, 
                dist: np.ndarray, config: DispersionConfig) -> np.ndarray:
        v = np.linspace(config.vmin, config.vmax, config.vnum)
        w = self._calculate_weights(dist)
        spec = np.zeros((len(f), len(v)))
        
        for i, fi in enumerate(f):
            if fi == 0:
                continue
            FJ_array = np.zeros(len(v))
            y0 = cc_array_f[:, i]
            
            for j, vj in enumerate(v):
                k1 = 2 * np.pi * fi / vj
                m1 = hankel2(1, k1 * dist)
                FJ_array[j] = m1 @ (y0 * w) * (2 * np.pi * fi) ** 2 / vj**3
                
            spec[i, :] = FJ_array
            
        return spec

class SlantStack(DispersionStrategy):
    """Slant_stack方法策略"""
    
    def compute(self, cc_array_f: np.ndarray, f: np.ndarray, 
                dist: np.ndarray, config: DispersionConfig) -> np.ndarray:
        v = np.linspace(config.vmin, config.vmax, config.vnum)
        w = np.sqrt(dist.copy())
        spec = np.zeros((len(f), len(v)))
        
        for i, fi in enumerate(f):
            if fi == 0:
                continue
            slant_stack_array = np.zeros(len(v))
            y0 = cc_array_f[:, i]
            y2 = y0 @ (y0 * w)
            
            for j, vj in enumerate(v):
                k1 = 2 * np.pi * fi / vj
                m1_cos = np.cos(k1 * dist)
                m1_sin = np.sin(k1 * dist)
                cross = (m1_cos @ (y0 * w)) ** 2 + (m1_sin @ (y0 * w)) ** 2
                slant_stack_array[j] = cross / y2
                
            spec[i, :] = slant_stack_array
            
        return spec

class SPAC(DispersionStrategy):
    """SPAC方法策略"""
    
    def compute(self, cc_array_f: np.ndarray, f: np.ndarray, 
                dist: np.ndarray, config: DispersionConfig) -> np.ndarray:
        v = np.linspace(config.vmin, config.vmax, config.vnum)
        dist_km = dist / 1000  # 转换为公里
        spec = np.zeros((len(f), len(v)))
        
        for i, fi in enumerate(f):
            if fi == 0:
                continue
            vr0 = np.zeros(len(v))
            y0 = cc_array_f[:, i]
            y2 = y0 @ y0
            
            for j, vj in enumerate(v):
                k1 = 2 * np.pi * fi / vj
                m1 = j0(k1 * dist_km)
                m2 = m1 @ m1
                cross = m1 @ y0
                vr0[j] = 1 - (np.real(y2) - cross**2 / m2) / np.real(y2)
                
            spec[i, :] = vr0
            
        return spec

class MASW(DispersionStrategy):
    """MASW方法策略"""
    
    def compute(self, u: np.ndarray, f: np.ndarray, dist: np.ndarray, 
                config: DispersionConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MASW方法需要不同的参数，重写compute方法"""
        omega_fs = 2 * np.pi * config.sampling_rate
        N = u.shape[1]
        Lu = len(u[:, 0])
        
        # 傅里叶变换
        U = np.zeros((Lu, N), dtype=complex)
        for j in range(N):
            U[:, j] = np.fft.fft(u[:, j])
        
        # 相位归一化
        P = np.zeros((Lu, N), dtype=complex)
        for j in range(N):
            P[:, j] = np.exp(-1j * np.angle(U[:, j]))
        
        # 频率和速度范围
        omega = np.arange(0, Lu) * (omega_fs / Lu)
        cT = np.arange(config.vmin, config.vmax + config.vnum, config.vnum)
        
        # 计算幅度谱
        A = np.zeros((len(omega), len(cT)))
        for j in range(len(omega)):
            for k in range(len(cT)):
                delta = omega[j] / cT[k]
                temp = 0
                for l in range(N):
                    temp += np.exp(-1j * delta * dist[l]) * P[j, l]
                A[j, k] = np.abs(temp) / N
        
        f_plot = omega / (2 * np.pi)
        return f_plot, cT, A


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
            DispersionMethod.SPAC: SPAC,
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
        return self.strategy.compute(*args, **kwargs)
    
    def plot_spectrum(self, f: np.ndarray, c: np.ndarray, A: np.ndarray, 
                     plot_config: Optional[PlotConfig] = None, 
                     saved_figname: Optional[str] = None) -> None:
        """绘制频散谱图"""
        config = plot_config or PlotConfig()
        
        # 频率范围选择
        fmin, fmax = self.config.freqmin, self.config.freqmax
        no_fmin = np.argmin(np.abs(f - fmin))
        no_fmax = np.argmin(np.abs(f - fmax))
        
        Aplot = A[no_fmin:no_fmax, :]
        fplot = f[no_fmin:no_fmax]
        
        # 绘图
        plt.pcolormesh(fplot, c, Aplot.T / np.max(np.abs(Aplot)), 
                      cmap=config.cmap, vmin=config.vmin, vmax=config.vmax)
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

if __name__ == "__main__":
    example_usage()