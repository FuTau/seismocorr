# config/builder.py
from enum import Enum
from dataclasses import dataclass
from seismocorr.config.default import SUPPORTED_METHODS


class CorrelationConfig:
    """
    初始化互相关配置

    Args:
        method: 计算方法 ('time-domain', 'freq-domain', 'deconv', 'coherency')
        max_lag: 最大滞后时间（秒）
        nfft: FFT长度
    """
    def __init__(self):
        self.sampling_rate = None
        self.freq_min = self.freq_max = None
        self.cc_window_seconds = 3600
        self.hdf5_path = ""
        self.reference_channel = ""  # 如 "STA01.CHZ"
        self.target_channels_pattern = "*"  # 或正则表达式
        self.normalization = 'one-bit'
        self.stacking_method = 'linear'  # 'pws', 'robust', 'selective'
        self.output_dir = "./output"
        self.dx = 10
        self.max_lag = 2
        self.n_parallel = 4
        self.use_gpu = False
        self.method = "time-domain"
        self.nfft = None

    def validate(self):
        if not self.sampling_rate or not self.hdf5_path or not self.reference_channel:
            raise ValueError("缺失关键参数，请检查")
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"不支持的计算方法: {self.method!r}。请从 {SUPPORTED_METHODS} 中选择")
        if self.max_lag is not None:
            if isinstance(self.max_lag, bool) or not isinstance(self.max_lag, (int, float)):
                raise TypeError(f"max_lag 类型有误，应为 float/int")
            if float(self.max_lag) < 0:
                raise ValueError(f"max_lag 应该 >= 0，当前为: {self.max_lag!r}")
        if self.nfft is not None:
            if isinstance(self.nfft, bool) or not isinstance(self.nfft, int):
                raise TypeError(f"nfft 类型有误，应为 int")
            if self.nfft <= 0:
                raise ValueError(f"nfft 应为正整数，当前为: {self.nfft!r}")

    def to_dict(self):
        return self.__dict__.copy()

class CorrelationConfigBuilder:
    def __init__(self):
        self.config = CorrelationConfig()

    def set_hdf5(self, path):
        self.config.hdf5_path = path
        return self

    def set_sampling_rate(self, sr):
        self.config.sampling_rate = sr
        return self

    def set_bandpass(self, fmin, fmax):
        self.config.freq_min, self.config.freq_max = fmin, fmax
        return self

    def set_reference(self, channel_key):
        self.config.reference_channel = channel_key
        return self

    def set_targets(self, pattern="*"):
        self.config.target_channels_pattern = pattern
        return self
    
    def set_dx(self, dx):
        self.config.dx = dx
        return self

    def set_method(self, method):
        self.config.method = method
        return self

    def set_nfft(self, nfft):
        self.config.nfft = nfft
        return self

    def set_max_lag(self, lag):
        self.config.max_lag = lag
        return self

    def use_normalization(self, method):
        valid_methods = ['zscore', 'one-bit', 'rms', 'no']
        if method not in valid_methods:
            raise ValueError(f"Normalization must be one of {valid_methods}")
        self.config.normalization = method
        return self

    def use_stacking(self, method):
        self.config.stacking_method = method
        return self

    def set_output(self, path):
        self.config.output_dir = path
        return self

    def build(self):
        self.config.validate()
        return self.config


# ===================
# SPFI Config
# ===================

class SPFIConfig:
    def __init__(self):
        self.geometry = "2d"
        self.assumption = "station_avg"

        # ray_avg 相关
        self.grid_x = None
        self.grid_y = None
        self.pair_sampling = None
        self.random_state = None

        # inversion 相关
        self.regularization = "l2"
        self.alpha = 0.0
        self.beta = 0.0

    def validate(self):
        # geometry 合法性检查
        if self.geometry not in ["1d", "2d"]:
            raise ValueError('geometry must be "1d" or "2d"')

        # assumption 合法性检查
        if self.assumption not in ["station_avg", "ray_avg"]:
            raise ValueError('assumption must be "station_avg" or "ray_avg"')

        # 正则化方式检查
        if self.regularization not in ["none", "l2", "l1", "l1_l2"]:
            raise ValueError('regularization must be one of ["none","l2","l1","l1_l2"]')

        # alpha/beta 非负
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        if self.beta < 0:
            raise ValueError("beta must be >= 0")

        # ray_avg 的 grid_x/grid_y 检查
        if self.assumption == "ray_avg" and self.geometry == "2d":
            if self.grid_x is None or self.grid_y is None:
                raise ValueError("ray_avg & geometry=2d requires grid_x and grid_y")

        # if self.assumption == "ray_avg" and self.geometry == "1d":
        #     pass


class SPFIConfigBuilder:
    def __init__(self):
        self.config = SPFIConfig()

    def set_geometry(self, geometry):
        self.config.geometry = geometry
        return self

    def set_assumption(self, assumption):
        self.config.assumption = assumption
        return self

    def set_grid(self, grid_x, grid_y):
        self.config.grid_x = grid_x
        self.config.grid_y = grid_y
        return self

    def set_pair_sampling(self, pair_sampling, random_state=None):
        self.config.pair_sampling = pair_sampling
        self.config.random_state = random_state
        return self

    def set_regularization(self, regularization):
        self.config.regularization = regularization
        return self

    def set_l2(self, alpha):
        self.config.alpha = float(alpha)
        return self

    def set_l1(self, beta):
        self.config.beta = float(beta)
        return self

    def set_l1_l2(self, alpha, beta):
        self.config.alpha = float(alpha)
        self.config.beta = float(beta)
        return self

    def build(self):
        self.config.validate()
        return self.config


# ===================
# 频散成像 Config
# ===================
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


# ===================
# beamforming Config
# ===================
class BeamformingConfig:
    def __init__(self):
        self.fs = None
        self.fmin = 1.0
        self.fmax = 20.0
        self.frame_len_s = 4.0
        self.hop_s = 2.0
        self.window = "hann"
        self.whiten = True
        self.eps = 1e-12

    def validate(self):
        if self.fs is None:
            raise ValueError("fs 不能为空")

        if self.fs <= 0:
            raise ValueError("fs 必须 > 0")

        if self.fmin < 0 or self.fmax <= self.fmin:
            raise ValueError("频带非法")

class BeamformingConfigBuilder:
    def __init__(self):
        self.config = BeamformingConfig()

    def set_sampling_rate(self, fs):
        self.config.fs = fs
        return self

    def set_bandpass(self, fmin, fmax):
        self.config.fmin = fmin
        self.config.fmax = fmax
        return self

    def set_frame(self, frame_len_s, hop_s):
        self.config.frame_len_s = frame_len_s
        self.config.hop_s = hop_s
        return self

    def set_window(self, window):
        self.config.window = window
        return self

    def use_whitening(self, flag=True):
        self.config.whiten = flag
        return self

    def build(self):
        self.config.validate()
        return self.config


# ===================
# three_stations_interferometry Config
# ===================
@dataclass
class ThreeStationConfig:
    """
    mode:
      - "correlation": 二次干涉固定用互相关
      - "convolution": 二次干涉固定用卷积
      - "auto": 线性阵列自动分段：
          k 在 i/j 中间 -> convolution
          否则 -> correlation
    """
    mode: str = "auto"                      # "correlation" | "convolution" | "auto"
    second_stage_nfft: int = None
    max_lag2: float = None