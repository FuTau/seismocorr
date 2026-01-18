# config/builder.py

class CorrelationConfig:
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

    def validate(self):
        if not self.sampling_rate or not self.hdf5_path or not self.reference_channel:
            raise ValueError("Missing required config fields")

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

