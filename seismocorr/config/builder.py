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
