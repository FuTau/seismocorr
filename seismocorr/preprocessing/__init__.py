# seismocorr/preprocessing/__init__.py

"""
Preprocessing Module

Provides tools for signal preprocessing, normalization, and filtering
"""

# Normalization utilities
from .normalization_utils import (
    list_available_normalization_methods,
    get_normalization_method_details
)

# Time domain normalization
from .time_norm import (
    TimeNormalizer,
    ZScoreNormalizer,
    OneBitNormalizer,
    RMSNormalizer,
    ClipNormalizer,
    NoTimeNorm,
    RAMNormalizer,
    WaterLevelNormalizer,
    CWTSoftThreshold1D,
    get_time_normalizer
)

# Frequency domain normalization
from .freq_norm import (
    FreqNormalizer,
    SpectralWhitening,
    BandWhitening,
    RmaFreqNorm,
    NoFreqNorm,
    PowerLawWhitening,
    BandwiseFreqNorm,
    ReferenceSpectrumNorm,
    ClippedSpectralWhitening,
    get_freq_normalizer
)

# Basic preprocessing functions
from .normal_func import (
    demean,
    detrend,
    taper,
    bandpass,
    lowpass,
    highpass
)

__all__ = [
    # Normalization utilities
    'list_available_normalization_methods',
    'get_normalization_method_details',
    
    # Time domain normalization
    'TimeNormalizer',
    'ZScoreNormalizer',
    'OneBitNormalizer',
    'RMSNormalizer',
    'ClipNormalizer',
    'NoTimeNorm',
    'RAMNormalizer',
    'WaterLevelNormalizer',
    'CWTSoftThreshold1D',
    'get_time_normalizer',
    
    # Frequency domain normalization
    'FreqNormalizer',
    'SpectralWhitening',
    'BandWhitening',
    'RmaFreqNorm',
    'NoFreqNorm',
    'PowerLawWhitening',
    'BandwiseFreqNorm',
    'ReferenceSpectrumNorm',
    'ClippedSpectralWhitening',
    'get_freq_normalizer',
    
    # Basic preprocessing functions
    'demean',
    'detrend',
    'taper',
    'bandpass',
    'lowpass',
    'highpass'
]