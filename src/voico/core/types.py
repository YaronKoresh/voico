from dataclasses import dataclass

import numpy as np


@dataclass
class PitchContour:
    f0: np.ndarray
    voiced_mask: np.ndarray
    f0_mean: float
    f0_std: float
    harmonic_to_noise_ratio: float
    jitter: float = 0.0


@dataclass
class FormantTrack:
    frequencies: np.ndarray
    bandwidths: np.ndarray
    mean_frequencies: np.ndarray
    mean_bandwidths: np.ndarray


@dataclass
class SpectralFeatures:
    envelope: np.ndarray
    spectral_tilt: float


@dataclass
class VoiceProfile:
    pitch: PitchContour
    formants: FormantTrack
    spectral: SpectralFeatures
    harmonic_ratios: np.ndarray
    harmonic_energy: np.ndarray
    sample_rate: int
