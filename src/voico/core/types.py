from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class PitchContour:
    f0: np.ndarray
    voiced_mask: np.ndarray
    f0_mean: float
    f0_std: float
    harmonic_to_noise_ratio: float


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


@dataclass
class ConversionReport:
    output_path: str
    pitch_shift_applied: float
    formant_shift_applied: float
    sample_rate: int
    input_duration_seconds: float
    output_duration_seconds: float
    snr_db: float
    spectral_centroid_deviation: float
    stages_timing: Dict[str, float]
