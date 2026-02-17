import logging
from typing import Optional

import numpy as np

from ..core.constants import AudioConstants
from ..core.types import VoiceProfile
from ..utils.decorators import timer
from .formant import FormantAnalyzer
from .pitch import PitchDetector
from .spectral import SpectralAnalyzer

logger = logging.getLogger(__name__)


class VoiceProfileBuilder:
    def __init__(
        self, sample_rate: int, n_fft: int = 2048, hop_length: int = 512
    ):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.pitch_detector = PitchDetector(sample_rate, hop_length, n_fft)
        self.formant_analyzer = FormantAnalyzer(sample_rate, hop_length)
        self.spectral_analyzer = SpectralAnalyzer(
            sample_rate, n_fft, hop_length
        )

    def build(self, y: np.ndarray, name: str = "Unknown") -> VoiceProfile:
        logger.info(f"Building voice profile for: {name}")

        with timer("Pitch Detection"):
            pitch_contour = self.pitch_detector.detect(y)

        with timer("Formant Analysis"):
            formant_track = self.formant_analyzer.analyze(y, pitch_contour.f0)

        with timer("Spectral Analysis"):
            spectral_features = self.spectral_analyzer.analyze(y)
            harm_energy, harm_ratios = (
                self.spectral_analyzer.compute_harmonic_stats(
                    y, pitch_contour.f0
                )
            )

        min_len = min(
            len(pitch_contour.f0),
            formant_track.frequencies.shape[1],
            spectral_features.envelope.shape[1],
            len(harm_energy),
        )

        pitch_contour.f0 = pitch_contour.f0[:min_len]
        pitch_contour.voiced_mask = pitch_contour.voiced_mask[:min_len]
        formant_track.frequencies = formant_track.frequencies[:, :min_len]
        formant_track.bandwidths = formant_track.bandwidths[:, :min_len]
        spectral_features.envelope = spectral_features.envelope[:, :min_len]
        harm_energy = harm_energy[:min_len]
        harm_ratios = harm_ratios[:min_len]

        logger.info(f"Profile built. Mean F0: {pitch_contour.f0_mean:.1f}Hz")

        return VoiceProfile(
            pitch=pitch_contour,
            formants=formant_track,
            spectral=spectral_features,
            harmonic_ratios=harm_ratios,
            harmonic_energy=harm_energy,
            sample_rate=self.sr,
        )
