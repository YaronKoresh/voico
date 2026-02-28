import logging

import numpy as np
import scipy.signal

from ..core.types import VoiceProfile
from ..utils.decorators import timer
from .formant import FormantAnalyzer
from .pitch import PitchAnalyzer
from .spectral import SpectralAnalyzer

logger = logging.getLogger(__name__)


class VoiceAnalysisEngine:
    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        self._sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self._build_analyzers()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int) -> None:
        if value != self._sample_rate:
            self._sample_rate = value
            self._build_analyzers()

    def _build_analyzers(self) -> None:
        self.pitch_analyzer = PitchAnalyzer(
            self._sample_rate, self.hop_length, self.n_fft
        )
        self.formant_analyzer = FormantAnalyzer(
            self._sample_rate, self.hop_length
        )
        self.spectral_analyzer = SpectralAnalyzer(
            self._sample_rate, self.n_fft, self.hop_length
        )

    def build(self, audio: np.ndarray, name: str = "Unknown") -> VoiceProfile:
        logger.info(f"Building voice profile for: {name}")

        _, _, stft_matrix = scipy.signal.stft(
            audio,
            fs=self._sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )
        shared_magnitude = np.abs(stft_matrix)

        with timer("Pitch Detection"):
            pitch_contour = self.pitch_analyzer.detect(audio)

        with timer("Formant Analysis"):
            formant_track = self.formant_analyzer.analyze(
                audio, pitch_contour.f0
            )

        with timer("Spectral Analysis"):
            spectral_features = self.spectral_analyzer.analyze_with_magnitude(shared_magnitude)
            harmonic_energy, harmonic_ratios = (
                self.spectral_analyzer.compute_harmonic_stats_with_magnitude(
                    shared_magnitude, pitch_contour.f0
                )
            )

        min_length = min(
            len(pitch_contour.f0),
            formant_track.frequencies.shape[1],
            spectral_features.envelope.shape[1],
            len(harmonic_energy),
        )

        pitch_contour.f0 = pitch_contour.f0[:min_length]
        pitch_contour.voiced_mask = pitch_contour.voiced_mask[:min_length]
        formant_track.frequencies = formant_track.frequencies[:, :min_length]
        formant_track.bandwidths = formant_track.bandwidths[:, :min_length]
        spectral_features.envelope = spectral_features.envelope[:, :min_length]
        harmonic_energy = harmonic_energy[:min_length]
        harmonic_ratios = harmonic_ratios[:min_length]

        logger.info(f"Profile built. Mean F0: {pitch_contour.f0_mean:.1f}Hz")

        return VoiceProfile(
            pitch=pitch_contour,
            formants=formant_track,
            spectral=spectral_features,
            harmonic_ratios=harmonic_ratios,
            harmonic_energy=harmonic_energy,
            sample_rate=self._sample_rate,
        )
