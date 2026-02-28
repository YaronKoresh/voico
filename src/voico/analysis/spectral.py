from typing import Optional, Tuple

import numpy as np
import scipy.signal as signal

from ..core.constants import AudioConstants
from ..core.types import SpectralFeatures
from ..utils.math_utils import safe_divide


class SpectralAnalyzer:
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self._cached_magnitude: Optional[np.ndarray] = None
        self._cached_audio_id: Optional[int] = None

    def _compute_stft_magnitude(self, audio: np.ndarray) -> np.ndarray:
        stft_matrix = signal.stft(
            audio,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )[2]
        return np.abs(stft_matrix)

    def _get_magnitude(self, audio: np.ndarray) -> np.ndarray:
        if (
            self._cached_audio_id == id(audio)
            and self._cached_magnitude is not None
        ):
            return self._cached_magnitude
        magnitude = self._compute_stft_magnitude(audio)
        self._cached_magnitude = magnitude
        self._cached_audio_id = id(audio)
        return magnitude

    def analyze(self, audio: np.ndarray) -> SpectralFeatures:
        magnitude = self._get_magnitude(audio)
        envelope = self._compute_cepstral_envelope(magnitude)
        tilt = self._compute_spectral_tilt(magnitude)
        return SpectralFeatures(envelope=envelope, spectral_tilt=tilt)

    def compute_harmonic_stats(
        self, audio: np.ndarray, f0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        magnitude = self._get_magnitude(audio)
        frequency_bins = np.fft.rfftfreq(self.n_fft, 1 / self.sample_rate)
        freq_resolution = (
            frequency_bins[1] if len(frequency_bins) > 1 else 1.0
        )

        n_frames = min(magnitude.shape[1], len(f0))
        total_energy = np.sum(magnitude[:, :n_frames] ** 2, axis=0)

        harmonic_energy = np.zeros(n_frames)
        harmonic_ratios = np.zeros(n_frames)

        voiced_mask = f0[:n_frames] > AudioConstants.MIN_F0_HZ
        voiced_indices = np.where(voiced_mask)[0]

        for t in voiced_indices:
            fundamental = f0[t]
            harmonic_mask = np.zeros(len(frequency_bins), dtype=bool)

            for h in range(1, 11):
                center = h * fundamental
                if center > frequency_bins[-1]:
                    break
                idx = min(
                    int(round(center / freq_resolution)),
                    len(frequency_bins) - 1,
                )
                width = max(1, int(idx * 0.05))
                lo = max(0, idx - width)
                hi = min(len(frequency_bins), idx + width + 1)
                harmonic_mask[lo:hi] = True

            frame_harmonic_energy = np.sum(
                magnitude[harmonic_mask, t] ** 2
            )
            harmonic_energy[t] = frame_harmonic_energy
            if total_energy[t] > 0:
                harmonic_ratios[t] = (
                    frame_harmonic_energy / total_energy[t]
                )

        return harmonic_energy, harmonic_ratios

    def analyze_with_magnitude(self, magnitude: np.ndarray) -> SpectralFeatures:
        envelope = self._compute_cepstral_envelope(magnitude)
        tilt = self._compute_spectral_tilt(magnitude)
        return SpectralFeatures(envelope=envelope, spectral_tilt=tilt)

    def compute_harmonic_stats_with_magnitude(
        self, magnitude: np.ndarray, f0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        frequency_bins = np.fft.rfftfreq(self.n_fft, 1 / self.sample_rate)
        freq_resolution = (
            frequency_bins[1] if len(frequency_bins) > 1 else 1.0
        )

        n_frames = min(magnitude.shape[1], len(f0))
        total_energy = np.sum(magnitude[:, :n_frames] ** 2, axis=0)

        harmonic_energy = np.zeros(n_frames)
        harmonic_ratios = np.zeros(n_frames)

        voiced_mask = f0[:n_frames] > AudioConstants.MIN_F0_HZ
        voiced_indices = np.where(voiced_mask)[0]

        for t in voiced_indices:
            fundamental = f0[t]
            harmonic_mask = np.zeros(len(frequency_bins), dtype=bool)

            for h in range(1, 11):
                center = h * fundamental
                if center > frequency_bins[-1]:
                    break
                idx = min(
                    int(round(center / freq_resolution)),
                    len(frequency_bins) - 1,
                )
                width = max(1, int(idx * 0.05))
                lo = max(0, idx - width)
                hi = min(len(frequency_bins), idx + width + 1)
                harmonic_mask[lo:hi] = True

            frame_harmonic_energy = np.sum(
                magnitude[harmonic_mask, t] ** 2
            )
            harmonic_energy[t] = frame_harmonic_energy
            if total_energy[t] > 0:
                harmonic_ratios[t] = (
                    frame_harmonic_energy / total_energy[t]
                )

        return harmonic_energy, harmonic_ratios

    def _compute_cepstral_envelope(
        self,
        magnitude_spectrum: np.ndarray,
        cepstral_coefficients: int = 20,
    ) -> np.ndarray:
        log_magnitude = np.log(magnitude_spectrum + AudioConstants.EPSILON)
        cepstrum = np.fft.rfft(log_magnitude, axis=0)
        cepstrum[cepstral_coefficients:] = 0
        envelope_log = np.fft.irfft(cepstrum, axis=0, n=self.n_fft)
        envelope = np.exp(envelope_log[: magnitude_spectrum.shape[0], :])
        return envelope

    def _compute_spectral_tilt(self, magnitude_spectrum: np.ndarray) -> float:
        average_spectrum = np.mean(magnitude_spectrum, axis=1)
        frequency_bins = np.fft.rfftfreq(self.n_fft, 1 / self.sample_rate)

        valid = (frequency_bins > 100) & (frequency_bins < 8000)
        if np.sum(valid) < 10:
            return 0.0

        log_frequencies = np.log(frequency_bins[valid])
        log_magnitudes = np.log(
            average_spectrum[valid] + AudioConstants.EPSILON
        )

        slope, _ = np.polyfit(log_frequencies, log_magnitudes, 1)
        return slope
