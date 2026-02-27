from typing import Tuple

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

    def analyze(self, audio: np.ndarray) -> SpectralFeatures:
        stft_matrix = signal.stft(
            audio,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )[2]
        magnitude = np.abs(stft_matrix)

        envelope = self._compute_cepstral_envelope(magnitude)
        tilt = self._compute_spectral_tilt(magnitude)

        return SpectralFeatures(envelope=envelope, spectral_tilt=tilt)

    def compute_harmonic_stats(
        self, audio: np.ndarray, f0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        stft_matrix = signal.stft(
            audio,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )[2]
        magnitude = np.abs(stft_matrix)
        frequency_bins = np.fft.rfftfreq(self.n_fft, 1 / self.sample_rate)

        n_frames = min(magnitude.shape[1], len(f0))
        harmonic_energy = np.zeros(n_frames)
        harmonic_ratios = np.zeros(n_frames)

        for t in range(n_frames):
            if f0[t] <= AudioConstants.MIN_F0_HZ:
                continue

            fundamental = f0[t]
            harmonic_mask = np.zeros_like(frequency_bins, dtype=bool)

            for harmonic in range(1, 11):
                center = harmonic * fundamental
                if center > frequency_bins[-1]:
                    break
                idx = np.argmin(np.abs(frequency_bins - center))
                width = max(1, int(idx * 0.05))
                harmonic_mask[
                    max(0, idx - width) : min(
                        len(frequency_bins), idx + width + 1
                    )
                ] = True

            total_energy = np.sum(magnitude[:, t] ** 2)
            frame_harmonic_energy = np.sum(magnitude[harmonic_mask, t] ** 2)

            harmonic_energy[t] = frame_harmonic_energy
            harmonic_ratios[t] = safe_divide(
                np.array([frame_harmonic_energy]),
                np.array([total_energy]),
            )[0]

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
