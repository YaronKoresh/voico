from typing import Tuple

import numpy as np
import scipy.signal as signal

from ..core.constants import AudioConstants
from ..core.types import SpectralFeatures
from ..utils.math_utils import safe_divide


class SpectralAnalyzer:
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = signal.get_window("hann", n_fft)

    def analyze(self, y: np.ndarray) -> SpectralFeatures:

        stft = signal.stft(
            y,
            fs=self.sr,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )[2]
        mag = np.abs(stft)

        envelope = self._compute_envelope(mag)

        tilt = self._compute_spectral_tilt(mag)

        return SpectralFeatures(envelope=envelope, spectral_tilt=tilt)

    def compute_harmonic_stats(
        self, y: np.ndarray, f0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes energy distribution relative to the fundamental frequency.
        Returns (harmonic_energy, harmonic_ratios).
        """
        stft = signal.stft(
            y,
            fs=self.sr,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )[2]
        mag = np.abs(stft)
        freqs = np.fft.rfftfreq(self.n_fft, 1 / self.sr)

        n_frames = min(mag.shape[1], len(f0))
        harmonic_energy = np.zeros(n_frames)
        harmonic_ratios = np.zeros(n_frames)

        for t in range(n_frames):
            if f0[t] <= AudioConstants.MIN_F0_HZ:
                continue

            f0_val = f0[t]

            harm_mask = np.zeros_like(freqs, dtype=bool)

            for h in range(1, 11):
                center = h * f0_val
                if center > freqs[-1]:
                    break
                idx = np.argmin(np.abs(freqs - center))
                width = max(1, int(idx * 0.05))
                harm_mask[
                    max(0, idx - width) : min(len(freqs), idx + width + 1)
                ] = True

            total_energy = np.sum(mag[:, t] ** 2)
            harm_energy = np.sum(mag[harm_mask, t] ** 2)

            harmonic_energy[t] = harm_energy
            harmonic_ratios[t] = safe_divide(
                np.array([harm_energy]), np.array([total_energy])
            )[0]

        return harmonic_energy, harmonic_ratios

    def _compute_envelope(
        self, mag_spec: np.ndarray, cepstral_coeffs: int = 20
    ) -> np.ndarray:

        log_mag = np.log(mag_spec + AudioConstants.EPSILON)

        cepstrum = np.fft.rfft(log_mag, axis=0)

        cepstrum[cepstral_coeffs:] = 0

        envelope_log = np.fft.irfft(cepstrum, axis=0, n=self.n_fft)

        envelope = np.exp(envelope_log[: mag_spec.shape[0], :])
        return envelope

    def _compute_spectral_tilt(self, mag_spec: np.ndarray) -> float:
        """Estimates global spectral tilt (slope of log magnitude vs log freq)."""
        avg_spec = np.mean(mag_spec, axis=1)
        freqs = np.fft.rfftfreq(self.n_fft, 1 / self.sr)

        valid = (freqs > 100) & (freqs < 8000)
        if np.sum(valid) < 10:
            return 0.0

        x = np.log(freqs[valid])
        y = np.log(avg_spec[valid] + AudioConstants.EPSILON)

        slope, _ = np.polyfit(x, y, 1)
        return slope
