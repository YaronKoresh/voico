import logging
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

from ..core.constants import AudioConstants

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


class SpectralShifter:
    def __init__(self, sample_rate: int, n_fft: int):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.freq_bins = np.fft.rfftfreq(n_fft, 1 / sample_rate)

    def shift_pitch(self, y: np.ndarray, n_steps: float) -> np.ndarray:
        """
        Shifts pitch using time-domain resampling or phase vocoding.
        Note: This shifts formants as well. Formant correction must be applied separately.
        """
        if abs(n_steps) < 0.01:
            return y

        if LIBROSA_AVAILABLE:
            return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
        else:
            factor = 2 ** (n_steps / 12.0)
            int(len(y) / factor)
            return np.interp(
                np.arange(0, len(y), factor), np.arange(len(y)), y
            ).astype(y.dtype)

    def shift_formants(
        self, magnitude: np.ndarray, shift_factor: float
    ) -> np.ndarray:
        """
        Shifts the spectral envelope (formants) by stretching/compressing the frequency axis.
        factor > 1.0: Formants move up (perceived as 'smaller' vocal tract)
        factor < 1.0: Formants move down (perceived as 'larger' vocal tract)
        """
        if abs(shift_factor - 1.0) < 0.01:
            return magnitude

        n_bins, n_frames = magnitude.shape
        shifted = np.zeros_like(magnitude)

        src_freqs = np.arange(n_bins)

        target_freqs = src_freqs * shift_factor

        target_freqs = np.clip(target_freqs, 0, n_bins - 1)

        for t in range(n_frames):
            interpolator = interp1d(
                src_freqs,
                magnitude[:, t],
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            shifted[:, t] = interpolator(target_freqs)

        return shifted

    def match_spectral_tilt(
        self, source_mag: np.ndarray, target_tilt: float
    ) -> np.ndarray:
        """
        Applies a spectral tilt correction to match a target slope.
        """

        avg_spec = np.mean(source_mag, axis=1)
        valid = (self.freq_bins > 100) & (self.freq_bins < 8000)
        if np.sum(valid) < 10:
            return source_mag

        x = np.log(self.freq_bins[valid])
        y = np.log(avg_spec[valid] + AudioConstants.EPSILON)
        current_slope, _ = np.polyfit(x, y, 1)

        diff_slope = target_tilt - current_slope

        correction = np.exp(
            diff_slope * np.log(self.freq_bins + AudioConstants.EPSILON)
        )

        idx_1k = np.argmin(np.abs(self.freq_bins - 1000))
        correction /= correction[idx_1k] + AudioConstants.EPSILON

        return source_mag * correction[:, np.newaxis]
