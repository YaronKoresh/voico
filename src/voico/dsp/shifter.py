import logging

import numpy as np
from scipy.ndimage import map_coordinates

from ..core.constants import AudioConstants

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


class SpectralShifter:
    def __init__(self, sample_rate: int, n_fft: int):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.frequency_bins = np.fft.rfftfreq(n_fft, 1 / sample_rate)

    def shift_pitch(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        if abs(semitones) < 0.01:
            return audio

        if LIBROSA_AVAILABLE:
            return librosa.effects.pitch_shift(
                audio, sr=self.sample_rate, n_steps=semitones
            )

        factor = 2 ** (semitones / 12.0)
        return np.interp(
            np.arange(0, len(audio), factor),
            np.arange(len(audio)),
            audio,
        ).astype(audio.dtype)

    def shift_formants(
        self, magnitude: np.ndarray, shift_factor: float
    ) -> np.ndarray:
        if abs(shift_factor - 1.0) < 0.01:
            return magnitude

        n_bins, n_frames = magnitude.shape

        target_bins = np.clip(np.arange(n_bins) * shift_factor, 0, n_bins - 1)

        row_coords = np.broadcast_to(
            target_bins[:, np.newaxis], (n_bins, n_frames)
        )
        col_coords = np.broadcast_to(
            np.arange(n_frames)[np.newaxis, :], (n_bins, n_frames)
        )

        return map_coordinates(
            magnitude,
            [row_coords, col_coords],
            order=1,
            mode="constant",
            cval=0.0,
        )

    def match_spectral_tilt(
        self, source_magnitude: np.ndarray, target_tilt: float
    ) -> np.ndarray:
        average_spectrum = np.mean(source_magnitude, axis=1)
        valid = (self.frequency_bins > 100) & (self.frequency_bins < 8000)
        if np.sum(valid) < 10:
            return source_magnitude

        log_frequencies = np.log(self.frequency_bins[valid])
        log_magnitudes = np.log(
            average_spectrum[valid] + AudioConstants.EPSILON
        )
        current_slope, _ = np.polyfit(log_frequencies, log_magnitudes, 1)

        slope_difference = target_tilt - current_slope

        correction = np.exp(
            slope_difference
            * np.log(self.frequency_bins + AudioConstants.EPSILON)
        )

        idx_1khz = np.argmin(np.abs(self.frequency_bins - 1000))
        correction /= correction[idx_1khz] + AudioConstants.EPSILON

        return source_magnitude * correction[:, np.newaxis]
