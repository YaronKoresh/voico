import logging
from typing import Optional

import numpy as np
import scipy.signal

from ..backends import LIBROSA_AVAILABLE

if LIBROSA_AVAILABLE:
    import librosa

logger = logging.getLogger(__name__)

_RTPGHI_GAMMA_SCALE = 0.25


class PhaseProcessor:
    def __init__(self, n_fft: int, hop_length: int):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def reconstruct(
        self,
        magnitude: np.ndarray,
        n_iter: int = 0,
        initial_phase: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if n_iter == 0:
            return self.reconstruct_rtpghi(magnitude)

        if LIBROSA_AVAILABLE:
            return librosa.griffinlim(
                magnitude,
                n_iter=n_iter,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                init=(initial_phase if initial_phase is not None else "random"),
            )
        return self._griffin_lim_native(magnitude, n_iter, initial_phase)

    def reconstruct_rtpghi(self, magnitude: np.ndarray) -> np.ndarray:
        n_bins, n_frames = magnitude.shape
        gamma = _RTPGHI_GAMMA_SCALE * (self.hop_length ** 2) / self.n_fft

        log_mag = np.log(np.maximum(magnitude.astype(np.float64), 1e-8))

        dlog_dt = np.gradient(log_mag, axis=1)

        omega = (
            2.0
            * np.pi
            * np.arange(n_bins, dtype=np.float64)
            * self.hop_length
            / self.n_fft
        )

        inst_freq = omega[:, np.newaxis] + gamma * dlog_dt

        phase = np.zeros((n_bins, n_frames), dtype=np.float64)
        phase[:, 0] = np.random.uniform(0.0, 2.0 * np.pi, n_bins)

        for t in range(1, n_frames):
            phase[:, t] = phase[:, t - 1] + inst_freq[:, t - 1]

        stft_matrix = magnitude.astype(np.float64) * np.exp(1j * phase)
        return self._inverse_stft(stft_matrix)

    def _griffin_lim_native(
        self,
        magnitude: np.ndarray,
        n_iter: int,
        initial_phase: Optional[np.ndarray],
    ) -> np.ndarray:
        if initial_phase is not None:
            stft_matrix = initial_phase
        else:
            random_phase = 2 * np.pi * np.random.random(magnitude.shape)
            stft_matrix = magnitude * np.exp(1j * random_phase)

        for _ in range(n_iter):
            audio = self._inverse_stft(stft_matrix)
            stft_matrix = self._forward_stft(audio)
            phase = np.angle(stft_matrix)
            stft_matrix = magnitude * np.exp(1j * phase)

        return self._inverse_stft(stft_matrix)

    def _forward_stft(self, audio: np.ndarray) -> np.ndarray:
        _, _, stft_result = scipy.signal.stft(
            audio,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )
        return stft_result

    def _inverse_stft(self, stft_matrix: np.ndarray) -> np.ndarray:
        _, audio = scipy.signal.istft(
            stft_matrix,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )
        return audio
