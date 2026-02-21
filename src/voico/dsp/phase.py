import logging
from typing import Optional

import numpy as np

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


class PhaseReconstructor:
    def __init__(self, n_fft: int, hop_length: int):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def reconstruct(
        self,
        magnitude: np.ndarray,
        n_iter: int = 32,
        init_phase: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Reconstructs phase from magnitude spectrogram using Griffin-Lim algorithm.
        """
        if LIBROSA_AVAILABLE:
            return librosa.griffinlim(
                magnitude,
                n_iter=n_iter,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                init=init_phase if init_phase is not None else "random",
            )
        else:
            return self._griffin_lim_native(magnitude, n_iter, init_phase)

    def _griffin_lim_native(
        self,
        magnitude: np.ndarray,
        n_iter: int,
        init_phase: Optional[np.ndarray],
    ) -> np.ndarray:
        """Native implementation of Griffin-Lim if Librosa is missing."""
        if init_phase is not None:
            stft_matrix = init_phase
        else:
            random_phase = 2 * np.pi * np.random.random(magnitude.shape)
            stft_matrix = magnitude * np.exp(1j * random_phase)

        for i in range(n_iter):
            y = self._istft(stft_matrix)

            stft_matrix = self._stft(y)

            phase = np.angle(stft_matrix)
            stft_matrix = magnitude * np.exp(1j * phase)

        return self._istft(stft_matrix)

    def _stft(self, y: np.ndarray) -> np.ndarray:

        import scipy.signal

        f, t, Zxx = scipy.signal.stft(
            y, nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length
        )
        return Zxx

    def _istft(self, stft_matrix: np.ndarray) -> np.ndarray:
        import scipy.signal

        t, y = scipy.signal.istft(
            stft_matrix,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )
        return y
