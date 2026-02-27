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
        initial_phase: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if LIBROSA_AVAILABLE:
            return librosa.griffinlim(
                magnitude,
                n_iter=n_iter,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                init=(initial_phase if initial_phase is not None else "random"),
            )
        return self._griffin_lim_native(magnitude, n_iter, initial_phase)

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
        import scipy.signal

        _, _, stft_result = scipy.signal.stft(
            audio,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )
        return stft_result

    def _inverse_stft(self, stft_matrix: np.ndarray) -> np.ndarray:
        import scipy.signal

        _, audio = scipy.signal.istft(
            stft_matrix,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )
        return audio
