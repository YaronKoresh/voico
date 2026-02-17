import logging
from typing import Tuple

import numpy as np

from ..core.constants import AudioConstants
from ..core.types import PitchContour

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


class PitchDetector:
    def __init__(self, sample_rate: int, hop_length: int, n_fft: int):
        self.sr = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.fmin = AudioConstants.MIN_F0_HZ
        self.fmax = AudioConstants.MAX_F0_HZ

    def detect(self, y: np.ndarray) -> PitchContour:

        if LIBROSA_AVAILABLE:
            f0, voiced_flag, voiced_prob = librosa.pyin(
                y,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=self.sr,
                hop_length=self.hop_length,
                fill_na=np.nan,
            )
        else:
            f0, voiced_prob = self._autocorrelation_detect(y)
            voiced_flag = voiced_prob > 0.3

        f0_clean = f0.copy()
        valid_mask = ~np.isnan(f0_clean)

        if np.sum(valid_mask) > 0:
            f0_mean = float(np.median(f0_clean[valid_mask]))
            f0_std = float(np.std(f0_clean[valid_mask]))
        else:
            f0_mean = 150.0
            f0_std = 0.0

        hnr = 20.0

        return PitchContour(f0_clean, voiced_flag, f0_mean, f0_std, hnr)

    def _autocorrelation_detect(
        self, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """YIN-based autocorrelation pitch detection fallback."""
        n_frames = len(y) // self.hop_length
        f0 = np.full(n_frames, np.nan)
        confidence = np.zeros(n_frames)

        min_lag = int(self.sr / self.fmax)
        max_lag = int(self.sr / self.fmin)
        W = max_lag * 2

        for t in range(n_frames):
            start = t * self.hop_length
            end = start + W
            if end > len(y):
                end = len(y)
                if end - start < min_lag * 2:
                    break
            frame = y[start:end].astype(np.float64)
            frame_max_lag = min(max_lag, len(frame) // 2)

            d = np.zeros(frame_max_lag + 1)
            for tau in range(1, frame_max_lag + 1):
                diff = frame[:frame_max_lag] - frame[tau : tau + frame_max_lag]
                if len(diff) == 0:
                    break
                d[tau] = np.sum(diff**2)

            d_prime = np.ones(frame_max_lag + 1)
            running_sum = 0.0
            for tau in range(1, frame_max_lag + 1):
                running_sum += d[tau]
                if running_sum > 0:
                    d_prime[tau] = d[tau] * tau / running_sum

            threshold = 0.1
            best_tau = -1
            search_max = min(max_lag, frame_max_lag)
            for tau in range(min_lag, search_max + 1):
                if d_prime[tau] < threshold:
                    if tau > 0 and tau < search_max:
                        alpha = d_prime[tau - 1]
                        beta = d_prime[tau]
                        gamma = d_prime[tau + 1]
                        denom = 2.0 * (alpha - 2.0 * beta + gamma)
                        if abs(denom) > 1e-12:
                            delta = (alpha - gamma) / denom
                        else:
                            delta = 0.0
                        best_tau = tau + delta
                    else:
                        best_tau = float(tau)
                    break

            if best_tau < 0:
                search = d_prime[min_lag : search_max + 1]
                idx = np.argmin(search)
                if search[idx] < 0.3:
                    best_tau = float(idx + min_lag)

            if best_tau > 0:
                f0[t] = self.sr / best_tau
                tau_idx = int(round(best_tau))
                if tau_idx < len(d_prime):
                    confidence[t] = max(0.0, 1.0 - d_prime[tau_idx])
                else:
                    confidence[t] = 0.0

        return f0, confidence
