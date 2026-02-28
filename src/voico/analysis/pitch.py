import logging
from typing import Tuple

import numpy as np

from ..backends import LIBROSA_AVAILABLE
from ..core.constants import AudioConstants
from ..core.types import PitchContour

if LIBROSA_AVAILABLE:
    import librosa

logger = logging.getLogger(__name__)


class PitchAnalyzer:
    def __init__(self, sample_rate: int, hop_length: int, n_fft: int):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.fmin = AudioConstants.MIN_F0_HZ
        self.fmax = AudioConstants.MAX_F0_HZ

    def detect(self, audio: np.ndarray) -> PitchContour:
        if LIBROSA_AVAILABLE:
            f0, voiced_flag, voiced_prob = librosa.pyin(
                audio,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                fill_na=np.nan,
            )
        else:
            f0, voiced_prob = self._autocorrelation_detect(audio)
            voiced_flag = voiced_prob > 0.3

        f0_clean = f0.copy()
        valid_mask = ~np.isnan(f0_clean)

        if np.sum(valid_mask) > 0:
            f0_mean = float(np.median(f0_clean[valid_mask]))
            f0_std = float(np.std(f0_clean[valid_mask]))
        else:
            f0_mean = 150.0
            f0_std = 0.0

        hnr = self._compute_hnr(audio, f0_mean)

        return PitchContour(
            f0_clean, voiced_flag, f0_mean, f0_std, hnr
        )

    def _compute_hnr(
        self, audio: np.ndarray, f0_mean: float
    ) -> float:
        if f0_mean <= 0 or np.isnan(f0_mean):
            return 0.0

        period = int(self.sample_rate / f0_mean)
        if period < 2 or period >= len(audio) // 2:
            return 0.0

        n_samples = min(len(audio), self.sample_rate)
        frame = audio[:n_samples].astype(np.float64)

        autocorr = np.correlate(frame, frame, mode="full")
        center = len(frame) - 1
        r0 = autocorr[center]
        if r0 < AudioConstants.EPSILON:
            return 0.0

        r_period = autocorr[center + period]
        noise_power = r0 - r_period

        if noise_power < AudioConstants.EPSILON:
            return 40.0

        hnr = 10.0 * np.log10(
            max(r_period, AudioConstants.EPSILON) / noise_power
        )
        return float(np.clip(hnr, 0.0, 40.0))

    def _autocorrelation_detect(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_frames = len(audio) // self.hop_length
        f0 = np.full(n_frames, np.nan)
        confidence = np.zeros(n_frames)

        min_lag = int(self.sample_rate / self.fmax)
        max_lag = int(self.sample_rate / self.fmin)
        window_size = max_lag * 2

        for t in range(n_frames):
            start = t * self.hop_length
            end = start + window_size
            if end > len(audio):
                end = len(audio)
                if end - start < min_lag * 2:
                    break
            frame = audio[start:end].astype(np.float64)
            frame_max_lag = min(max_lag, len(frame) // 2)

            fft_size = 1
            while fft_size < 2 * len(frame):
                fft_size *= 2
            frame_fft = np.fft.rfft(frame, n=fft_size)
            acf = np.fft.irfft(frame_fft * np.conj(frame_fft))[
                : frame_max_lag + 1
            ]

            frame_sq_cumsum = np.cumsum(frame**2)
            n = len(frame)

            difference = np.zeros(frame_max_lag + 1)
            taus = np.arange(1, frame_max_lag + 1)
            energy_left = frame_sq_cumsum[
                np.clip(n - taus - 1, 0, n - 1)
            ]
            energy_right = (
                frame_sq_cumsum[n - 1] - frame_sq_cumsum[taus - 1]
            )
            difference[1:] = np.maximum(
                energy_left + energy_right - 2 * acf[1 : frame_max_lag + 1],
                0.0,
            )

            cumsum_diff = np.cumsum(difference[1:])
            normalized_difference = np.ones(frame_max_lag + 1)
            tau_values = np.arange(
                1, frame_max_lag + 1, dtype=np.float64
            )
            nonzero_mask = cumsum_diff > 0
            normalized_difference[1:][nonzero_mask] = (
                difference[1:][nonzero_mask]
                * tau_values[nonzero_mask]
                / cumsum_diff[nonzero_mask]
            )

            threshold = 0.1
            best_tau = -1.0
            search_max = min(max_lag, frame_max_lag)

            search_range = normalized_difference[min_lag : search_max + 1]
            below_threshold = np.where(search_range < threshold)[0]

            if len(below_threshold) > 0:
                tau = below_threshold[0] + min_lag
                if 0 < tau < search_max:
                    alpha = normalized_difference[tau - 1]
                    beta = normalized_difference[tau]
                    gamma = normalized_difference[tau + 1]
                    denom = 2.0 * (alpha - 2.0 * beta + gamma)
                    if abs(denom) > 1e-12:
                        delta = (alpha - gamma) / denom
                    else:
                        delta = 0.0
                    best_tau = tau + delta
                else:
                    best_tau = float(tau)

            if best_tau < 0:
                idx = np.argmin(search_range)
                if search_range[idx] < 0.3:
                    best_tau = float(idx + min_lag)

            if best_tau > 0:
                f0[t] = self.sample_rate / best_tau
                tau_idx = round(best_tau)
                if tau_idx < len(normalized_difference):
                    confidence[t] = max(
                        0.0, 1.0 - normalized_difference[tau_idx]
                    )
                else:
                    confidence[t] = 0.0

        return f0, confidence
