from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, get_window, medfilt, sosfiltfilt

from ..core.constants import AudioConstants
from ..core.types import FormantTrack

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class FormantAnalyzer:
    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        n_formants: int = 5,
        lpc_order: int = 14,
    ):
        self.sr = sample_rate
        self.hop_length = hop_length
        self.n_formants = n_formants
        self.analysis_sr = AudioConstants.FORMANT_ANALYSIS_SR
        self.lpc_order = lpc_order

    def analyze(self, y: np.ndarray, f0_contour: np.ndarray) -> FormantTrack:

        if LIBROSA_AVAILABLE:
            y_analysis = librosa.resample(
                y, orig_sr=self.sr, target_sr=self.analysis_sr
            )
        else:
            ratio = max(1, int(self.sr / self.analysis_sr))
            if ratio > 1:
                nyq = 0.5 * self.sr
                cutoff = 0.5 * self.analysis_sr / nyq
                if cutoff < 1.0:
                    sos_aa = butter(4, cutoff, btype="low", output="sos")
                    y_filtered = sosfiltfilt(sos_aa, y)
                else:
                    y_filtered = y
                y_analysis = y_filtered[::ratio]
            else:
                y_analysis = y.copy()

        n_frames = len(f0_contour)
        freqs = np.zeros((self.n_formants, n_frames))
        bws = np.zeros((self.n_formants, n_frames))

        frame_len = int(0.025 * self.analysis_sr)
        hop_analysis = max(1, len(y_analysis) // n_frames)

        for t in range(n_frames):
            start = t * hop_analysis
            end = start + frame_len
            if end > len(y_analysis):
                end = len(y_analysis)
                start = max(0, end - frame_len)

            frame = y_analysis[start:end].astype(np.float64)
            if len(frame) < self.lpc_order + 2:
                continue

            frame = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])

            win = get_window("hamming", len(frame))
            frame = frame * win

            f0_val = f0_contour[t] if t < len(f0_contour) else 0.0
            if (
                np.isfinite(f0_val)
                and f0_val < AudioConstants.PITCH_THRESHOLD_LOW
            ):
                order = AudioConstants.LPC_ORDER_LOW_PITCH
            else:
                order = self.lpc_order
            order = min(order, len(frame) - 2)

            lpc_coeffs = self._levinson_durbin(frame, order)
            if lpc_coeffs is None:
                continue

            f_freqs, f_bws = self._lpc_to_formants(lpc_coeffs, self.analysis_sr)

            n_found = min(len(f_freqs), self.n_formants)
            freqs[:n_found, t] = f_freqs[:n_found]
            bws[:n_found, t] = f_bws[:n_found]

        for i in range(self.n_formants):
            valid = freqs[i] > 0
            if np.sum(valid) > 5:
                kernel = min(5, np.sum(valid) // 2 * 2 + 1)
                if kernel >= 3:
                    freqs[i, valid] = medfilt(
                        freqs[i, valid], kernel_size=kernel
                    )

        mean_freqs = np.zeros(self.n_formants)
        mean_bws = np.zeros(self.n_formants)
        for i in range(self.n_formants):
            valid = freqs[i] > 0
            if np.sum(valid) > 0:
                mean_freqs[i] = np.median(freqs[i, valid])
                mean_bws[i] = np.median(bws[i, valid])
            else:
                mean_freqs[i] = AudioConstants.DEFAULT_FORMANT_FREQS[i]
                mean_bws[i] = AudioConstants.DEFAULT_FORMANT_BANDWIDTHS[i]

        return FormantTrack(freqs, bws, mean_freqs, mean_bws)

    def _levinson_durbin(
        self, frame: np.ndarray, order: int
    ) -> Optional[np.ndarray]:
        r = np.correlate(frame, frame, mode="full")
        r = r[len(frame) - 1 :]
        r = r[: order + 1]

        if r[0] < AudioConstants.EPSILON:
            return None

        try:
            a = np.zeros(order + 1)
            a[0] = 1.0
            e = r[0]

            for i in range(1, order + 1):
                lam = -np.sum(a[1:i] * r[i - 1 : 0 : -1]) - r[i]
                lam /= e

                a_new = a.copy()
                for j in range(1, i):
                    a_new[j] = a[j] + lam * a[i - j]
                a_new[i] = lam
                a = a_new

                e *= 1.0 - lam * lam
                if e <= 0:
                    return None
            return a
        except (np.linalg.LinAlgError, FloatingPointError):
            return None

    def _lpc_to_formants(
        self, lpc_coeffs: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        roots = np.roots(lpc_coeffs)
        roots = roots[np.imag(roots) >= 0]

        angles = np.angle(roots)
        freqs = angles * sr / (2.0 * np.pi)
        bws = (
            -sr / (2.0 * np.pi) * np.log(np.abs(roots) + AudioConstants.EPSILON)
        )

        valid = (
            (freqs > 90)
            & (freqs < sr / 2 - 50)
            & (bws > 0)
            & (bws < AudioConstants.MAX_FORMANT_BANDWIDTH)
        )
        freqs = freqs[valid]
        bws = bws[valid]

        order = np.argsort(freqs)
        return freqs[order], bws[order]
