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
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_formants = n_formants
        self.analysis_sample_rate = AudioConstants.FORMANT_ANALYSIS_SR
        self.lpc_order = lpc_order

    def analyze(
        self, audio: np.ndarray, f0_contour: np.ndarray
    ) -> FormantTrack:
        if LIBROSA_AVAILABLE:
            resampled_audio = librosa.resample(
                audio,
                orig_sr=self.sample_rate,
                target_sr=self.analysis_sample_rate,
            )
        else:
            ratio = max(1, int(self.sample_rate / self.analysis_sample_rate))
            if ratio > 1:
                nyquist = 0.5 * self.sample_rate
                cutoff = 0.5 * self.analysis_sample_rate / nyquist
                if cutoff < 1.0:
                    anti_alias_filter = butter(
                        4, cutoff, btype="low", output="sos"
                    )
                    filtered_audio = sosfiltfilt(anti_alias_filter, audio)
                else:
                    filtered_audio = audio
                resampled_audio = filtered_audio[::ratio]
            else:
                resampled_audio = audio.copy()

        n_frames = len(f0_contour)
        frequencies = np.zeros((self.n_formants, n_frames))
        bandwidths = np.zeros((self.n_formants, n_frames))

        frame_length = int(0.025 * self.analysis_sample_rate)
        analysis_hop = max(1, len(resampled_audio) // n_frames)

        for t in range(n_frames):
            start = t * analysis_hop
            end = start + frame_length
            if end > len(resampled_audio):
                end = len(resampled_audio)
                start = max(0, end - frame_length)

            frame = resampled_audio[start:end].astype(np.float64)
            if len(frame) < self.lpc_order + 2:
                continue

            frame = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])

            window = get_window("hamming", len(frame))
            frame = frame * window

            f0_value = f0_contour[t] if t < len(f0_contour) else 0.0
            if (
                np.isfinite(f0_value)
                and f0_value < AudioConstants.PITCH_THRESHOLD_LOW
            ):
                order = AudioConstants.LPC_ORDER_LOW_PITCH
            else:
                order = self.lpc_order
            order = min(order, len(frame) - 2)

            lpc_coefficients = self._levinson_durbin(frame, order)
            if lpc_coefficients is None:
                continue

            frame_frequencies, frame_bandwidths = self._lpc_to_formants(
                lpc_coefficients, self.analysis_sample_rate
            )

            n_found = min(len(frame_frequencies), self.n_formants)
            frequencies[:n_found, t] = frame_frequencies[:n_found]
            bandwidths[:n_found, t] = frame_bandwidths[:n_found]

        for i in range(self.n_formants):
            valid = frequencies[i] > 0
            if np.sum(valid) > 5:
                kernel = min(5, np.sum(valid) // 2 * 2 + 1)
                if kernel >= 3:
                    frequencies[i, valid] = medfilt(
                        frequencies[i, valid], kernel_size=kernel
                    )

        mean_frequencies = np.zeros(self.n_formants)
        mean_bandwidths = np.zeros(self.n_formants)
        for i in range(self.n_formants):
            valid = frequencies[i] > 0
            if np.sum(valid) > 0:
                mean_frequencies[i] = np.median(frequencies[i, valid])
                mean_bandwidths[i] = np.median(bandwidths[i, valid])
            else:
                mean_frequencies[i] = AudioConstants.DEFAULT_FORMANT_FREQS[i]
                mean_bandwidths[i] = AudioConstants.DEFAULT_FORMANT_BANDWIDTHS[
                    i
                ]

        return FormantTrack(
            frequencies, bandwidths, mean_frequencies, mean_bandwidths
        )

    def _levinson_durbin(
        self, frame: np.ndarray, order: int
    ) -> Optional[np.ndarray]:
        autocorrelation = np.correlate(frame, frame, mode="full")
        autocorrelation = autocorrelation[len(frame) - 1 :]
        autocorrelation = autocorrelation[: order + 1]

        if autocorrelation[0] < AudioConstants.EPSILON:
            return None

        try:
            coefficients = np.zeros(order + 1)
            coefficients[0] = 1.0
            prediction_error = autocorrelation[0]

            for i in range(1, order + 1):
                reflection_coeff = (
                    -np.sum(coefficients[1:i] * autocorrelation[i - 1 : 0 : -1])
                    - autocorrelation[i]
                )
                reflection_coeff /= prediction_error

                updated = coefficients.copy()
                for j in range(1, i):
                    updated[j] = (
                        coefficients[j] + reflection_coeff * coefficients[i - j]
                    )
                updated[i] = reflection_coeff
                coefficients = updated

                prediction_error *= 1.0 - reflection_coeff * reflection_coeff
                if prediction_error <= 0:
                    return None
            return coefficients
        except (np.linalg.LinAlgError, FloatingPointError):
            return None

    def _lpc_to_formants(
        self, lpc_coefficients: np.ndarray, sample_rate: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        roots = np.roots(lpc_coefficients)
        roots = roots[np.imag(roots) >= 0]

        angles = np.angle(roots)
        frequencies = angles * sample_rate / (2.0 * np.pi)
        bandwidths = (
            -sample_rate
            / (2.0 * np.pi)
            * np.log(np.abs(roots) + AudioConstants.EPSILON)
        )

        valid = (
            (frequencies > 90)
            & (frequencies < sample_rate / 2 - 50)
            & (bandwidths > 0)
            & (bandwidths < AudioConstants.MAX_FORMANT_BANDWIDTH)
        )
        frequencies = frequencies[valid]
        bandwidths = bandwidths[valid]

        sort_order = np.argsort(frequencies)
        return frequencies[sort_order], bandwidths[sort_order]
