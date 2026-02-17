import logging
from math import gcd
from typing import Optional, Tuple

import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample_poly

from ..core.constants import AudioConstants

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_audio(
    path: str, target_sr: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Robust audio loader that handles various bit depths, stereo-to-mono conversion,
    and resampling using Librosa (if available) or Scipy fallback.
    """
    if LIBROSA_AVAILABLE:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y, sr

    try:
        sr_file, y = wav.read(path)
    except ValueError as e:
        logger.error(f"Failed to read wav file: {e}")
        raise

    sr = sr_file

    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2147483648.0
    elif y.dtype == np.uint8:
        y = (y.astype(np.float32) - 128.0) / 128.0
    elif y.dtype not in [np.float32, np.float64]:
        y = y.astype(np.float32)

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    if target_sr is not None and target_sr != sr:
        g = gcd(target_sr, sr)
        up = target_sr // g
        down = sr // g
        y = resample_poly(y, up, down).astype(np.float32)
        sr = target_sr

    return y, sr


def normalize_audio(y: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalizes audio to a target peak amplitude."""
    peak = np.max(np.abs(y))
    if peak > AudioConstants.EPSILON:
        return y * (target_peak / peak)
    return y


def apply_audio_gate(
    y: np.ndarray, threshold_db: float = -60.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a hard gate to silence noise below a threshold.
    Returns (gated_audio, non_silent_mask).
    """
    threshold_amp = 10 ** (threshold_db / 20.0)
    silence_mask = np.abs(y) < threshold_amp
    y_gated = y.copy()
    y_gated[silence_mask] = 0.0
    return y_gated, ~silence_mask


def save_audio(path: str, y: np.ndarray, sr: int):
    """Saves audio to disk, ensuring 16-bit PCM format."""

    y = np.clip(y, -1.0, 1.0)

    y_int16 = (y * 32767).astype(np.int16)
    wav.write(path, sr, y_int16)
