import logging
from math import gcd
from typing import Optional, Tuple

import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample_poly

from ..core.constants import AudioConstants
from ..core.errors import AudioLoadError, AudioSaveError

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_audio(
    path: str, target_sr: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    try:
        if LIBROSA_AVAILABLE:
            audio, sample_rate = librosa.load(path, sr=target_sr, mono=True)
            return audio, sample_rate

        file_sample_rate, audio = wav.read(path)
        sample_rate = file_sample_rate

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128.0) / 128.0
        elif audio.dtype not in [np.float32, np.float64]:
            audio = audio.astype(np.float32)

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        if target_sr is not None and target_sr != sample_rate:
            divisor = gcd(target_sr, sample_rate)
            up = target_sr // divisor
            down = sample_rate // divisor
            audio = resample_poly(audio, up, down).astype(np.float32)
            sample_rate = target_sr

        return audio, sample_rate
    except AudioLoadError:
        raise
    except Exception as e:
        raise AudioLoadError(f"Failed to load '{path}': {e}") from e


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > AudioConstants.EPSILON:
        return audio * (target_peak / peak)
    return audio


def save_audio(path: str, audio: np.ndarray, sample_rate: int) -> None:
    try:
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        wav.write(path, sample_rate, audio_int16)
    except Exception as e:
        raise AudioSaveError(f"Failed to save '{path}': {e}") from e
