import logging
import os
from math import gcd
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample_poly

from ..backends import LIBROSA_AVAILABLE, SOUNDFILE_AVAILABLE
from ..core.constants import AudioConstants
from ..core.errors import AudioLoadError, AudioSaveError

if LIBROSA_AVAILABLE:
    import librosa

if SOUNDFILE_AVAILABLE:
    import soundfile as sf

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}


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


def normalize_audio(
    audio: np.ndarray, target_peak: float = 0.95
) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > AudioConstants.EPSILON:
        return audio * (target_peak / peak)
    return audio


def save_audio(
    path: str,
    audio: np.ndarray,
    sample_rate: int,
    bit_depth: int = 16,
) -> None:
    try:
        audio = np.clip(audio, -1.0, 1.0)
        ext = os.path.splitext(path)[1].lower()

        if ext == ".wav":
            if bit_depth == 16:
                audio_out = (audio * 32767).astype(np.int16)
            elif bit_depth == 32:
                audio_out = audio.astype(np.float32)
            else:
                raise AudioSaveError(
                    f"Unsupported WAV bit depth: {bit_depth}. Use 16 or 32."
                )
            wav.write(path, sample_rate, audio_out)
        elif ext in (".flac", ".ogg"):
            if not SOUNDFILE_AVAILABLE:
                raise AudioSaveError(
                    f"Saving {ext} requires soundfile. "
                    f"Install with: pip install voico[full]"
                )
            subtype = "FLOAT" if bit_depth == 32 else "PCM_16"
            sf.write(path, audio, sample_rate, subtype=subtype)
        else:
            raise AudioSaveError(
                f"Unsupported output format: {ext}. "
                f"Supported: .wav, .flac, .ogg"
            )
    except AudioSaveError:
        raise
    except Exception as e:
        raise AudioSaveError(f"Failed to save '{path}': {e}") from e


def get_audio_info(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        raise AudioLoadError(f"File not found: {path}")

    info: Dict[str, object] = {"path": path}
    ext = os.path.splitext(path)[1].lower()
    info["format"] = ext.lstrip(".")
    info["file_size_bytes"] = os.path.getsize(path)

    try:
        if SOUNDFILE_AVAILABLE:
            sf_info = sf.info(path)
            info["sample_rate"] = sf_info.samplerate
            info["channels"] = sf_info.channels
            info["frames"] = sf_info.frames
            info["duration_seconds"] = sf_info.duration
            info["subtype"] = sf_info.subtype
        elif LIBROSA_AVAILABLE:
            audio, sr = librosa.load(path, sr=None, mono=False)
            if len(audio.shape) > 1:
                info["channels"] = audio.shape[0]
                info["frames"] = audio.shape[1]
            else:
                info["channels"] = 1
                info["frames"] = len(audio)
            info["sample_rate"] = sr
            info["duration_seconds"] = round(
                int(info["frames"]) / sr, 3
            )
        else:
            file_sr, audio = wav.read(path)
            info["sample_rate"] = file_sr
            if len(audio.shape) > 1:
                info["channels"] = audio.shape[1]
                info["frames"] = audio.shape[0]
            else:
                info["channels"] = 1
                info["frames"] = len(audio)
            info["duration_seconds"] = round(
                int(info["frames"]) / file_sr, 3
            )
            info["dtype"] = str(audio.dtype)
    except Exception as e:
        raise AudioLoadError(
            f"Failed to read info for '{path}': {e}"
        ) from e

    return info
