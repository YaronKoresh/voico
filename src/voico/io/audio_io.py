from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from ..core.errors import AudioLoadError, AudioSaveError
from ..core.constants import AudioConstants


class AudioIO(ABC):

    @abstractmethod
    def load(self, path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        pass

    @abstractmethod
    def save(self, path: str, audio: np.ndarray, sample_rate: int, bit_depth: int = 16) -> None:
        pass

    @abstractmethod
    def normalize(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        pass

    @abstractmethod
    def get_info(self, path: str) -> dict[str, object]:
        pass


class FileAudioIO(AudioIO):

    def __init__(self) -> None:
        from ..utils.audio_io import load_audio, save_audio, normalize_audio, get_audio_info

        self._load = load_audio
        self._save = save_audio
        self._normalize = normalize_audio
        self._get_info = get_audio_info

    def load(self, path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        return self._load(path, target_sr)

    def save(self, path: str, audio: np.ndarray, sample_rate: int, bit_depth: int = 16) -> None:
        self._save(path, audio, sample_rate, bit_depth)

    def normalize(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        return self._normalize(audio, target_peak)

    def get_info(self, path: str) -> dict[str, object]:
        return self._get_info(path)
