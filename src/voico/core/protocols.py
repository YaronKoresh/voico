from typing import Protocol, Tuple, runtime_checkable

import numpy as np

from .types import FormantTrack, PitchContour, SpectralFeatures


@runtime_checkable
class PitchAnalyzerProtocol(Protocol):
    sample_rate: int
    hop_length: int
    n_fft: int

    def detect(self, audio: np.ndarray) -> PitchContour:
        ...


@runtime_checkable
class FormantAnalyzerProtocol(Protocol):
    sample_rate: int
    hop_length: int

    def analyze(self, audio: np.ndarray, f0_contour: np.ndarray) -> FormantTrack:
        ...


@runtime_checkable
class SpectralAnalyzerProtocol(Protocol):
    sample_rate: int
    n_fft: int
    hop_length: int

    def analyze(self, audio: np.ndarray) -> SpectralFeatures:
        ...

    def compute_harmonic_stats(
        self, audio: np.ndarray, f0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...


@runtime_checkable
class ShifterProtocol(Protocol):
    sample_rate: int
    n_fft: int

    def shift_pitch(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        ...

    def shift_formants(
        self, magnitude: np.ndarray, shift_factor: float
    ) -> np.ndarray:
        ...


@runtime_checkable
class PhaseProcessorProtocol(Protocol):
    n_fft: int
    hop_length: int

    def reconstruct(
        self,
        magnitude: np.ndarray,
        n_iter: int,
    ) -> np.ndarray:
        ...
