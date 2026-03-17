import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..core.config import QualitySettings
from ..core.types import VoiceProfile
from ..io.audio_io import AudioIO, FileAudioIO
from ..quality.diagnostic import DiagnosticLogger
from ..quality.quality_score import ConversionQualityScore

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, float], None]


@dataclass
class PipelineContext:
    input_path: str
    output_path: str
    pitch_shift: float
    formant_shift: float
    target_path: Optional[str]
    bit_depth: int
    on_progress: Optional[ProgressCallback]
    n_fft: int
    hop_length: int
    settings: QualitySettings
    audio: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    output_audio: Optional[np.ndarray] = None
    input_duration: float = 0.0
    output_duration: float = 0.0
    snr_db: float = 0.0
    spectral_centroid_deviation: float = 0.0
    stages_timing: dict[str, float] = field(default_factory=dict)
    source_profile: Optional[VoiceProfile] = None
    target_profile: Optional[VoiceProfile] = None
    quality_score: Optional[ConversionQualityScore] = None
    diagnostic_logger: Optional[DiagnosticLogger] = None
    audio_io: Optional[AudioIO] = None


def _emit(ctx: PipelineContext, step: str, fraction: float) -> None:
    if ctx.on_progress is not None:
        ctx.on_progress(step, fraction)


class Pipeline:
    def __init__(self, stages: list[object]) -> None:
        self._stages = stages

    def run(self, ctx: PipelineContext) -> PipelineContext:
        for stage in self._stages:
            ctx = stage.execute(ctx)
        return ctx
