import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

import numpy as np

from .analysis.profile import VoiceAnalysisEngine
from .core.config import ConversionQuality, QualitySettings
from .core.constants import AudioConstants
from .core.errors import AnalysisError, ConversionError, ProfileQualityError
from .core.types import ConversionReport
from .dsp.phase import PhaseProcessor
from .stream.streamer import VoiceStreamProcessor
from .quality.diagnostic import DiagnosticLogger

from .pipeline import (
    Pipeline,
    PipelineContext,
    LoadStage,
    AnalysisStage,
    MatchingStage,
    ShiftingStage,
    MetricsStage,
    OutputStage,
)

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, float], None]


# primary converter class
class VoiceConverter:
    def __init__(
        self, quality: ConversionQuality = ConversionQuality.BALANCED
    ) -> None:
        self.settings = QualitySettings.from_preset(quality)
        self.n_fft = AudioConstants.DEFAULT_N_FFT
        self.hop_length = self.n_fft // self.settings.hop_divisor
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.profile_engine = VoiceAnalysisEngine(
            sample_rate=44100, n_fft=self.n_fft, hop_length=self.hop_length
        )
        self.phase_processor = PhaseProcessor(self.n_fft, self.hop_length)

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                LoadStage(),
                AnalysisStage(self.profile_engine, self.n_fft, self.hop_length),
                MatchingStage(self.profile_engine, self.n_fft, self.hop_length),
                ShiftingStage(self.phase_processor),
                MetricsStage(),
                OutputStage(),
            ]
        )

    def process(
        self,
        input_path: str,
        output_path: str,
        pitch_shift: float = 0.0,
        formant_shift: float = 1.0,
        target_path: Optional[str] = None,
        bit_depth: int = 16,
        on_progress: Optional[ProgressCallback] = None,
        diagnostic_logger: Optional[DiagnosticLogger] = None,
    ) -> ConversionReport:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if diagnostic_logger is None:
            import uuid

            diagnostic_logger = DiagnosticLogger(str(uuid.uuid4())[:8])
        diagnostic_logger.log_input(
            input_path, output_path, self.settings.__class__.__name__
        )
        ctx = PipelineContext(
            input_path=input_path,
            output_path=output_path,
            pitch_shift=pitch_shift,
            formant_shift=formant_shift,
            target_path=target_path,
            bit_depth=bit_depth,
            on_progress=on_progress,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            settings=self.settings,
            diagnostic_logger=diagnostic_logger,
        )
        try:
            pipeline = self._build_pipeline()
            ctx = pipeline.run(ctx)
        except (FileNotFoundError, AnalysisError, ProfileQualityError):
            raise
        except Exception as e:
            diagnostic_logger.log_error(str(e), "pipeline")
            raise ConversionError(f"Conversion failed: {e}") from e
        finally:
            try:
                diagnostic_logger.finalize()
            except Exception as finalize_error:
                logger.error(
                    f"Diagnostic finalization failed: {finalize_error}"
                )
        return ConversionReport(
            output_path=output_path,
            pitch_shift_applied=ctx.pitch_shift,
            formant_shift_applied=ctx.formant_shift,
            sample_rate=ctx.sample_rate,
            input_duration_seconds=ctx.input_duration,
            output_duration_seconds=ctx.output_duration,
            snr_db=ctx.snr_db,
            spectral_centroid_deviation=ctx.spectral_centroid_deviation,
            stages_timing=ctx.stages_timing,
        )

    def process_batch(
        self,
        file_pairs: List[Tuple[str, str]],
        pitch_shift: float = 0.0,
        formant_shift: float = 1.0,
        target_path: Optional[str] = None,
        bit_depth: int = 16,
        on_file_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[str]:
        results: List[str] = []
        total = len(file_pairs)
        for idx, (inp, out) in enumerate(file_pairs):
            if on_file_progress is not None:
                on_file_progress(idx, total, inp)
            self.process(
                input_path=inp,
                output_path=out,
                pitch_shift=pitch_shift,
                formant_shift=formant_shift,
                target_path=target_path,
                bit_depth=bit_depth,
            )
            results.append(out)
        if on_file_progress is not None:
            on_file_progress(total, total, "Complete")
        return results

    async def aprocess(
        self,
        input_path: str,
        output_path: str,
        pitch_shift: float = 0.0,
        formant_shift: float = 1.0,
        target_path: Optional[str] = None,
        bit_depth: int = 16,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ConversionReport:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.process(
                input_path,
                output_path,
                pitch_shift=pitch_shift,
                formant_shift=formant_shift,
                target_path=target_path,
                bit_depth=bit_depth,
                on_progress=on_progress,
            ),
        )

    async def aprocess_batch(
        self,
        file_pairs: List[Tuple[str, str]],
        pitch_shift: float = 0.0,
        formant_shift: float = 1.0,
        target_path: Optional[str] = None,
        bit_depth: int = 16,
        on_file_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.process_batch(
                file_pairs,
                pitch_shift=pitch_shift,
                formant_shift=formant_shift,
                target_path=target_path,
                bit_depth=bit_depth,
                on_file_progress=on_file_progress,
            ),
        )

    def stream(
        self,
        audio_iterator,
        pitch_shift: Optional[float] = None,
        formant_shift: Optional[float] = None,
        quality: Optional[ConversionQuality] = None,
    ):
        effective_pitch = pitch_shift if pitch_shift is not None else 0.0
        effective_formant = formant_shift if formant_shift is not None else 1.0
        effective_quality = (
            quality if quality is not None else ConversionQuality.FAST
        )
        processor = VoiceStreamProcessor(
            sample_rate=44100,
            pitch_shift=effective_pitch,
            formant_shift=effective_formant,
            quality=effective_quality,
        )
        return processor.stream(audio_iterator)
