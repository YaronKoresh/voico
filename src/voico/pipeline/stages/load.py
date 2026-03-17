import logging
import time
from ..core import PipelineContext, _emit
from ...io.audio_io import FileAudioIO

logger = logging.getLogger(__name__)


class LoadStage:
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        _emit(ctx, "Loading", 0.0)
        logger.info(f"Loading source: {ctx.input_path}")
        try:
            audio_io = ctx.audio_io if ctx.audio_io is not None else FileAudioIO()
            (audio, sample_rate) = audio_io.load(ctx.input_path)
            ctx.audio = audio_io.normalize(audio)
            ctx.sample_rate = sample_rate
            ctx.input_duration = len(ctx.audio) / sample_rate
            if ctx.diagnostic_logger:
                ctx.diagnostic_logger.log_event(
                    "load",
                    "audio_loaded",
                    {
                        "sample_rate": sample_rate,
                        "duration_seconds": ctx.input_duration,
                        "samples": len(ctx.audio),
                    },
                )
        except Exception as e:
            if ctx.diagnostic_logger:
                ctx.diagnostic_logger.log_error(
                    f"Failed to load audio: {e}", "load"
                )
            raise
        ctx.stages_timing["load"] = time.perf_counter() - t0
        return ctx
