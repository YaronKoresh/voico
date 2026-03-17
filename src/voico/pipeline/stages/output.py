import logging
import time

from ..core import PipelineContext, _emit
from ...io.audio_io import FileAudioIO

logger = logging.getLogger(__name__)


class OutputStage:
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        _emit(ctx, "Saving", 0.9)
        logger.info(f"Saving to {ctx.output_path}...")
        try:
            audio_io = ctx.audio_io if ctx.audio_io is not None else FileAudioIO()
            ctx.output_audio = audio_io.normalize(ctx.output_audio)
            audio_io.save(
                ctx.output_path,
                ctx.output_audio,
                ctx.sample_rate,
                ctx.bit_depth,
            )
            ctx.output_duration = len(ctx.output_audio) / ctx.sample_rate
            if ctx.diagnostic_logger:
                ctx.diagnostic_logger.log_event(
                    "output",
                    "audio_saved",
                    {
                        "output_path": ctx.output_path,
                        "bit_depth": ctx.bit_depth,
                        "duration_seconds": ctx.output_duration,
                    },
                )
        except Exception as e:
            if ctx.diagnostic_logger:
                ctx.diagnostic_logger.log_error(
                    f"Failed to save audio: {e}", "output"
                )
            raise
        _emit(ctx, "Done", 1.0)
        logger.info("Done.")
        ctx.stages_timing["output"] = time.perf_counter() - t0
        return ctx
