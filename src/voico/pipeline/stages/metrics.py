import time

from ..core import PipelineContext
from ..utils import compute_snr, compute_spectral_centroid


class MetricsStage:
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        ctx.snr_db = compute_snr(ctx.audio, ctx.output_audio)
        in_centroid = compute_spectral_centroid(
            ctx.audio, ctx.sample_rate, ctx.n_fft
        )
        out_centroid = compute_spectral_centroid(
            ctx.output_audio, ctx.sample_rate, ctx.n_fft
        )
        if in_centroid > 1e-10:
            ctx.spectral_centroid_deviation = (
                abs(out_centroid - in_centroid) / in_centroid
            )
        if ctx.diagnostic_logger:
            ctx.diagnostic_logger.log_quality_score("snr_db", ctx.snr_db)
            ctx.diagnostic_logger.log_quality_score(
                "spectral_centroid_deviation", ctx.spectral_centroid_deviation
            )
        ctx.stages_timing["metrics"] = time.perf_counter() - t0
        return ctx
