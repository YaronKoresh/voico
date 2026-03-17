import logging
import time

from ..core import PipelineContext, _emit
from ...analysis.profile import VoiceAnalysisEngine
from ...core.errors import AnalysisError, ProfileQualityError
from ...quality.quality_score import QualityScorer

logger = logging.getLogger(__name__)


class AnalysisStage:
    def __init__(
        self, profile_engine: VoiceAnalysisEngine, n_fft: int, hop_length: int
    ):
        self._engine = profile_engine
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._quality_scorer = QualityScorer()

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        _emit(ctx, "Analyzing source voice", 0.08)
        try:
            self._engine.sample_rate = ctx.sample_rate
            ctx.source_profile = self._engine.build(ctx.audio, "Source")
            quality = self._quality_scorer.score_profile(ctx.source_profile)
            ctx.quality_score = quality
            if ctx.diagnostic_logger:
                ctx.diagnostic_logger.log_quality_score(
                    "source_voice", quality.overall_score
                )
                ctx.diagnostic_logger.log_validation(
                    "source_profile", quality.is_viable, quality.critical_issues
                )
            if not quality.is_viable:
                error_msg = f"Source profile quality insufficient: {quality.overall_score:.1f}/100"
                if ctx.diagnostic_logger:
                    ctx.diagnostic_logger.log_error(error_msg, "analysis")
                raise ProfileQualityError(error_msg, quality.recommendations)
            logger.info(
                f"Source profile quality: {quality.overall_score:.1f}/100"
            )
            if quality.warnings and ctx.diagnostic_logger:
                for warning in quality.warnings:
                    ctx.diagnostic_logger.log_warning(warning, "analysis")
        except (AnalysisError, ProfileQualityError):
            raise
        except Exception as e:
            raise AnalysisError(f"Audio analysis failed: {e}") from e
        ctx.stages_timing["analysis"] = time.perf_counter() - t0
        return ctx
