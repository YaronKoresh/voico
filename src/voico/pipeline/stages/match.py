import logging
import os
import time

from ..core import PipelineContext, _emit
from ...analysis.profile import VoiceAnalysisEngine
from ...core.errors import AnalysisError, ProfileQualityError
from ...matching.matcher import VoiceMatcher
from ...quality.quality_score import QualityScorer

logger = logging.getLogger(__name__)


class MatchingStage:
    def __init__(
        self, profile_engine: VoiceAnalysisEngine, n_fft: int, hop_length: int
    ):
        self._engine = profile_engine
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._quality_scorer = QualityScorer()

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.target_path:
            return ctx
        if not os.path.exists(ctx.target_path):
            raise FileNotFoundError(f"Target file not found: {ctx.target_path}")
        t0 = time.perf_counter()
        _emit(ctx, "Matching voices", 0.1)
        try:
            from ...utils.audio_io import load_audio

            logger.info(f"Loading target for matching: {ctx.target_path}")
            (target_audio, target_sr) = load_audio(ctx.target_path)
            target_engine = VoiceAnalysisEngine(
                target_sr, self._n_fft, self._hop_length
            )
            ctx.target_profile = target_engine.build(target_audio, "Target")
            target_quality = self._quality_scorer.score_profile(
                ctx.target_profile
            )
            if ctx.diagnostic_logger:
                ctx.diagnostic_logger.log_quality_score(
                    "target_voice", target_quality.overall_score
                )
                ctx.diagnostic_logger.log_validation(
                    "target_profile",
                    target_quality.is_viable,
                    target_quality.critical_issues,
                )
            if not target_quality.is_viable:
                error_msg = f"Target profile quality insufficient: {target_quality.overall_score:.1f}/100"
                if ctx.diagnostic_logger:
                    ctx.diagnostic_logger.log_error(error_msg, "matching")
                raise ProfileQualityError(
                    error_msg, target_quality.recommendations
                )
            (ctx.pitch_shift, ctx.formant_shift) = VoiceMatcher.match(
                ctx.source_profile, ctx.target_profile
            )
            logger.info(
                f"Auto-match: Pitch {ctx.pitch_shift:.2f}st, Formant {ctx.formant_shift:.2f}x"
            )
        except (AnalysisError, ProfileQualityError):
            raise
        except Exception as e:
            raise AnalysisError(f"Voice matching failed: {e}") from e
        ctx.stages_timing["matching"] = time.perf_counter() - t0
        return ctx
