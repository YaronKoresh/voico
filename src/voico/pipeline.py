import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import scipy.signal

from .analysis.profile import VoiceAnalysisEngine
from .core.config import QualitySettings
from .core.errors import AnalysisError, ProfileQualityError
from .core.types import VoiceProfile
from .dsp.phase import PhaseProcessor
from .quality.diagnostic import DiagnosticLogger
from .quality.quality_score import ConversionQualityScore, QualityScorer

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
    stages_timing: Dict[str, float] = field(default_factory=dict)
    source_profile: Optional[VoiceProfile] = None
    target_profile: Optional[VoiceProfile] = None
    quality_score: Optional[ConversionQualityScore] = None
    diagnostic_logger: Optional[DiagnosticLogger] = None


def _emit(ctx: PipelineContext, step: str, fraction: float) -> None:
    if ctx.on_progress is not None:
        ctx.on_progress(step, fraction)


def _compute_snr(original: np.ndarray, processed: np.ndarray) -> float:
    n = min(len(original), len(processed))
    if n == 0:
        return 0.0
    orig = original[:n].astype(np.float64)
    proc = processed[:n].astype(np.float64)
    signal_power = np.mean(orig**2)
    noise_power = np.mean((orig - proc) ** 2)
    if noise_power < 1e-10:
        return 60.0
    return float(10.0 * np.log10(max(signal_power / noise_power, 1e-10)))


def _compute_spectral_centroid(audio: np.ndarray, sr: int, n_fft: int) -> float:
    if len(audio) < n_fft:
        return 0.0
    spectrum = np.abs(np.fft.rfft(audio[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    total = np.sum(spectrum)
    if total < 1e-10:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)


class LoadStage:
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        _emit(ctx, "Loading", 0.0)
        logger.info(f"Loading source: {ctx.input_path}")
        try:
            from .utils.audio_io import load_audio, normalize_audio

            (audio, sample_rate) = load_audio(ctx.input_path)
            ctx.audio = normalize_audio(audio)
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
            from .utils.audio_io import load_audio

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
            from .matching.matcher import VoiceMatcher

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


class ShiftingStage:
    def __init__(self, phase_processor: "PhaseProcessor"):
        self._phase_processor = phase_processor

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        _emit(ctx, "Shifting pitch", 0.4)
        logger.info(
            f"Applying: Pitch={ctx.pitch_shift:.2f}st, Formant={ctx.formant_shift:.2f}x"
        )
        from .dsp.shifter import SpectralProcessor

        processor = SpectralProcessor(ctx.sample_rate, ctx.n_fft)
        pitch_shifted = processor.shift_pitch(ctx.audio, ctx.pitch_shift)
        if abs(ctx.formant_shift - 1.0) > 0.01:
            _emit(ctx, "Shifting formants", 0.6)
            logger.info(f"Shifting formants by factor {ctx.formant_shift}...")
            (_, _, stft_matrix) = scipy.signal.stft(
                pitch_shifted,
                fs=ctx.sample_rate,
                nperseg=ctx.n_fft,
                noverlap=ctx.n_fft - ctx.hop_length,
            )
            magnitude = np.abs(stft_matrix)
            phase_angles = np.angle(stft_matrix)
            shifted_magnitude = processor.shift_formants(
                magnitude, ctx.formant_shift
            )
            if ctx.settings.use_advanced_phase:
                logger.info("Reconstructing phase...")
                if ctx.settings.griffin_lim_iters <= 32:
                    ctx.output_audio = self._phase_processor.reconstruct_rtpghi(
                        shifted_magnitude
                    )
                else:
                    ctx.output_audio = self._phase_processor.reconstruct(
                        shifted_magnitude, n_iter=ctx.settings.griffin_lim_iters
                    )
            else:
                reconstructed_stft = shifted_magnitude * np.exp(
                    1j * phase_angles
                )
                (_, ctx.output_audio) = scipy.signal.istft(
                    reconstructed_stft,
                    fs=ctx.sample_rate,
                    nperseg=ctx.n_fft,
                    noverlap=ctx.n_fft - ctx.hop_length,
                )
        else:
            ctx.output_audio = pitch_shifted
        ctx.stages_timing["shifting"] = time.perf_counter() - t0
        return ctx


class MetricsStage:
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        ctx.snr_db = _compute_snr(ctx.audio, ctx.output_audio)
        in_centroid = _compute_spectral_centroid(
            ctx.audio, ctx.sample_rate, ctx.n_fft
        )
        out_centroid = _compute_spectral_centroid(
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


class OutputStage:
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        _emit(ctx, "Saving", 0.9)
        logger.info(f"Saving to {ctx.output_path}...")
        try:
            from .utils.audio_io import normalize_audio, save_audio

            ctx.output_audio = normalize_audio(ctx.output_audio)
            save_audio(
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


class Pipeline:
    def __init__(self, stages: list[object]) -> None:
        self._stages = stages

    def run(self, ctx: PipelineContext) -> PipelineContext:
        for stage in self._stages:
            ctx = stage.execute(ctx)
        return ctx
