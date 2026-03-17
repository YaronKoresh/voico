import logging
import time

import numpy as np
import scipy.signal

from ..core import PipelineContext, _emit

logger = logging.getLogger(__name__)


class ShiftingStage:
    def __init__(self, phase_processor: "PhaseProcessor"):
        self._phase_processor = phase_processor

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        _emit(ctx, "Shifting pitch", 0.4)
        logger.info(
            f"Applying: Pitch={ctx.pitch_shift:.2f}st, Formant={ctx.formant_shift:.2f}x"
        )
        from ...dsp.shifter import SpectralProcessor

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
