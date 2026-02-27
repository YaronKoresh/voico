import logging
import os
from typing import Optional

import numpy as np

from .analysis.matcher import VoiceMatcher
from .analysis.profile import VoiceProfileBuilder
from .core.config import ConversionQuality, QualitySettings
from .core.constants import AudioConstants
from .core.errors import AnalysisError, ConversionError
from .dsp.phase import PhaseReconstructor
from .dsp.shifter import SpectralShifter
from .utils.audio_io import load_audio, normalize_audio, save_audio

logger = logging.getLogger(__name__)


class VoiceConverter:
    def __init__(
        self,
        quality: ConversionQuality = ConversionQuality.BALANCED,
    ) -> None:
        self.settings = QualitySettings.from_preset(quality)
        self.n_fft = AudioConstants.DEFAULT_N_FFT
        self.hop_length = self.n_fft // self.settings.hop_divisor

        self.profile_builder = VoiceProfileBuilder(
            sample_rate=44100,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        self.shifter: Optional[SpectralShifter] = None
        self.phase_reconstructor = PhaseReconstructor(
            self.n_fft, self.hop_length
        )

    def _match_voices(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_path: str,
    ) -> tuple:
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target file not found: {target_path}")

        try:
            logger.info(f"Loading target for matching: {target_path}")
            target_audio, target_sample_rate = load_audio(target_path)

            original_sample_rate = self.profile_builder.sample_rate
            self.profile_builder.sample_rate = target_sample_rate
            target_profile = self.profile_builder.build(target_audio, "Target")

            self.profile_builder.sample_rate = original_sample_rate
            source_profile = self.profile_builder.build(audio, "Source")

            return VoiceMatcher.match(source_profile, target_profile)
        except AnalysisError:
            raise
        except Exception as e:
            raise AnalysisError(f"Voice matching failed: {e}") from e

    def _apply_formant_shift(
        self,
        pitch_shifted_audio: np.ndarray,
        formant_shift: float,
        sample_rate: int,
    ) -> np.ndarray:
        logger.info(f"Shifting formants by factor {formant_shift}...")

        stft_matrix = np.fft.rfft(
            np.array(
                [
                    pitch_shifted_audio[i : i + self.n_fft]
                    for i in range(
                        0,
                        len(pitch_shifted_audio) - self.n_fft,
                        self.hop_length,
                    )
                ]
            )
            * np.hanning(self.n_fft),
            axis=1,
        ).T

        magnitude = np.abs(stft_matrix)
        phase_angles = np.angle(stft_matrix)

        shifted_magnitude = self.shifter.shift_formants(
            magnitude, formant_shift
        )

        if self.settings.use_advanced_phase:
            logger.info("Reconstructing phase (Griffin-Lim)...")
            return self.phase_reconstructor.reconstruct(
                shifted_magnitude,
                n_iter=self.settings.griffin_lim_iters,
            )

        reconstructed_stft = shifted_magnitude * np.exp(1j * phase_angles)
        import scipy.signal

        _, output_audio = scipy.signal.istft(
            reconstructed_stft,
            fs=sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )
        return output_audio

    def process(
        self,
        input_path: str,
        output_path: str,
        pitch_shift: float = 0.0,
        formant_shift: float = 1.0,
        target_path: Optional[str] = None,
    ) -> None:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        try:
            logger.info(f"Loading source: {input_path}")
            audio, sample_rate = load_audio(input_path)
            audio = normalize_audio(audio)
            self.profile_builder.sample_rate = sample_rate
            self.shifter = SpectralShifter(sample_rate, self.n_fft)

            if target_path:
                matched_pitch, matched_formant = self._match_voices(
                    audio, sample_rate, target_path
                )
                logger.info(
                    f"Overriding manual settings with auto-match: "
                    f"Pitch {matched_pitch:.2f}, "
                    f"Formant {matched_formant:.2f}"
                )
                pitch_shift = matched_pitch
                formant_shift = matched_formant

            logger.info(
                f"Applying: Pitch={pitch_shift:.2f}st, "
                f"Formant={formant_shift:.2f}x"
            )
            pitch_shifted_audio = self.shifter.shift_pitch(audio, pitch_shift)

            if abs(formant_shift - 1.0) > 0.01:
                output_audio = self._apply_formant_shift(
                    pitch_shifted_audio, formant_shift, sample_rate
                )
            else:
                output_audio = pitch_shifted_audio

            logger.info(f"Saving to {output_path}...")
            output_audio = normalize_audio(output_audio)
            save_audio(output_path, output_audio, sample_rate)
            logger.info("Done.")
        except (FileNotFoundError, AnalysisError):
            raise
        except Exception as e:
            raise ConversionError(f"Conversion failed: {e}") from e
