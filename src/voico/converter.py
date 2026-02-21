import logging
import os
from typing import Optional

import numpy as np

from .analysis.matcher import VoiceMatcher
from .analysis.profile import VoiceProfileBuilder
from .core.config import ConversionQuality, QualitySettings
from .core.constants import AudioConstants
from .dsp.phase import PhaseReconstructor
from .dsp.shifter import SpectralShifter
from .utils.audio_io import load_audio, normalize_audio, save_audio

logger = logging.getLogger(__name__)


class VoiceConverter:
    def __init__(self, quality: ConversionQuality = ConversionQuality.BALANCED):
        self.settings = QualitySettings.from_preset(quality)
        self.n_fft = AudioConstants.DEFAULT_N_FFT
        self.hop_length = self.n_fft // self.settings.hop_divisor

        self.builder = VoiceProfileBuilder(
            sample_rate=44100, n_fft=self.n_fft, hop_length=self.hop_length
        )
        self.shifter = None
        self.reconstructor = PhaseReconstructor(self.n_fft, self.hop_length)

    def process(
        self,
        input_path: str,
        output_path: str,
        pitch_shift: float = 0.0,
        formant_shift: float = 1.0,
        target_path: Optional[str] = None,
    ):

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(f"Loading source: {input_path}")
        y, sr = load_audio(input_path)
        y = normalize_audio(y)
        self.builder.sr = sr
        self.shifter = SpectralShifter(sr, self.n_fft)

        if target_path:
            if not os.path.exists(target_path):
                raise FileNotFoundError(f"Target file not found: {target_path}")

            logger.info(f"Loading target for matching: {target_path}")
            y_target, sr_target = load_audio(target_path)

            original_sr = self.builder.sr
            self.builder.sr = sr_target
            target_profile = self.builder.build(y_target, "Target")

            self.builder.sr = original_sr
            source_profile = self.builder.build(y, "Source")

            calc_pitch, calc_formant = VoiceMatcher.match(
                source_profile, target_profile
            )

            logger.info(
                f"Overriding manual settings with auto-match: Pitch {calc_pitch:.2f}, Formant {calc_formant:.2f}"
            )
            pitch_shift = calc_pitch
            formant_shift = calc_formant

        logger.info(
            f"Applying: Pitch={pitch_shift:.2f}st, Formant={formant_shift:.2f}x"
        )
        y_pitch_shifted = self.shifter.shift_pitch(y, pitch_shift)

        if abs(formant_shift - 1.0) > 0.01:
            logger.info(f"Shifting formants by factor {formant_shift}...")

            stft = np.fft.rfft(
                np.array(
                    [
                        y_pitch_shifted[i : i + self.n_fft]
                        for i in range(
                            0,
                            len(y_pitch_shifted) - self.n_fft,
                            self.hop_length,
                        )
                    ]
                )
                * np.hanning(self.n_fft),
                axis=1,
            ).T

            mag = np.abs(stft)
            phase = np.angle(stft)

            mag_shifted = self.shifter.shift_formants(mag, formant_shift)

            if self.settings.use_advanced_phase:
                logger.info("Reconstructing phase (Griffin-Lim)...")
                y_final = self.reconstructor.reconstruct(
                    mag_shifted, n_iter=self.settings.griffin_lim_iters
                )
            else:
                stft_new = mag_shifted * np.exp(1j * phase)
                import scipy.signal

                _, y_final = scipy.signal.istft(
                    stft_new,
                    fs=sr,
                    nperseg=self.n_fft,
                    noverlap=self.n_fft - self.hop_length,
                )
        else:
            y_final = y_pitch_shifted

        logger.info(f"Saving to {output_path}...")
        y_final = normalize_audio(y_final)
        save_audio(output_path, y_final, sr)
        logger.info("Done.")
