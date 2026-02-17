import logging
from typing import Tuple

import numpy as np

from ..core.types import VoiceProfile

logger = logging.getLogger(__name__)


class VoiceMatcher:
    @staticmethod
    def match(
        source: VoiceProfile, target: VoiceProfile
    ) -> Tuple[float, float]:
        """
        Compares source and target profiles to calculate optimal shift parameters.
        Returns: (pitch_shift_semitones, formant_shift_factor)
        """

        if source.pitch.f0_mean > 0 and target.pitch.f0_mean > 0:
            pitch_ratio = target.pitch.f0_mean / source.pitch.f0_mean
            semitones = 12.0 * np.log2(pitch_ratio)
        else:
            logger.warning(
                "Invalid pitch means detected, defaulting to 0 semitones."
            )
            semitones = 0.0

        src_formants = source.formants.mean_frequencies[:3]
        tgt_formants = target.formants.mean_frequencies[:3]

        if len(src_formants) > 0 and len(tgt_formants) > 0:
            ratios = tgt_formants / (src_formants + 1e-6)
            formant_factor = float(np.median(ratios))

            formant_factor = np.clip(formant_factor, 0.5, 2.0)
        else:
            formant_factor = 1.0

        logger.info(
            f"Auto-Match Result: Shift {semitones:.2f} st, Formant Factor {formant_factor:.2f}x"
        )
        return semitones, formant_factor
