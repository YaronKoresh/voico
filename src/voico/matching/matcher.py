import logging
from typing import Tuple

import numpy as np

from ..core.constants import AudioConstants
from ..core.types import VoiceProfile

logger = logging.getLogger(__name__)


class VoiceMatcher:
    @staticmethod
    def match(
        source: VoiceProfile, target: VoiceProfile
    ) -> Tuple[float, float]:
        if source.pitch.f0_mean > 0 and target.pitch.f0_mean > 0:
            pitch_ratio = target.pitch.f0_mean / source.pitch.f0_mean
            semitones = 12.0 * np.log2(pitch_ratio)
        else:
            logger.warning(
                "Invalid pitch means detected, defaulting to 0 semitones."
            )
            semitones = 0.0

        source_formants = source.formants.mean_frequencies[:3]
        target_formants = target.formants.mean_frequencies[:3]

        if len(source_formants) > 0 and len(target_formants) > 0:
            ratios = target_formants / (source_formants + AudioConstants.EPSILON)
            formant_factor = float(np.median(ratios))
            formant_factor = np.clip(formant_factor, 0.5, 2.0)
        else:
            formant_factor = 1.0

        logger.info(
            f"Auto-Match Result: Shift {semitones:.2f} st, "
            f"Formant Factor {formant_factor:.2f}x"
        )
        return semitones, formant_factor
