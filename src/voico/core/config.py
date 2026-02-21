from dataclasses import dataclass
from enum import Enum


class ConversionQuality(Enum):
    TURBO = "turbo"
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    ULTRA = "ultra"
    MASTER = "master"


@dataclass
class QualitySettings:
    hop_divisor: int
    griffin_lim_iters: int
    envelope_smoothing: int
    formant_tracking_order: int
    spectral_detail_preservation: float
    use_advanced_phase: bool
    use_formant_correction: bool

    @classmethod
    def from_preset(cls, quality: ConversionQuality) -> "QualitySettings":
        presets = {
            ConversionQuality.TURBO: cls(2, 16, 9, 14, 0.15, False, False),
            ConversionQuality.FAST: cls(4, 32, 5, 14, 0.2, False, True),
            ConversionQuality.BALANCED: cls(4, 64, 3, 14, 0.3, True, True),
            ConversionQuality.HIGH: cls(4, 100, 2, 14, 0.4, True, True),
            ConversionQuality.ULTRA: cls(8, 200, 1, 14, 0.5, True, True),
            ConversionQuality.MASTER: cls(8, 500, 1, 16, 0.6, True, True),
        }
        return presets[quality]
