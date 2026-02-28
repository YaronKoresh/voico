from enum import Enum

from pydantic import BaseModel, field_validator


class ConversionQuality(Enum):
    TURBO = "turbo"
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    ULTRA = "ultra"
    MASTER = "master"


class QualitySettings(BaseModel):
    hop_divisor: int
    griffin_lim_iters: int
    envelope_smoothing: int
    formant_tracking_order: int
    spectral_detail_preservation: float
    use_advanced_phase: bool
    use_formant_correction: bool

    model_config = {"frozen": True}

    @field_validator("hop_divisor")
    @classmethod
    def hop_divisor_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"hop_divisor must be > 0, got {v}")
        return v

    @field_validator("griffin_lim_iters")
    @classmethod
    def griffin_lim_iters_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"griffin_lim_iters must be > 0, got {v}")
        return v

    @field_validator("formant_tracking_order")
    @classmethod
    def formant_tracking_order_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"formant_tracking_order must be > 0, got {v}")
        return v

    @field_validator("spectral_detail_preservation")
    @classmethod
    def spectral_detail_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"spectral_detail_preservation must be in [0, 1], got {v}")
        return v

    @classmethod
    def from_preset(cls, quality: ConversionQuality) -> "QualitySettings":
        presets = {
            ConversionQuality.TURBO: cls(
                hop_divisor=2,
                griffin_lim_iters=16,
                envelope_smoothing=9,
                formant_tracking_order=14,
                spectral_detail_preservation=0.15,
                use_advanced_phase=False,
                use_formant_correction=False,
            ),
            ConversionQuality.FAST: cls(
                hop_divisor=4,
                griffin_lim_iters=32,
                envelope_smoothing=5,
                formant_tracking_order=14,
                spectral_detail_preservation=0.2,
                use_advanced_phase=False,
                use_formant_correction=True,
            ),
            ConversionQuality.BALANCED: cls(
                hop_divisor=4,
                griffin_lim_iters=64,
                envelope_smoothing=3,
                formant_tracking_order=14,
                spectral_detail_preservation=0.3,
                use_advanced_phase=True,
                use_formant_correction=True,
            ),
            ConversionQuality.HIGH: cls(
                hop_divisor=4,
                griffin_lim_iters=100,
                envelope_smoothing=2,
                formant_tracking_order=14,
                spectral_detail_preservation=0.4,
                use_advanced_phase=True,
                use_formant_correction=True,
            ),
            ConversionQuality.ULTRA: cls(
                hop_divisor=8,
                griffin_lim_iters=200,
                envelope_smoothing=1,
                formant_tracking_order=14,
                spectral_detail_preservation=0.5,
                use_advanced_phase=True,
                use_formant_correction=True,
            ),
            ConversionQuality.MASTER: cls(
                hop_divisor=8,
                griffin_lim_iters=500,
                envelope_smoothing=1,
                formant_tracking_order=16,
                spectral_detail_preservation=0.6,
                use_advanced_phase=True,
                use_formant_correction=True,
            ),
        }
        return presets[quality]
