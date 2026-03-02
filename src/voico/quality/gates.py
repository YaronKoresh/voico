from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from ..core.constants import AudioConstants
from ..core.types import FormantTrack, PitchContour, VoiceProfile


@dataclass
class ValidationResult:
    passed: bool
    score: float
    issues: List[str]
    recovery_suggestions: List[str]


class ValidationGate(ABC):
    @abstractmethod
    def validate(self) -> ValidationResult:
        raise NotImplementedError("validate must be implemented")


class PitchValidationGate(ValidationGate):
    def __init__(self, pitch: PitchContour):
        self.pitch = pitch
        self.min_f0 = AudioConstants.MIN_F0_HZ
        self.max_f0 = AudioConstants.MAX_F0_HZ

    def validate(self) -> ValidationResult:
        issues = []
        suggestions = []
        score = 100.0
        if len(self.pitch.f0) == 0:
            issues.append("Pitch contour is empty")
            suggestions.append("Provide an input with voiced speech content")
            return ValidationResult(
                passed=False,
                score=0.0,
                issues=issues,
                recovery_suggestions=suggestions,
            )
        f0_valid = ~np.isnan(self.pitch.f0)
        voiced_count = np.sum(self.pitch.voiced_mask)
        total_frames = len(self.pitch.f0)
        voiced_ratio = voiced_count / max(total_frames, 1)
        if voiced_ratio < 0.2:
            issues.append(
                f"Low voiced ratio: {voiced_ratio:.1%} (minimum: 20%)"
            )
            suggestions.append(
                "Input may be noisy, whispered, or unvoiced speech"
            )
            suggestions.append("Ensure clean audio without background noise")
            score -= 40
        nan_ratio = 1.0 - np.sum(f0_valid) / max(total_frames, 1)
        if nan_ratio > 0.3:
            issues.append(f"High NaN count: {nan_ratio:.1%} (maximum: 30%)")
            suggestions.append("Audio contains undetected pitch regions")
            suggestions.append(
                "Try manual pitch shift instead of auto-matching"
            )
            score -= 30
        if np.sum(f0_valid) > 0:
            valid_f0 = self.pitch.f0[f0_valid]
            out_of_range = np.sum(
                (valid_f0 < self.min_f0) | (valid_f0 > self.max_f0)
            )
            if out_of_range > 0:
                out_ratio = out_of_range / len(valid_f0)
                if out_ratio > 0.1:
                    issues.append(f"Out-of-range F0 values: {out_ratio:.1%}")
                    suggestions.append("May be synthesized or modified audio")
                    score -= 20
        passed = len(issues) == 0
        return ValidationResult(
            passed=passed,
            score=max(0, score),
            issues=issues,
            recovery_suggestions=suggestions,
        )


class FormantValidationGate(ValidationGate):
    def __init__(
        self, formants: FormantTrack, sample_rate: int, pitch: PitchContour
    ):
        self.formants = formants
        self.sample_rate = sample_rate
        self.pitch = pitch
        self.max_formant_freq = sample_rate / 2 - 500

    def validate(self) -> ValidationResult:
        issues = []
        suggestions = []
        score = 100.0
        num_formants = self.formants.frequencies.shape[0]
        if num_formants < 3:
            issues.append(f"Only {num_formants} formants detected (need 4-5)")
            suggestions.append(
                "Try increasing formant_tracking_order in quality settings"
            )
            suggestions.append("Ensure audio has sufficient spectral content")
            score -= 50
        if self.formants.frequencies.shape[1] > 0:
            mean_freqs = np.nanmean(self.formants.frequencies, axis=1)
            for i in range(len(mean_freqs) - 1):
                if mean_freqs[i] >= mean_freqs[i + 1]:
                    issues.append(
                        f"Formant ordering violation at F{i + 1} >= F{i + 2}"
                    )
                    suggestions.append(
                        "May indicate low SNR or algorithm instability"
                    )
                    score -= 25
                    break
            bandwidth_valid = (self.formants.bandwidths > 10) & (
                self.formants.bandwidths < AudioConstants.MAX_FORMANT_BANDWIDTH
            )
            invalid_ratio = np.sum(~bandwidth_valid) / max(
                bandwidth_valid.size, 1
            )
            if invalid_ratio > 0.2:
                issues.append(
                    f"Invalid bandwidths: {invalid_ratio:.1%} of values"
                )
                suggestions.append("LPC model may be poorly fitted")
                score -= 20
        passed = len(issues) == 0
        return ValidationResult(
            passed=passed,
            score=max(0, score),
            issues=issues,
            recovery_suggestions=suggestions,
        )


class ProfileValidationGate(ValidationGate):
    def __init__(self, profile: VoiceProfile):
        self.profile = profile
        self.min_snr_db = 10.0
        self.acceptable_tilt_range = (-2.0, 2.0)

    def validate(self) -> ValidationResult:
        issues = []
        suggestions = []
        score = 100.0
        pitch_result = PitchValidationGate(self.profile.pitch).validate()
        if not pitch_result.passed:
            issues.extend(pitch_result.issues)
            suggestions.extend(pitch_result.recovery_suggestions)
            score -= 100 - pitch_result.score
        formant_result = FormantValidationGate(
            self.profile.formants, self.profile.sample_rate, self.profile.pitch
        ).validate()
        if not formant_result.passed:
            issues.extend(formant_result.issues)
            suggestions.extend(formant_result.recovery_suggestions)
            score -= 100 - formant_result.score
        if (
            self.profile.spectral.spectral_tilt < self.acceptable_tilt_range[0]
            or self.profile.spectral.spectral_tilt
            > self.acceptable_tilt_range[1]
        ):
            issues.append(
                f"Spectral tilt out of range: {self.profile.spectral.spectral_tilt:.2f} (expected: {self.acceptable_tilt_range[0]:.1f} to {self.acceptable_tilt_range[1]:.1f})"
            )
            suggestions.append(
                "May indicate unnatural or heavily processed audio"
            )
            score -= 15
        harmonic_energy_count = len(self.profile.harmonic_energy)
        if harmonic_energy_count == 0:
            issues.append("Harmonic energy track is empty")
            suggestions.append("Increase voiced content in the source audio")
            score -= 35
            harmonic_ratio = 0.0
        else:
            harmonic_ratio = np.sum(self.profile.harmonic_energy > 0) / max(
                harmonic_energy_count, 1
            )
        if harmonic_ratio < 0.5:
            issues.append(f"Low harmonic content: {harmonic_ratio:.1%} frames")
            suggestions.append(
                "Audio may be noisy, whispered, or contain artifacts"
            )
            score -= 20
        passed = len(issues) == 0
        return ValidationResult(
            passed=passed,
            score=max(0, score),
            issues=issues,
            recovery_suggestions=suggestions,
        )
