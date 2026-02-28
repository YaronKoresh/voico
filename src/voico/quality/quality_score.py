from dataclasses import dataclass

import numpy as np

from ..core.types import VoiceProfile
from .gates import (
    FormantValidationGate,
    PitchValidationGate,
    ProfileValidationGate,
)


@dataclass
class ConversionQualityScore:
    overall_score: float
    pitch_score: float
    formant_score: float
    profile_score: float
    is_viable: bool
    critical_issues: list
    warnings: list
    recommendations: list

    def __str__(self) -> str:
        viability = "✓ VIABLE" if self.is_viable else "✗ NOT VIABLE"
        return (
            f"[{viability}] Overall Score: {self.overall_score:.1f}/100\n"
            f"  Pitch:   {self.pitch_score:.1f}/100\n"
            f"  Formant: {self.formant_score:.1f}/100\n"
            f"  Profile: {self.profile_score:.1f}/100\n"
            f"Critical Issues: {len(self.critical_issues)}\n"
            f"Warnings: {len(self.warnings)}"
        )


class QualityScorer:
    def __init__(self):
        self.min_viable_score = 30.0

    def score_profile(self, profile: VoiceProfile) -> ConversionQualityScore:
        pitch_result = PitchValidationGate(profile.pitch).validate()
        formant_result = FormantValidationGate(
            profile.formants, profile.sample_rate, profile.pitch
        ).validate()
        profile_result = ProfileValidationGate(profile).validate()

        overall = np.mean(
            [pitch_result.score, formant_result.score, profile_result.score]
        )

        critical_issues = []
        warnings = []

        if not pitch_result.passed:
            critical_issues.extend(pitch_result.issues)
        else:
            if pitch_result.score < 70:
                warnings.extend(pitch_result.issues)

        if not formant_result.passed:
            critical_issues.extend(formant_result.issues)
        else:
            if formant_result.score < 70:
                warnings.extend(formant_result.issues)

        if not profile_result.passed:
            critical_issues.extend(profile_result.issues)
        else:
            if profile_result.score < 70:
                warnings.extend(profile_result.issues)

        all_suggestions = (
            pitch_result.recovery_suggestions
            + formant_result.recovery_suggestions
            + profile_result.recovery_suggestions
        )
        unique_suggestions = list(dict.fromkeys(all_suggestions))

        is_viable = overall >= self.min_viable_score

        return ConversionQualityScore(
            overall_score=float(overall),
            pitch_score=float(pitch_result.score),
            formant_score=float(formant_result.score),
            profile_score=float(profile_result.score),
            is_viable=is_viable,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=unique_suggestions,
        )
