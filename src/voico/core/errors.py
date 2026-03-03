from collections.abc import Sequence
from typing import Optional


class VoicoError(Exception):
    def __init__(
        self,
        message: str,
        recovery_suggestions: Optional[Sequence[str]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.recovery_suggestions: list[str] = list(recovery_suggestions or [])

    def with_suggestions(self, suggestions: Sequence[str]) -> "VoicoError":
        self.recovery_suggestions = list(suggestions)
        return self


class AudioLoadError(VoicoError):
    __slots__ = ()


class AudioSaveError(VoicoError):
    __slots__ = ()


class AnalysisError(VoicoError):
    __slots__ = ()


class PitchDetectionError(AnalysisError):
    __slots__ = ()


class FormantAnalysisError(AnalysisError):
    __slots__ = ()


class SpectralAnalysisError(AnalysisError):
    __slots__ = ()


class ProfileQualityError(AnalysisError):
    __slots__ = ()


class ConversionError(VoicoError):
    __slots__ = ()


class MatchingError(ConversionError):
    __slots__ = ()


class ValidationError(ConversionError):
    __slots__ = ()
