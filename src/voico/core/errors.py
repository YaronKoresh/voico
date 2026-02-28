class VoicoError(Exception):
    def __init__(self, message: str, recovery_suggestions: list = None):
        super().__init__(message)
        self.message = message
        self.recovery_suggestions = recovery_suggestions or []

    def with_suggestions(self, suggestions: list) -> "VoicoError":
        self.recovery_suggestions = suggestions
        return self


class AudioLoadError(VoicoError):
    pass


class AudioSaveError(VoicoError):
    pass


class AnalysisError(VoicoError):
    pass


class PitchDetectionError(AnalysisError):
    pass


class FormantAnalysisError(AnalysisError):
    pass


class SpectralAnalysisError(AnalysisError):
    pass


class ProfileQualityError(AnalysisError):
    pass


class ConversionError(VoicoError):
    pass


class MatchingError(ConversionError):
    pass


class ValidationError(ConversionError):
    pass
