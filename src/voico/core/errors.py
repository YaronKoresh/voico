class VoicoError(Exception):
    pass


class AudioLoadError(VoicoError):
    pass


class AudioSaveError(VoicoError):
    pass


class AnalysisError(VoicoError):
    pass


class ConversionError(VoicoError):
    pass
