import logging

_logger = logging.getLogger(__name__)

try:
    import librosa as _librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    _librosa = None
    LIBROSA_AVAILABLE = False
    _logger.warning("librosa is not installed; scipy/numpy fallbacks will be used")

try:
    import soundfile as _soundfile
    SOUNDFILE_AVAILABLE = True
except ImportError:
    _soundfile = None
    SOUNDFILE_AVAILABLE = False


def get_backend_info() -> dict:
    return {
        "librosa": LIBROSA_AVAILABLE,
        "soundfile": SOUNDFILE_AVAILABLE,
    }
