import importlib
import logging

_logger = logging.getLogger(__name__)
try:
    importlib.util.find_spec("librosa")

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    _logger.warning(
        "librosa is not installed; scipy/numpy fallbacks will be used"
    )
try:
    importlib.util.find_spec("soundfile")

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


def get_backend_info() -> dict:
    return {"librosa": LIBROSA_AVAILABLE, "soundfile": SOUNDFILE_AVAILABLE}
