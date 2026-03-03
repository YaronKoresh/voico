from voico.backends import (
    LIBROSA_AVAILABLE,
    SOUNDFILE_AVAILABLE,
    get_backend_info,
)


def test_backend_info_keys() -> None:
    info = get_backend_info()
    assert "librosa" in info and "soundfile" in info
    assert info["librosa"] == LIBROSA_AVAILABLE
    assert info["soundfile"] == SOUNDFILE_AVAILABLE
