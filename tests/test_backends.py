from voico.backends import get_backend_info, LIBROSA_AVAILABLE, SOUNDFILE_AVAILABLE


def test_backend_info_keys() -> None:
    info = get_backend_info()
    assert "librosa" in info and "soundfile" in info
    assert info["librosa"] == LIBROSA_AVAILABLE
    assert info["soundfile"] == SOUNDFILE_AVAILABLE
