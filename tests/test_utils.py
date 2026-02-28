import os
import tempfile

import numpy as np
import pytest

from voico.core.errors import AudioLoadError, AudioSaveError
from voico.utils.audio_io import (
    get_audio_info,
    load_audio,
    normalize_audio,
    save_audio,
)
from voico.utils.decorators import timer
from voico.utils.math_utils import safe_divide


class TestSafeDivide:
    def test_normal_division(self) -> None:
        a = np.array([10.0, 20.0])
        b = np.array([2.0, 5.0])
        result = safe_divide(a, b)
        np.testing.assert_array_almost_equal(result, [5.0, 4.0])

    def test_division_by_zero(self) -> None:
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        result = safe_divide(a, b)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_custom_fill_value(self) -> None:
        a = np.array([1.0])
        b = np.array([0.0])
        result = safe_divide(a, b, fill=-1.0)
        assert result[0] == -1.0

    def test_mixed_zeros(self) -> None:
        a = np.array([10.0, 5.0, 0.0])
        b = np.array([2.0, 0.0, 0.0])
        result = safe_divide(a, b)
        assert result[0] == 5.0
        assert result[1] == 0.0
        assert result[2] == 0.0


class TestTimer:
    def test_timer_executes_block(self) -> None:
        executed = False
        with timer("test"):
            executed = True
        assert executed

    def test_timer_records_to_dict(self) -> None:
        timing = {}
        with timer("test_op", timing):
            _ = sum(range(100))
        assert "test_op" in timing
        assert timing["test_op"] >= 0

    def test_timer_without_dict(self) -> None:
        with timer("test"):
            pass


class TestAudioIO:
    def test_normalize_audio_peak(self) -> None:
        audio = np.array([0.5, -0.5, 0.25], dtype=np.float32)
        normalized = normalize_audio(audio, target_peak=0.95)
        assert np.max(np.abs(normalized)) == pytest.approx(0.95, abs=1e-5)

    def test_normalize_silence(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        result = normalize_audio(audio)
        np.testing.assert_array_equal(result, audio)

    def test_normalize_preserves_shape(self) -> None:
        audio = np.random.randn(1000).astype(np.float32)
        result = normalize_audio(audio)
        assert result.shape == audio.shape

    def test_save_and_load_roundtrip(self) -> None:
        audio = np.sin(
            2 * np.pi * 440 * np.linspace(0, 0.1, 4410)
        ).astype(np.float32)
        audio = normalize_audio(audio, target_peak=0.9)

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as f:
            temp_path = f.name

        try:
            save_audio(temp_path, audio, 44100)
            loaded, sr = load_audio(temp_path)
            assert sr == 44100
            assert len(loaded) > 0
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(AudioLoadError):
            load_audio("/nonexistent/file.wav")

    def test_save_to_invalid_path_raises(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        with pytest.raises(AudioSaveError):
            save_audio("/nonexistent/dir/out.wav", audio, 44100)

    def test_save_clips_audio(self) -> None:
        audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as f:
            temp_path = f.name
        try:
            save_audio(temp_path, audio, 44100)
            loaded, _ = load_audio(temp_path)
            assert np.max(np.abs(loaded)) <= 1.0 + 1e-3
        finally:
            os.unlink(temp_path)

    def test_save_float32_bit_depth(self) -> None:
        audio = np.sin(
            2 * np.pi * 440 * np.linspace(0, 0.1, 4410)
        ).astype(np.float32)

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as f:
            temp_path = f.name

        try:
            save_audio(temp_path, audio, 44100, bit_depth=32)
            loaded, sr = load_audio(temp_path)
            assert sr == 44100
            assert len(loaded) > 0
        finally:
            os.unlink(temp_path)

    def test_save_unsupported_bit_depth_raises(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as f:
            temp_path = f.name
        try:
            with pytest.raises(AudioSaveError):
                save_audio(temp_path, audio, 44100, bit_depth=24)
        finally:
            os.unlink(temp_path)

    def test_save_unsupported_format_raises(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        with tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False
        ) as f:
            temp_path = f.name
        try:
            with pytest.raises(AudioSaveError):
                save_audio(temp_path, audio, 44100)
        finally:
            os.unlink(temp_path)


class TestGetAudioInfo:
    def test_get_info_wav(self) -> None:
        audio = np.sin(
            2 * np.pi * 440 * np.linspace(0, 0.5, 22050)
        ).astype(np.float32)

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as f:
            temp_path = f.name

        try:
            save_audio(temp_path, audio, 44100)
            info = get_audio_info(temp_path)
            assert info["path"] == temp_path
            assert info["format"] == "wav"
            assert info["sample_rate"] == 44100
            assert int(str(info["file_size_bytes"])) > 0
        finally:
            os.unlink(temp_path)

    def test_get_info_nonexistent_raises(self) -> None:
        with pytest.raises(AudioLoadError):
            get_audio_info("/nonexistent/file.wav")


class TestAudioIOScipyFallback:
    def test_load_wav_scipy_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import voico.utils.audio_io as aio

        audio = np.sin(
            2 * np.pi * 440 * np.linspace(0, 0.1, 4410)
        ).astype(np.float32)
        audio = normalize_audio(audio, target_peak=0.9)

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as f:
            temp_path = f.name

        try:
            save_audio(temp_path, audio, 44100)
            monkeypatch.setattr(aio, "LIBROSA_AVAILABLE", False)
            loaded, sr = load_audio(temp_path)
            assert sr == 44100
            assert len(loaded) > 0
        finally:
            os.unlink(temp_path)

    def test_load_wav_int32_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scipy.io.wavfile as wavmod

        import voico.utils.audio_io as aio

        audio_int32 = (
            np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))
            * 2147483647
        ).astype(np.int32)

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as f:
            temp_path = f.name
            wavmod.write(temp_path, 44100, audio_int32)

        try:
            monkeypatch.setattr(aio, "LIBROSA_AVAILABLE", False)
            loaded, sr = load_audio(temp_path)
            assert sr == 44100
            assert loaded.dtype == np.float32
            assert np.max(np.abs(loaded)) <= 1.0 + 1e-3
        finally:
            os.unlink(temp_path)

    def test_load_stereo_to_mono_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scipy.io.wavfile as wavmod

        import voico.utils.audio_io as aio

        t = np.linspace(0, 0.1, 4410, endpoint=False)
        left = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)
        right = (np.sin(2 * np.pi * 880 * t) * 16384).astype(np.int16)
        stereo = np.stack([left, right], axis=1)

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as f:
            temp_path = f.name
            wavmod.write(temp_path, 44100, stereo)

        try:
            monkeypatch.setattr(aio, "LIBROSA_AVAILABLE", False)
            loaded, sr = load_audio(temp_path)
            assert sr == 44100
            assert len(loaded.shape) == 1
        finally:
            os.unlink(temp_path)

    def test_load_with_resampling_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import voico.utils.audio_io as aio

        audio = np.sin(
            2 * np.pi * 440 * np.linspace(0, 0.1, 4410)
        ).astype(np.float32)
        audio = normalize_audio(audio, target_peak=0.9)

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as f:
            temp_path = f.name

        try:
            save_audio(temp_path, audio, 44100)
            monkeypatch.setattr(aio, "LIBROSA_AVAILABLE", False)
            loaded, sr = load_audio(temp_path, target_sr=22050)
            assert sr == 22050
        finally:
            os.unlink(temp_path)
