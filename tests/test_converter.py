import os
import tempfile

import numpy as np
import pytest

from voico.converter import VoiceConverter
from voico.core.config import ConversionQuality
from voico.core.errors import ConversionError
from voico.utils.audio_io import save_audio


def _create_test_wav(path: str, frequency: float = 440.0) -> None:
    sample_rate = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * frequency * t) * 0.8).astype(np.float32)
    save_audio(path, audio, sample_rate)


class TestVoiceConverter:
    def test_init_default_quality(self) -> None:
        converter = VoiceConverter()
        assert converter.settings is not None
        assert converter.n_fft == 2048

    @pytest.mark.parametrize("quality", list(ConversionQuality))
    def test_init_all_qualities(self, quality: ConversionQuality) -> None:
        converter = VoiceConverter(quality)
        assert converter.settings is not None

    def test_process_pitch_shift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            output_path = os.path.join(tmpdir, "output.wav")
            _create_test_wav(input_path)

            converter = VoiceConverter(ConversionQuality.TURBO)
            converter.process(
                input_path=input_path,
                output_path=output_path,
                pitch_shift=2.0,
            )
            assert os.path.exists(output_path)

    def test_process_formant_shift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            output_path = os.path.join(tmpdir, "output.wav")
            _create_test_wav(input_path)

            converter = VoiceConverter(ConversionQuality.TURBO)
            converter.process(
                input_path=input_path,
                output_path=output_path,
                formant_shift=1.3,
            )
            assert os.path.exists(output_path)

    def test_process_with_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "source.wav")
            target_path = os.path.join(tmpdir, "target.wav")
            output_path = os.path.join(tmpdir, "output.wav")
            _create_test_wav(input_path, frequency=200.0)
            _create_test_wav(target_path, frequency=400.0)

            converter = VoiceConverter(ConversionQuality.TURBO)
            converter.process(
                input_path=input_path,
                output_path=output_path,
                target_path=target_path,
            )
            assert os.path.exists(output_path)

    def test_process_missing_input(self) -> None:
        converter = VoiceConverter()
        with pytest.raises(FileNotFoundError):
            converter.process(
                input_path="/nonexistent.wav",
                output_path="/output.wav",
            )

    def test_process_missing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            _create_test_wav(input_path)

            converter = VoiceConverter()
            with pytest.raises(FileNotFoundError):
                converter.process(
                    input_path=input_path,
                    output_path=os.path.join(tmpdir, "out.wav"),
                    target_path="/nonexistent_target.wav",
                )

    def test_process_no_shift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            output_path = os.path.join(tmpdir, "output.wav")
            _create_test_wav(input_path)

            converter = VoiceConverter(ConversionQuality.TURBO)
            converter.process(
                input_path=input_path,
                output_path=output_path,
                pitch_shift=0.0,
                formant_shift=1.0,
            )
            assert os.path.exists(output_path)
