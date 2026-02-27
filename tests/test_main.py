import os
import tempfile

import numpy as np
import pytest

from voico.core.config import ConversionQuality
from voico.main import main, parse_args, setup_logging
from voico.utils.audio_io import save_audio


def _create_wav(path: str) -> None:
    sample_rate = 44100
    t = np.linspace(0, 0.3, int(sample_rate * 0.3), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.8).astype(np.float32)
    save_audio(path, audio, sample_rate)


class TestSetupLogging:
    def test_verbose_mode(self) -> None:
        setup_logging(verbose=True)

    def test_normal_mode(self) -> None:
        setup_logging(verbose=False)


class TestParseArgs:
    def test_minimal_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "sys.argv", ["voico", "input.wav"]
        )
        args = parse_args()
        assert args.input_file == "input.wav"
        assert args.pitch == 0.0
        assert args.formant == 1.0
        assert args.quality == "balanced"
        assert args.verbose is False

    def test_all_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "sys.argv",
            [
                "voico", "in.wav",
                "-t", "target.wav",
                "-o", "out.wav",
                "-p", "3.5",
                "-f", "1.2",
                "-q", "master",
                "-v",
            ],
        )
        args = parse_args()
        assert args.input_file == "in.wav"
        assert args.target == "target.wav"
        assert args.output == "out.wav"
        assert args.pitch == 3.5
        assert args.formant == 1.2
        assert args.quality == "master"
        assert args.verbose is True


class TestMain:
    def test_missing_input_exits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sys.argv", ["voico", "/nonexistent_file.wav"]
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_missing_target_exits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            _create_wav(input_path)
            monkeypatch.setattr(
                "sys.argv",
                ["voico", input_path, "-t", "/nonexistent_target.wav"],
            )
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_successful_conversion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            output_path = os.path.join(tmpdir, "output.wav")
            _create_wav(input_path)
            monkeypatch.setattr(
                "sys.argv",
                [
                    "voico", input_path,
                    "-o", output_path,
                    "-p", "1.0",
                    "-q", "turbo",
                ],
            )
            main()
            assert os.path.exists(output_path)

    def test_auto_output_path_shift(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            _create_wav(input_path)
            monkeypatch.setattr(
                "sys.argv",
                ["voico", input_path, "-p", "2.0", "-q", "turbo"],
            )
            main()
            expected = os.path.join(tmpdir, "input_shifted_p2.0_f1.0.wav")
            assert os.path.exists(expected)

    def test_auto_output_path_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "source.wav")
            target_path = os.path.join(tmpdir, "target.wav")
            _create_wav(input_path)
            _create_wav(target_path)
            monkeypatch.setattr(
                "sys.argv",
                [
                    "voico", input_path,
                    "-t", target_path,
                    "-q", "turbo",
                ],
            )
            main()
            expected = os.path.join(tmpdir, "source_to_target.wav")
            assert os.path.exists(expected)
