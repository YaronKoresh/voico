import os
import tempfile

import numpy as np
import pytest

from voico.converter import VoiceConverter
from voico.core.config import ConversionQuality
from voico.core.errors import ConversionError
from voico.core.types import ConversionReport
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

    def test_process_with_progress_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            output_path = os.path.join(tmpdir, "output.wav")
            _create_test_wav(input_path)

            steps = []
            converter = VoiceConverter(ConversionQuality.TURBO)
            converter.process(
                input_path=input_path,
                output_path=output_path,
                pitch_shift=2.0,
                on_progress=lambda step, frac: steps.append(
                    (step, frac)
                ),
            )
            assert os.path.exists(output_path)
            assert len(steps) > 0
            assert steps[-1] == ("Done", 1.0)

    def test_process_float32_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            output_path = os.path.join(tmpdir, "output.wav")
            _create_test_wav(input_path)

            converter = VoiceConverter(ConversionQuality.TURBO)
            converter.process(
                input_path=input_path,
                output_path=output_path,
                pitch_shift=1.0,
                bit_depth=32,
            )
            assert os.path.exists(output_path)

    def test_process_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pairs = []
            for i in range(3):
                inp = os.path.join(tmpdir, f"input_{i}.wav")
                out = os.path.join(tmpdir, f"output_{i}.wav")
                _create_test_wav(inp)
                pairs.append((inp, out))

            progress_calls = []
            converter = VoiceConverter(ConversionQuality.TURBO)
            results = converter.process_batch(
                pairs,
                pitch_shift=1.0,
                on_file_progress=lambda i, t, p: progress_calls.append(
                    (i, t, p)
                ),
            )

            assert len(results) == 3
            for out_path in results:
                assert os.path.exists(out_path)
            assert len(progress_calls) == 4
            assert progress_calls[-1][0] == 3


class TestConversionReport:
    def test_process_returns_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            output_path = os.path.join(tmpdir, "output.wav")
            _create_test_wav(input_path)
            converter = VoiceConverter(ConversionQuality.TURBO)
            report = converter.process(
                input_path=input_path,
                output_path=output_path,
                pitch_shift=1.0,
            )
            assert isinstance(report, ConversionReport)
            assert report.output_path == output_path
            assert report.sample_rate == 44100
            assert report.input_duration_seconds > 0
            assert isinstance(report.stages_timing, dict)
            assert "load" in report.stages_timing


class TestAsyncAPI:
    def test_aprocess(self) -> None:
        import asyncio
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            output_path = os.path.join(tmpdir, "output.wav")
            _create_test_wav(input_path)

            converter = VoiceConverter(ConversionQuality.TURBO)

            async def run():
                return await converter.aprocess(
                    input_path=input_path,
                    output_path=output_path,
                    pitch_shift=1.0,
                )

            report = asyncio.run(run())
            assert os.path.exists(output_path)
            assert isinstance(report, ConversionReport)

    def test_aprocess_batch(self) -> None:
        import asyncio
        with tempfile.TemporaryDirectory() as tmpdir:
            pairs = [(
                os.path.join(tmpdir, "in.wav"),
                os.path.join(tmpdir, "out.wav"),
            )]
            _create_test_wav(pairs[0][0])
            converter = VoiceConverter(ConversionQuality.TURBO)

            async def run():
                return await converter.aprocess_batch(pairs, pitch_shift=1.0)

            results = asyncio.run(run())
            assert len(results) == 1
            assert os.path.exists(results[0])


class TestStreaming:
    def test_stream_yields_output(self) -> None:
        sample_rate = 44100
        t = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

        chunk_size = 512
        chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

        converter = VoiceConverter(ConversionQuality.FAST)
        outputs = list(converter.stream(iter(chunks), pitch_shift=2.0))
        assert len(outputs) > 0

    def test_voice_stream_processor_direct(self) -> None:
        from voico.stream.streamer import VoiceStreamProcessor
        processor = VoiceStreamProcessor(
            sample_rate=44100,
            pitch_shift=2.0,
            formant_shift=1.0,
            quality=ConversionQuality.FAST,
        )
        chunk = np.random.randn(512).astype(np.float32)
        result = processor.process_chunk(chunk)
        assert len(result) > 0
