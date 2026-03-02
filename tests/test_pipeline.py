import numpy as np
import pytest

from voico.pipeline import (
    PipelineContext,
    LoadStage,
    MetricsStage,
)


class DummyLogger:
    def __init__(self):
        self.events = []

    def log_event(self, *args, **kwargs):
        self.events.append((args, kwargs))


def test_metrics_computes_snr_and_centroid() -> None:
    ctx = PipelineContext(
        input_path="",
        output_path="",
        pitch_shift=0.0,
        formant_shift=1.0,
        target_path=None,
        bit_depth=16,
        on_progress=None,
        n_fft=8,
        hop_length=4,
        settings=None,
    )
    # create simple audio
    ctx.audio = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)
    ctx.output_audio = ctx.audio.copy()
    metrics = MetricsStage()
    ctx = metrics.execute(ctx)
    assert ctx.snr_db > 50.0  # identical audio produces very high SNR
    assert ctx.spectral_centroid_deviation == pytest.approx(0.0)


def test_load_stage_nonexistent(monkeypatch) -> None:
    ctx = PipelineContext(
        input_path="/no/such/file.wav",
        output_path="",
        pitch_shift=0,
        formant_shift=1.0,
        target_path=None,
        bit_depth=16,
        on_progress=None,
        n_fft=8,
        hop_length=4,
        settings=None,
        diagnostic_logger=DummyLogger(),
    )
    loader = LoadStage()
    with pytest.raises(Exception):
        loader.execute(ctx)
