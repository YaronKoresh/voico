import numpy as np
import pytest

from voico.dsp.phase import PhaseProcessor
from voico.dsp.shifter import SpectralProcessor


class TestPhaseProcessor:
    def test_reconstruct_produces_output(self) -> None:
        processor = PhaseProcessor(n_fft=256, hop_length=64)
        magnitude = np.random.rand(129, 20).astype(np.float32)
        audio = processor.reconstruct(magnitude, n_iter=8)
        assert len(audio) > 0

    def test_reconstruct_native_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import voico.dsp.phase as phase_mod

        monkeypatch.setattr(phase_mod, "LIBROSA_AVAILABLE", False)
        processor = PhaseProcessor(n_fft=256, hop_length=64)
        magnitude = np.random.rand(129, 20).astype(np.float32)
        audio = processor.reconstruct(magnitude, n_iter=4)
        assert len(audio) > 0

    def test_reconstruct_with_initial_phase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import voico.dsp.phase as phase_mod

        monkeypatch.setattr(phase_mod, "LIBROSA_AVAILABLE", False)
        processor = PhaseProcessor(n_fft=256, hop_length=64)
        magnitude = np.random.rand(129, 20).astype(np.float32)
        phase = magnitude * np.exp(1j * np.random.rand(129, 20) * 2 * np.pi)
        audio = processor.reconstruct(
            magnitude, n_iter=4, initial_phase=phase
        )
        assert len(audio) > 0

    def test_reconstruct_roundtrip_quality(self) -> None:
        processor = PhaseProcessor(n_fft=256, hop_length=64)
        magnitude = np.random.rand(129, 20).astype(np.float32)
        audio = processor.reconstruct(magnitude, n_iter=16)
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))

    def test_reconstruct_rtpghi(self) -> None:
        processor = PhaseProcessor(n_fft=256, hop_length=64)
        magnitude = np.random.rand(129, 20).astype(np.float32)
        audio = processor.reconstruct_rtpghi(magnitude)
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))


class TestSpectralProcessor:
    def test_shift_pitch_zero(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        processor = SpectralProcessor(sample_rate, n_fft=2048)
        result = processor.shift_pitch(sine_wave_440hz, 0.0)
        np.testing.assert_array_equal(result, sine_wave_440hz)

    def test_shift_pitch_up(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        processor = SpectralProcessor(sample_rate, n_fft=2048)
        result = processor.shift_pitch(sine_wave_440hz, 2.0)
        assert len(result) > 0

    def test_shift_pitch_down(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        processor = SpectralProcessor(sample_rate, n_fft=2048)
        result = processor.shift_pitch(sine_wave_440hz, -2.0)
        assert len(result) > 0

    def test_shift_formants_identity(self) -> None:
        processor = SpectralProcessor(44100, n_fft=2048)
        magnitude = np.random.rand(1025, 50).astype(np.float32)
        result = processor.shift_formants(magnitude, 1.0)
        np.testing.assert_array_equal(result, magnitude)

    def test_shift_formants_up(self) -> None:
        processor = SpectralProcessor(44100, n_fft=2048)
        magnitude = np.random.rand(1025, 50).astype(np.float32)
        result = processor.shift_formants(magnitude, 1.5)
        assert result.shape == magnitude.shape

    def test_shift_formants_down(self) -> None:
        processor = SpectralProcessor(44100, n_fft=2048)
        magnitude = np.random.rand(1025, 50).astype(np.float32)
        result = processor.shift_formants(magnitude, 0.7)
        assert result.shape == magnitude.shape

    def test_match_spectral_tilt(self) -> None:
        processor = SpectralProcessor(44100, n_fft=2048)
        magnitude = np.random.rand(1025, 50).astype(np.float32) + 0.01
        result = processor.match_spectral_tilt(magnitude, -2.0)
        assert result.shape == magnitude.shape

    def test_match_spectral_tilt_narrow_range(self) -> None:
        processor = SpectralProcessor(44100, n_fft=32)
        magnitude = np.random.rand(17, 10).astype(np.float32)
        result = processor.match_spectral_tilt(magnitude, -1.0)
        np.testing.assert_array_equal(result, magnitude)

    def test_frequency_bins_initialized(self) -> None:
        processor = SpectralProcessor(44100, n_fft=2048)
        assert len(processor.frequency_bins) == 1025
        assert processor.frequency_bins[0] == 0.0
