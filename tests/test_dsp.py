import numpy as np
import pytest

from voico.dsp.phase import PhaseReconstructor
from voico.dsp.shifter import SpectralShifter


class TestPhaseReconstructor:
    def test_reconstruct_produces_output(self) -> None:
        reconstructor = PhaseReconstructor(n_fft=256, hop_length=64)
        magnitude = np.random.rand(129, 20).astype(np.float32)
        audio = reconstructor.reconstruct(magnitude, n_iter=8)
        assert len(audio) > 0

    def test_griffin_lim_native(self) -> None:
        reconstructor = PhaseReconstructor(n_fft=256, hop_length=64)
        magnitude = np.random.rand(129, 20).astype(np.float32)
        audio = reconstructor._griffin_lim_native(magnitude, n_iter=4, initial_phase=None)
        assert len(audio) > 0

    def test_griffin_lim_with_initial_phase(self) -> None:
        reconstructor = PhaseReconstructor(n_fft=256, hop_length=64)
        magnitude = np.random.rand(129, 20).astype(np.float32)
        phase = magnitude * np.exp(1j * np.random.rand(129, 20) * 2 * np.pi)
        audio = reconstructor._griffin_lim_native(
            magnitude, n_iter=4, initial_phase=phase
        )
        assert len(audio) > 0

    def test_forward_and_inverse_stft_roundtrip(self) -> None:
        reconstructor = PhaseReconstructor(n_fft=256, hop_length=64)
        audio = np.random.randn(1024).astype(np.float32)
        stft = reconstructor._forward_stft(audio)
        reconstructed = reconstructor._inverse_stft(stft)
        assert len(reconstructed) > 0


class TestSpectralShifter:
    def test_shift_pitch_zero(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        shifter = SpectralShifter(sample_rate, n_fft=2048)
        result = shifter.shift_pitch(sine_wave_440hz, 0.0)
        np.testing.assert_array_equal(result, sine_wave_440hz)

    def test_shift_pitch_up(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        shifter = SpectralShifter(sample_rate, n_fft=2048)
        result = shifter.shift_pitch(sine_wave_440hz, 2.0)
        assert len(result) > 0

    def test_shift_pitch_down(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        shifter = SpectralShifter(sample_rate, n_fft=2048)
        result = shifter.shift_pitch(sine_wave_440hz, -2.0)
        assert len(result) > 0

    def test_shift_formants_identity(self) -> None:
        shifter = SpectralShifter(44100, n_fft=2048)
        magnitude = np.random.rand(1025, 50).astype(np.float32)
        result = shifter.shift_formants(magnitude, 1.0)
        np.testing.assert_array_equal(result, magnitude)

    def test_shift_formants_up(self) -> None:
        shifter = SpectralShifter(44100, n_fft=2048)
        magnitude = np.random.rand(1025, 50).astype(np.float32)
        result = shifter.shift_formants(magnitude, 1.5)
        assert result.shape == magnitude.shape

    def test_shift_formants_down(self) -> None:
        shifter = SpectralShifter(44100, n_fft=2048)
        magnitude = np.random.rand(1025, 50).astype(np.float32)
        result = shifter.shift_formants(magnitude, 0.7)
        assert result.shape == magnitude.shape

    def test_match_spectral_tilt(self) -> None:
        shifter = SpectralShifter(44100, n_fft=2048)
        magnitude = np.random.rand(1025, 50).astype(np.float32) + 0.01
        result = shifter.match_spectral_tilt(magnitude, -2.0)
        assert result.shape == magnitude.shape

    def test_match_spectral_tilt_narrow_range(self) -> None:
        shifter = SpectralShifter(44100, n_fft=32)
        magnitude = np.random.rand(17, 10).astype(np.float32)
        result = shifter.match_spectral_tilt(magnitude, -1.0)
        np.testing.assert_array_equal(result, magnitude)

    def test_frequency_bins_initialized(self) -> None:
        shifter = SpectralShifter(44100, n_fft=2048)
        assert len(shifter.frequency_bins) == 1025
        assert shifter.frequency_bins[0] == 0.0
