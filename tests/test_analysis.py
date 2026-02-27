import numpy as np
import pytest

from voico.analysis.formant import FormantAnalyzer
from voico.analysis.matcher import VoiceMatcher
from voico.analysis.pitch import PitchDetector
from voico.analysis.profile import VoiceProfileBuilder
from voico.analysis.spectral import SpectralAnalyzer
from voico.core.types import (
    FormantTrack,
    PitchContour,
    SpectralFeatures,
    VoiceProfile,
)


class TestPitchDetector:
    def test_detect_sine_wave(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        detector = PitchDetector(sample_rate, hop_length=512, n_fft=2048)
        contour = detector.detect(sine_wave_440hz)

        assert isinstance(contour, PitchContour)
        assert contour.f0_mean > 0
        assert len(contour.f0) > 0
        assert len(contour.voiced_mask) == len(contour.f0)

    def test_detect_silence(
        self, silence: np.ndarray, sample_rate: int
    ) -> None:
        detector = PitchDetector(sample_rate, hop_length=512, n_fft=2048)
        contour = detector.detect(silence)
        assert isinstance(contour, PitchContour)

    def test_autocorrelation_fallback(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        detector = PitchDetector(sample_rate, hop_length=512, n_fft=2048)
        f0, confidence = detector._autocorrelation_detect(sine_wave_440hz)
        assert len(f0) > 0
        assert len(confidence) == len(f0)

    def test_detect_different_pitches(self, sample_rate: int) -> None:
        t = np.linspace(0, 1.0, sample_rate, endpoint=False)
        low = np.sin(2 * np.pi * 100 * t).astype(np.float32)
        high = np.sin(2 * np.pi * 400 * t).astype(np.float32)

        detector = PitchDetector(sample_rate, hop_length=512, n_fft=2048)
        low_contour = detector.detect(low)
        high_contour = detector.detect(high)

        assert low_contour.f0_mean < high_contour.f0_mean


class TestFormantAnalyzer:
    def test_analyze_returns_formant_track(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        analyzer = FormantAnalyzer(sample_rate, hop_length=512)
        f0_contour = np.full(86, 440.0)
        track = analyzer.analyze(sine_wave_440hz, f0_contour)

        assert isinstance(track, FormantTrack)
        assert track.frequencies.shape[0] == 5
        assert len(track.mean_frequencies) == 5
        assert len(track.mean_bandwidths) == 5

    def test_levinson_durbin_with_signal(self, sample_rate: int) -> None:
        t = np.linspace(0, 0.025, int(sample_rate * 0.025), endpoint=False)
        frame = np.sin(2 * np.pi * 200 * t).astype(np.float64)

        analyzer = FormantAnalyzer(sample_rate, hop_length=512)
        coeffs = analyzer._levinson_durbin(frame, order=14)

        assert coeffs is not None
        assert len(coeffs) == 15
        assert coeffs[0] == 1.0

    def test_levinson_durbin_with_silence(self) -> None:
        analyzer = FormantAnalyzer(44100, hop_length=512)
        frame = np.zeros(256, dtype=np.float64)
        result = analyzer._levinson_durbin(frame, order=14)
        assert result is None

    def test_lpc_to_formants(self) -> None:
        analyzer = FormantAnalyzer(10000, hop_length=512)
        coeffs = np.array([1.0, -1.5, 0.7, -0.2, 0.05])
        freqs, bws = analyzer._lpc_to_formants(coeffs, 10000)
        assert isinstance(freqs, np.ndarray)
        assert isinstance(bws, np.ndarray)


class TestSpectralAnalyzer:
    def test_analyze_returns_features(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        analyzer = SpectralAnalyzer(sample_rate, n_fft=2048, hop_length=512)
        features = analyzer.analyze(sine_wave_440hz)

        assert isinstance(features, SpectralFeatures)
        assert features.envelope.shape[0] > 0
        assert isinstance(features.spectral_tilt, float)

    def test_harmonic_stats(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        analyzer = SpectralAnalyzer(sample_rate, n_fft=2048, hop_length=512)
        f0 = np.full(86, 440.0)
        energy, ratios = analyzer.compute_harmonic_stats(
            sine_wave_440hz, f0
        )

        assert len(energy) > 0
        assert len(ratios) == len(energy)
        assert np.all(ratios >= 0)
        assert np.all(ratios <= 1.0 + 1e-6)

    def test_analyze_noise(
        self, white_noise: np.ndarray, sample_rate: int
    ) -> None:
        analyzer = SpectralAnalyzer(sample_rate, n_fft=2048, hop_length=512)
        features = analyzer.analyze(white_noise)
        assert isinstance(features, SpectralFeatures)


class TestVoiceProfileBuilder:
    def test_build_profile(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        builder = VoiceProfileBuilder(
            sample_rate, n_fft=2048, hop_length=512
        )
        profile = builder.build(sine_wave_440hz, "Test")

        assert isinstance(profile, VoiceProfile)
        assert profile.sample_rate == sample_rate
        assert profile.pitch.f0_mean > 0

    def test_sample_rate_property_updates_analyzers(self) -> None:
        builder = VoiceProfileBuilder(44100, n_fft=2048, hop_length=512)
        assert builder.sample_rate == 44100

        builder.sample_rate = 22050
        assert builder.sample_rate == 22050
        assert builder.pitch_detector.sample_rate == 22050
        assert builder.formant_analyzer.sample_rate == 22050
        assert builder.spectral_analyzer.sample_rate == 22050

    def test_same_sample_rate_no_rebuild(self) -> None:
        builder = VoiceProfileBuilder(44100, n_fft=2048, hop_length=512)
        original_detector = builder.pitch_detector
        builder.sample_rate = 44100
        assert builder.pitch_detector is original_detector

    def test_aligned_output_lengths(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        builder = VoiceProfileBuilder(
            sample_rate, n_fft=2048, hop_length=512
        )
        profile = builder.build(sine_wave_440hz)

        pitch_len = len(profile.pitch.f0)
        formant_len = profile.formants.frequencies.shape[1]
        spectral_len = profile.spectral.envelope.shape[1]
        energy_len = len(profile.harmonic_energy)

        assert pitch_len == formant_len
        assert formant_len == spectral_len
        assert spectral_len == energy_len


class TestVoiceMatcher:
    def _make_profile(
        self, f0_mean: float, formant_freqs: np.ndarray
    ) -> VoiceProfile:
        pitch = PitchContour(
            f0=np.array([f0_mean]),
            voiced_mask=np.array([True]),
            f0_mean=f0_mean,
            f0_std=5.0,
            harmonic_to_noise_ratio=20.0,
        )
        formants = FormantTrack(
            frequencies=np.zeros((5, 1)),
            bandwidths=np.zeros((5, 1)),
            mean_frequencies=formant_freqs,
            mean_bandwidths=np.array([80, 100, 120, 150, 200]),
        )
        spectral = SpectralFeatures(np.ones((100, 1)), -1.0)
        return VoiceProfile(
            pitch=pitch,
            formants=formants,
            spectral=spectral,
            harmonic_ratios=np.array([0.5]),
            harmonic_energy=np.array([1.0]),
            sample_rate=44100,
        )

    def test_match_same_voice(self) -> None:
        freqs = np.array([500.0, 1500.0, 2500.0, 3500.0, 4500.0])
        source = self._make_profile(200.0, freqs)
        target = self._make_profile(200.0, freqs)

        semitones, factor = VoiceMatcher.match(source, target)
        assert abs(semitones) < 0.01
        assert abs(factor - 1.0) < 0.01

    def test_match_octave_up(self) -> None:
        freqs = np.array([500.0, 1500.0, 2500.0, 3500.0, 4500.0])
        source = self._make_profile(100.0, freqs)
        target = self._make_profile(200.0, freqs)

        semitones, _ = VoiceMatcher.match(source, target)
        assert abs(semitones - 12.0) < 0.01

    def test_match_formant_shift(self) -> None:
        src_freqs = np.array([500.0, 1500.0, 2500.0, 3500.0, 4500.0])
        tgt_freqs = np.array([600.0, 1800.0, 3000.0, 3500.0, 4500.0])
        source = self._make_profile(200.0, src_freqs)
        target = self._make_profile(200.0, tgt_freqs)

        _, factor = VoiceMatcher.match(source, target)
        assert factor > 1.0

    def test_match_clamps_formant_factor(self) -> None:
        src_freqs = np.array([500.0, 1500.0, 2500.0, 3500.0, 4500.0])
        tgt_freqs = np.array([5000.0, 15000.0, 25000.0, 3500.0, 4500.0])
        source = self._make_profile(200.0, src_freqs)
        target = self._make_profile(200.0, tgt_freqs)

        _, factor = VoiceMatcher.match(source, target)
        assert factor <= 2.0

    def test_match_zero_pitch_defaults(self) -> None:
        freqs = np.array([500.0, 1500.0, 2500.0, 3500.0, 4500.0])
        source = self._make_profile(0.0, freqs)
        target = self._make_profile(200.0, freqs)

        semitones, _ = VoiceMatcher.match(source, target)
        assert semitones == 0.0
