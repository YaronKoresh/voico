import numpy as np
import pytest

from voico.analysis.formant import FormantAnalyzer
from voico.analysis.matcher import VoiceMatcher
from voico.analysis.pitch import PitchAnalyzer
from voico.analysis.profile import VoiceAnalysisEngine
from voico.analysis.spectral import SpectralAnalyzer
from voico.core.types import (
    FormantTrack,
    PitchContour,
    SpectralFeatures,
    VoiceProfile,
)


class TestPitchAnalyzer:
    def test_detect_sine_wave(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        analyzer = PitchAnalyzer(sample_rate, hop_length=512, n_fft=2048)
        contour = analyzer.detect(sine_wave_440hz)

        assert isinstance(contour, PitchContour)
        assert contour.f0_mean > 0
        assert len(contour.f0) > 0
        assert len(contour.voiced_mask) == len(contour.f0)

    def test_detect_silence(
        self, silence: np.ndarray, sample_rate: int
    ) -> None:
        analyzer = PitchAnalyzer(sample_rate, hop_length=512, n_fft=2048)
        contour = analyzer.detect(silence)
        assert isinstance(contour, PitchContour)

    def test_autocorrelation_fallback(
        self,
        sine_wave_440hz: np.ndarray,
        sample_rate: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import voico.analysis.pitch as pitch_mod

        monkeypatch.setattr(pitch_mod, "LIBROSA_AVAILABLE", False)
        analyzer = PitchAnalyzer(sample_rate, hop_length=512, n_fft=2048)
        contour = analyzer.detect(sine_wave_440hz)
        assert isinstance(contour, PitchContour)
        assert len(contour.f0) > 0
        assert len(contour.voiced_mask) == len(contour.f0)

    def test_detect_different_pitches(self, sample_rate: int) -> None:
        t = np.linspace(0, 1.0, sample_rate, endpoint=False)
        low = np.sin(2 * np.pi * 100 * t).astype(np.float32)
        high = np.sin(2 * np.pi * 400 * t).astype(np.float32)

        analyzer = PitchAnalyzer(sample_rate, hop_length=512, n_fft=2048)
        low_contour = analyzer.detect(low)
        high_contour = analyzer.detect(high)

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

    def test_analyze_with_silence(
        self, silence: np.ndarray, sample_rate: int
    ) -> None:
        analyzer = FormantAnalyzer(sample_rate, hop_length=512)
        f0_contour = np.zeros(86)
        track = analyzer.analyze(silence, f0_contour)
        assert isinstance(track, FormantTrack)
        assert all(f > 0 for f in track.mean_frequencies)

    def test_analyze_with_low_pitch(self, sample_rate: int) -> None:
        t = np.linspace(0, 1.0, sample_rate, endpoint=False)
        low_pitch = np.sin(2 * np.pi * 80 * t).astype(np.float32)
        f0_contour = np.full(86, 80.0)
        analyzer = FormantAnalyzer(sample_rate, hop_length=512)
        track = analyzer.analyze(low_pitch, f0_contour)
        assert isinstance(track, FormantTrack)

    def test_analyze_scipy_fallback(
        self,
        sine_wave_440hz: np.ndarray,
        sample_rate: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import voico.analysis.formant as formant_mod

        monkeypatch.setattr(formant_mod, "LIBROSA_AVAILABLE", False)
        analyzer = FormantAnalyzer(sample_rate, hop_length=512)
        f0_contour = np.full(86, 440.0)
        track = analyzer.analyze(sine_wave_440hz, f0_contour)
        assert isinstance(track, FormantTrack)
        assert track.frequencies.shape[0] == 5


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

    def test_stft_cache_reuse(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        analyzer = SpectralAnalyzer(sample_rate, n_fft=2048, hop_length=512)
        analyzer.analyze(sine_wave_440hz)
        assert analyzer._cached_audio_id == id(sine_wave_440hz)
        f0 = np.full(86, 440.0)
        analyzer.compute_harmonic_stats(sine_wave_440hz, f0)
        assert analyzer._cached_audio_id == id(sine_wave_440hz)


class TestVoiceAnalysisEngine:
    def test_build_profile(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        engine = VoiceAnalysisEngine(
            sample_rate, n_fft=2048, hop_length=512
        )
        profile = engine.build(sine_wave_440hz, "Test")

        assert isinstance(profile, VoiceProfile)
        assert profile.sample_rate == sample_rate
        assert profile.pitch.f0_mean > 0

    def test_sample_rate_property_updates_analyzers(self) -> None:
        engine = VoiceAnalysisEngine(44100, n_fft=2048, hop_length=512)
        assert engine.sample_rate == 44100

        engine.sample_rate = 22050
        assert engine.sample_rate == 22050
        assert engine.pitch_analyzer.sample_rate == 22050
        assert engine.formant_analyzer.sample_rate == 22050
        assert engine.spectral_analyzer.sample_rate == 22050

    def test_same_sample_rate_no_rebuild(self) -> None:
        engine = VoiceAnalysisEngine(44100, n_fft=2048, hop_length=512)
        original_analyzer = engine.pitch_analyzer
        engine.sample_rate = 44100
        assert engine.pitch_analyzer is original_analyzer

    def test_aligned_output_lengths(
        self, sine_wave_440hz: np.ndarray, sample_rate: int
    ) -> None:
        engine = VoiceAnalysisEngine(
            sample_rate, n_fft=2048, hop_length=512
        )
        profile = engine.build(sine_wave_440hz)

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
