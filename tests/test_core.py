import numpy as np
import pytest

from voico.core.config import ConversionQuality, QualitySettings
from voico.core.constants import AudioConstants
from voico.core.errors import (
    AnalysisError,
    AudioLoadError,
    AudioSaveError,
    ConversionError,
    VoicoError,
)
from voico.core.types import (
    FormantTrack,
    PitchContour,
    SpectralFeatures,
    VoiceProfile,
)


class TestConversionQuality:
    def test_all_presets_exist(self) -> None:
        expected = {"turbo", "fast", "balanced", "high", "ultra", "master"}
        assert {q.value for q in ConversionQuality} == expected

    def test_default_is_balanced(self) -> None:
        assert ConversionQuality.BALANCED.value == "balanced"


class TestQualitySettings:
    @pytest.mark.parametrize("quality", list(ConversionQuality))
    def test_from_preset_returns_settings(
        self, quality: ConversionQuality
    ) -> None:
        settings = QualitySettings.from_preset(quality)
        assert isinstance(settings, QualitySettings)
        assert settings.hop_divisor > 0
        assert settings.griffin_lim_iters > 0

    def test_turbo_is_fastest(self) -> None:
        turbo = QualitySettings.from_preset(ConversionQuality.TURBO)
        master = QualitySettings.from_preset(ConversionQuality.MASTER)
        assert turbo.griffin_lim_iters < master.griffin_lim_iters

    def test_master_has_advanced_phase(self) -> None:
        settings = QualitySettings.from_preset(ConversionQuality.MASTER)
        assert settings.use_advanced_phase is True

    def test_turbo_no_advanced_phase(self) -> None:
        settings = QualitySettings.from_preset(ConversionQuality.TURBO)
        assert settings.use_advanced_phase is False


class TestAudioConstants:
    def test_frequency_range(self) -> None:
        assert AudioConstants.MIN_F0_HZ < AudioConstants.MAX_F0_HZ
        assert AudioConstants.MIN_F0_HZ == 50.0
        assert AudioConstants.MAX_F0_HZ == 600.0

    def test_fft_size_is_power_of_two(self) -> None:
        n = AudioConstants.DEFAULT_N_FFT
        assert n > 0 and (n & (n - 1)) == 0

    def test_formant_defaults_length(self) -> None:
        assert len(AudioConstants.DEFAULT_FORMANT_FREQS) == 5
        assert len(AudioConstants.DEFAULT_FORMANT_BANDWIDTHS) == 5

    def test_epsilon_is_small_positive(self) -> None:
        assert 0 < AudioConstants.EPSILON < 1e-5


class TestDataclasses:
    def test_pitch_contour_creation(self) -> None:
        contour = PitchContour(
            f0=np.array([100.0, 110.0]),
            voiced_mask=np.array([True, True]),
            f0_mean=105.0,
            f0_std=5.0,
            harmonic_to_noise_ratio=20.0,
        )
        assert contour.f0_mean == 105.0
        assert contour.harmonic_to_noise_ratio == 20.0

    def test_formant_track_creation(self) -> None:
        track = FormantTrack(
            frequencies=np.zeros((5, 10)),
            bandwidths=np.zeros((5, 10)),
            mean_frequencies=np.array([500, 1500, 2500, 3500, 4500]),
            mean_bandwidths=np.array([80, 100, 120, 150, 200]),
        )
        assert track.frequencies.shape == (5, 10)

    def test_spectral_features_creation(self) -> None:
        features = SpectralFeatures(
            envelope=np.ones((100, 10)),
            spectral_tilt=-1.5,
        )
        assert features.spectral_tilt == -1.5

    def test_voice_profile_creation(self) -> None:
        pitch = PitchContour(
            np.array([100.0]),
            np.array([True]),
            100.0,
            0.0,
            20.0,
        )
        formants = FormantTrack(
            np.zeros((5, 1)),
            np.zeros((5, 1)),
            np.zeros(5),
            np.zeros(5),
        )
        spectral = SpectralFeatures(np.ones((100, 1)), -1.0)
        profile = VoiceProfile(
            pitch=pitch,
            formants=formants,
            spectral=spectral,
            harmonic_ratios=np.array([0.5]),
            harmonic_energy=np.array([1.0]),
            sample_rate=44100,
        )
        assert profile.sample_rate == 44100


class TestErrors:
    def test_hierarchy(self) -> None:
        assert issubclass(AudioLoadError, VoicoError)
        assert issubclass(AudioSaveError, VoicoError)
        assert issubclass(AnalysisError, VoicoError)
        assert issubclass(ConversionError, VoicoError)
        assert issubclass(VoicoError, Exception)

    def test_raise_and_catch(self) -> None:
        with pytest.raises(VoicoError):
            raise AudioLoadError("test")

        with pytest.raises(VoicoError):
            raise ConversionError("test")
