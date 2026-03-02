import numpy as np

from voico.core.types import (
    FormantTrack,
    PitchContour,
    SpectralFeatures,
    VoiceProfile,
)
from voico.quality.gates import PitchValidationGate, ProfileValidationGate


def _make_profile(harmonic_energy: np.ndarray) -> VoiceProfile:
    pitch = PitchContour(
        f0=np.array([120.0, 130.0], dtype=np.float32),
        voiced_mask=np.array([True, True]),
        f0_mean=125.0,
        f0_std=5.0,
        harmonic_to_noise_ratio=18.0,
    )
    formants = FormantTrack(
        frequencies=np.array(
            [
                [500.0, 510.0],
                [1500.0, 1510.0],
                [2500.0, 2510.0],
                [3500.0, 3510.0],
                [4500.0, 4510.0],
            ],
            dtype=np.float32,
        ),
        bandwidths=np.full((5, 2), 120.0, dtype=np.float32),
        mean_frequencies=np.array([505.0, 1505.0, 2505.0, 3505.0, 4505.0]),
        mean_bandwidths=np.array([120.0, 120.0, 120.0, 120.0, 120.0]),
    )
    spectral = SpectralFeatures(
        envelope=np.ones((32, 2), dtype=np.float32),
        spectral_tilt=-0.5,
    )
    return VoiceProfile(
        pitch=pitch,
        formants=formants,
        spectral=spectral,
        harmonic_ratios=np.array([0.7, 0.8], dtype=np.float32),
        harmonic_energy=harmonic_energy,
        sample_rate=44100,
    )


def test_pitch_gate_handles_empty_contour() -> None:
    contour = PitchContour(
        f0=np.array([], dtype=np.float32),
        voiced_mask=np.array([], dtype=bool),
        f0_mean=0.0,
        f0_std=0.0,
        harmonic_to_noise_ratio=0.0,
    )
    result = PitchValidationGate(contour).validate()
    assert result.passed is False
    assert result.score == 0.0
    assert len(result.issues) > 0


def test_profile_gate_handles_empty_harmonic_energy() -> None:
    profile = _make_profile(np.array([], dtype=np.float32))
    result = ProfileValidationGate(profile).validate()
    assert result.passed is False
    assert result.score < 100.0
    assert any(
        "Harmonic energy track is empty" in issue for issue in result.issues
    )
