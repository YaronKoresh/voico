import tempfile

import numpy as np

from voico.core.types import (
    FormantTrack,
    PitchContour,
    SpectralFeatures,
    VoiceProfile,
)
from voico.store.profile_store import ProfileStore


def _sample_profile() -> VoiceProfile:
    pitch = PitchContour(
        f0=np.array([100.0, 105.0], dtype=np.float32),
        voiced_mask=np.array([True, True]),
        f0_mean=102.5,
        f0_std=2.5,
        harmonic_to_noise_ratio=16.0,
    )
    formants = FormantTrack(
        frequencies=np.full((5, 2), 700.0, dtype=np.float32),
        bandwidths=np.full((5, 2), 120.0, dtype=np.float32),
        mean_frequencies=np.array([700.0, 1400.0, 2200.0, 3200.0, 4200.0]),
        mean_bandwidths=np.array([90.0, 100.0, 110.0, 130.0, 150.0]),
    )
    spectral = SpectralFeatures(
        envelope=np.ones((64, 2), dtype=np.float32),
        spectral_tilt=-1.0,
    )
    return VoiceProfile(
        pitch=pitch,
        formants=formants,
        spectral=spectral,
        harmonic_ratios=np.array([0.5, 0.6], dtype=np.float32),
        harmonic_energy=np.array([1.0, 2.0], dtype=np.float32),
        sample_rate=44100,
    )


def test_profile_store_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/profiles.db"
        store = ProfileStore(db_path)
        profile = _sample_profile()
        store.save("speaker", profile)

        loaded = store.load("speaker")
        assert loaded is not None
        assert loaded.sample_rate == profile.sample_rate
        assert loaded.pitch.f0_mean == profile.pitch.f0_mean
        assert store.exists("speaker") is True


def test_profile_store_delete_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/profiles.db"
        store = ProfileStore(db_path)
        assert store.delete("missing") is False
        assert store.load("missing") is None
