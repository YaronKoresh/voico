import io
import tempfile

import numpy as np
import pytest
from scipy.io.wavfile import write as wav_write

from voico.api.app import FASTAPI_AVAILABLE, create_app
from voico.core.types import (
    FormantTrack,
    PitchContour,
    SpectralFeatures,
    VoiceProfile,
)
from voico.store.profile_store import ProfileStore

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_health_endpoint() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(f"{tmpdir}/profiles.db")
        client = testclient.TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_profile_not_found() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(f"{tmpdir}/profiles.db")
        client = testclient.TestClient(app)
        response = client.get("/profiles/missing")
        assert response.status_code == 404


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_analyze_and_list_profile() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(f"{tmpdir}/profiles.db")
        client = testclient.TestClient(app)

        sample_rate = 44100
        duration = 0.15
        timeline = np.linspace(
            0, duration, int(sample_rate * duration), endpoint=False
        )
        audio = (np.sin(2 * np.pi * 220 * timeline) * 0.5).astype(np.float32)

        payload = io.BytesIO()
        wav_write(payload, sample_rate, audio)
        payload.seek(0)

        response = client.post(
            "/profiles/speaker-a/analyze",
            files={"file": ("sample.wav", payload.getvalue(), "audio/wav")},
            data={"quality": "turbo"},
        )
        assert response.status_code == 200
        assert response.json()["saved"] is True

        profiles_response = client.get("/profiles")
        assert profiles_response.status_code == 200
        names = [entry["name"] for entry in profiles_response.json()]
        assert "speaker-a" in names


def test_store_load_none_defensive_branch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfileStore(f"{tmpdir}/profiles.db")
        assert store.load("missing") is None


def test_profile_store_exists_after_save() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfileStore(f"{tmpdir}/profiles.db")
        pitch = PitchContour(
            f0=np.array([200.0], dtype=np.float32),
            voiced_mask=np.array([True]),
            f0_mean=200.0,
            f0_std=0.0,
            harmonic_to_noise_ratio=20.0,
        )
        formants = FormantTrack(
            frequencies=np.ones((5, 1), dtype=np.float32),
            bandwidths=np.ones((5, 1), dtype=np.float32),
            mean_frequencies=np.ones(5, dtype=np.float32),
            mean_bandwidths=np.ones(5, dtype=np.float32),
        )
        spectral = SpectralFeatures(
            envelope=np.ones((8, 1), dtype=np.float32), spectral_tilt=-0.3
        )
        profile = VoiceProfile(
            pitch=pitch,
            formants=formants,
            spectral=spectral,
            harmonic_ratios=np.array([0.4], dtype=np.float32),
            harmonic_energy=np.array([1.0], dtype=np.float32),
            sample_rate=44100,
        )
        store.save("speaker-b", profile)
        assert store.exists("speaker-b") is True
