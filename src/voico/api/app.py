import os
import tempfile
from typing import Optional

from ..analysis.profile import VoiceProfileBuilder
from ..converter import VoiceConverter
from ..core.config import ConversionQuality
from ..store.profile_store import ProfileStore

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def create_app(db_path: Optional[str] = None) -> "FastAPI":
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "fastapi is required for the API server. "
            "Install with: pip install voico[server]"
        )

    app = FastAPI(title="Voico API", version="0.1.0")
    store = ProfileStore(db_path) if db_path else ProfileStore()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/profiles")
    def list_profiles() -> list:
        return store.list_profiles()

    @app.get("/profiles/{name}")
    def get_profile(name: str) -> dict:
        if not store.exists(name):
            raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
        profile = store.load(name)
        return {
            "name": name,
            "sample_rate": profile.sample_rate,
            "f0_mean": profile.pitch.f0_mean,
            "f0_std": profile.pitch.f0_std,
            "hnr": profile.pitch.harmonic_to_noise_ratio,
            "mean_formants": profile.formants.mean_frequencies.tolist(),
            "spectral_tilt": profile.spectral.spectral_tilt,
        }

    @app.post("/profiles/{name}/analyze")
    async def analyze_and_save(
        name: str,
        file: UploadFile = File(...),
        quality: str = Form(default="balanced"),
    ) -> dict:
        try:
            q = ConversionQuality(quality)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid quality: {quality}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            from ..utils.audio_io import load_audio
            audio, sr = load_audio(tmp_path)
            from ..core.config import QualitySettings
            settings = QualitySettings.from_preset(q)
            from ..core.constants import AudioConstants
            n_fft = AudioConstants.DEFAULT_N_FFT
            hop_length = n_fft // settings.hop_divisor
            builder = VoiceProfileBuilder(sr, n_fft=n_fft, hop_length=hop_length)
            profile = builder.build(audio, name)
            store.save(name, profile)
        finally:
            os.unlink(tmp_path)

        return {
            "name": name,
            "sample_rate": profile.sample_rate,
            "f0_mean": profile.pitch.f0_mean,
            "saved": True,
        }

    @app.delete("/profiles/{name}")
    def delete_profile(name: str) -> dict:
        deleted = store.delete(name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
        return {"deleted": name}

    @app.post("/convert")
    async def convert_audio(
        file: UploadFile = File(...),
        pitch_shift: float = Form(default=0.0),
        formant_shift: float = Form(default=1.0),
        quality: str = Form(default="balanced"),
        bit_depth: int = Form(default=16),
    ) -> FileResponse:
        try:
            q = ConversionQuality(quality)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid quality: {quality}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            tmp_in.write(await file.read())
            input_path = tmp_in.name

        output_path = input_path + "_converted.wav"

        try:
            converter = VoiceConverter(q)
            converter.process(
                input_path=input_path,
                output_path=output_path,
                pitch_shift=pitch_shift,
                formant_shift=formant_shift,
                bit_depth=bit_depth,
            )
        finally:
            os.unlink(input_path)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="converted.wav",
            background=None,
        )

    return app
