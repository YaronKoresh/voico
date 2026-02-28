import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..core.types import FormantTrack, PitchContour, SpectralFeatures, VoiceProfile

DEFAULT_DB_PATH = str(Path.home() / ".voico" / "profiles.db")


def _serialize_profile(profile: VoiceProfile) -> str:
    data = {
        "pitch": {
            "f0": profile.pitch.f0.tolist(),
            "voiced_mask": profile.pitch.voiced_mask.tolist(),
            "f0_mean": profile.pitch.f0_mean,
            "f0_std": profile.pitch.f0_std,
            "harmonic_to_noise_ratio": profile.pitch.harmonic_to_noise_ratio,
        },
        "formants": {
            "frequencies": profile.formants.frequencies.tolist(),
            "bandwidths": profile.formants.bandwidths.tolist(),
            "mean_frequencies": profile.formants.mean_frequencies.tolist(),
            "mean_bandwidths": profile.formants.mean_bandwidths.tolist(),
        },
        "spectral": {
            "envelope": profile.spectral.envelope.tolist(),
            "spectral_tilt": profile.spectral.spectral_tilt,
        },
        "harmonic_ratios": profile.harmonic_ratios.tolist(),
        "harmonic_energy": profile.harmonic_energy.tolist(),
        "sample_rate": profile.sample_rate,
    }
    return json.dumps(data)


def _deserialize_profile(data: str) -> VoiceProfile:
    d = json.loads(data)
    pitch = PitchContour(
        f0=np.array(d["pitch"]["f0"], dtype=np.float32),
        voiced_mask=np.array(d["pitch"]["voiced_mask"], dtype=bool),
        f0_mean=d["pitch"]["f0_mean"],
        f0_std=d["pitch"]["f0_std"],
        harmonic_to_noise_ratio=d["pitch"]["harmonic_to_noise_ratio"],
    )
    formants = FormantTrack(
        frequencies=np.array(d["formants"]["frequencies"], dtype=np.float32),
        bandwidths=np.array(d["formants"]["bandwidths"], dtype=np.float32),
        mean_frequencies=np.array(d["formants"]["mean_frequencies"], dtype=np.float32),
        mean_bandwidths=np.array(d["formants"]["mean_bandwidths"], dtype=np.float32),
    )
    spectral = SpectralFeatures(
        envelope=np.array(d["spectral"]["envelope"], dtype=np.float32),
        spectral_tilt=d["spectral"]["spectral_tilt"],
    )
    return VoiceProfile(
        pitch=pitch,
        formants=formants,
        spectral=spectral,
        harmonic_ratios=np.array(d["harmonic_ratios"], dtype=np.float32),
        harmonic_energy=np.array(d["harmonic_energy"], dtype=np.float32),
        sample_rate=d["sample_rate"],
    )


class ProfileStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    name TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    sample_rate INTEGER NOT NULL,
                    f0_mean REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.commit()

    def save(self, name: str, profile: VoiceProfile) -> None:
        serialized = _serialize_profile(profile)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO profiles (name, data, sample_rate, f0_mean)
                VALUES (?, ?, ?, ?)
                """,
                (name, serialized, profile.sample_rate, profile.pitch.f0_mean),
            )
            conn.commit()

    def load(self, name: str) -> Optional[VoiceProfile]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM profiles WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            return None
        return _deserialize_profile(row[0])

    def delete(self, name: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM profiles WHERE name = ?", (name,)
            )
            conn.commit()
        return cursor.rowcount > 0

    def list_profiles(self) -> List[Dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, sample_rate, f0_mean, created_at FROM profiles ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "name": row[0],
                "sample_rate": row[1],
                "f0_mean": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]

    def exists(self, name: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM profiles WHERE name = ?", (name,)
            ).fetchone()
        return row is not None
