# Voico API Specification

## Base Service

- Framework: FastAPI
- Application factory: `voico.api.app.create_app`
- Content types: `application/json`, `multipart/form-data`, `audio/wav`

## Endpoints

### `GET /health`

Response:

```json
{
  "status": "ok"
}
```

### `GET /profiles`

Response:

```json
[
  {
    "name": "speaker_a",
    "sample_rate": 44100,
    "f0_mean": 180.2,
    "created_at": "2026-03-01 10:20:33"
  }
]
```

### `GET /profiles/{name}`

Response fields:

- `name`: profile identifier
- `sample_rate`: integer
- `f0_mean`: float
- `f0_std`: float
- `hnr`: float
- `mean_formants`: array of floats
- `spectral_tilt`: float

Errors:

- `404` when profile does not exist
- `500` when profile exists but cannot be loaded

### `POST /profiles/{name}/analyze`

Form fields:

- `file`: WAV file
- `quality`: one of `turbo|fast|balanced|high|ultra|master`

Behavior:

- Loads uploaded audio
- Builds a voice profile using quality-dependent FFT and hop settings
- Saves profile under `name`

Response:

```json
{
  "name": "speaker_a",
  "sample_rate": 44100,
  "f0_mean": 180.2,
  "saved": true
}
```

Errors:

- `400` for invalid quality value

### `DELETE /profiles/{name}`

Response:

```json
{
  "deleted": "speaker_a"
}
```

Errors:

- `404` when profile does not exist

### `POST /convert`

Form fields:

- `file`: WAV file
- `pitch_shift`: float semitones
- `formant_shift`: float scale
- `quality`: one of `turbo|fast|balanced|high|ultra|master`
- `bit_depth`: one of `16|32`

Behavior:

- Converts uploaded audio and returns `audio/wav`
- Temporary files are removed after response completion

Errors:

- `400` for invalid quality value

## Reliability Boundaries

- Profile operations validate profile existence before retrieval or deletion.
- Conversion and analysis isolate uploaded content in temporary files.
- Output artifacts from `/convert` are scheduled for post-response deletion.
