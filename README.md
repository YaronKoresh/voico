# Modular Voice Converter

A professional, modular Python framework for voice conversion and audio manipulation.

---

## Features

* **Pitch Shifting**: High-quality pitch transposition (semitones).
* **Formant Shifting**: Independent control over vocal tract size (timbre).
* **Auto-Matching (Cloning)**: Automatically analyzes a target voice and calculates the necessary pitch/formant shifts to mimic it.
* **Phase Reconstruction**: Uses Griffin-Lim for high-fidelity output.
* **Quality Presets**: From "Turbo" (real-time capable) to "Master" (high-precision offline rendering).
* **Modular Design**: Clean separation of Analysis, DSP, and Core logic.

---

## Installation

### Prerequisites

* Python 3.8+
* Pip

### Quick Start

```bash
pip install .
```

### Development Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

### Verify Installation

```bash
poe check
```

---

## Usage

### CLI

Run via the `voico` command.

```bash
# Shift pitch UP by 2 semitones, make voice "smaller" (1.1x formant)
voico input.wav -p 2.0 -f 1.1

# Shift pitch DOWN by 5 semitones, make voice "deeper" (0.9x formant)
voico input.wav -p -5.0 -f 0.9

# Auto-match a target voice
voico source_voice.wav -t target_voice.wav

# Use highest quality preset
voico input.wav -p 2.0 -q master

# Custom output path with verbose logging
voico input.wav -p 1.5 -o result.wav -v
```

**Quality Presets:** `turbo`, `fast`, `balanced` (default), `high`, `ultra`, `master`

### Python API

```python
from voico import VoiceConverter, ConversionQuality

converter = VoiceConverter(ConversionQuality.BALANCED)

# Manual pitch/formant shift
converter.process(
    input_path="source.wav",
    output_path="output.wav",
    pitch_shift=2.0,       # semitones
    formant_shift=1.1,     # scale factor
)

# Auto-match a target voice
converter.process(
    input_path="source.wav",
    output_path="cloned.wav",
    target_path="target.wav",
)
```

---

## Architecture

```
voico/
  core/           Domain types, constants, config, errors
    config.py       Quality presets (6 levels: turbo -> master)
    constants.py    Audio processing constants
    types.py        PitchContour, FormantTrack, SpectralFeatures, VoiceProfile
    errors.py       VoicoError hierarchy

  analysis/       Voice analysis pipeline
    pitch.py        F0 detection (librosa pyin / YIN autocorrelation fallback)
    formant.py      LPC-based formant extraction (Levinson-Durbin)
    spectral.py     Cepstral envelope, spectral tilt, harmonic stats
    profile.py      Orchestrates analysis into VoiceProfile
    matcher.py      Compares source/target profiles for auto-matching

  dsp/            Signal processing
    shifter.py      Pitch shifting, formant warping, tilt correction
    phase.py        Griffin-Lim phase reconstruction

  utils/          Shared utilities
    audio_io.py     Load/save WAV (librosa or scipy fallback)
    decorators.py   Performance timing
    math_utils.py   Safe division
```

### Processing Pipeline

```
Input WAV -> Load & Normalize -> [Analysis -> Matching] -> Pitch Shift -> Formant Shift -> Phase Reconstruct -> Normalize -> Output WAV
```

1. **Load**: Audio loaded as float32 mono, normalized to 0.95 peak
2. **Analysis** (auto-match mode): Builds VoiceProfile from pitch, formant, and spectral analysis
3. **Matching** (auto-match mode): Compares source/target profiles to compute shift parameters
4. **Pitch Shift**: Resamples audio via librosa or linear interpolation
5. **Formant Shift**: Warps spectral envelope using vectorized frequency-axis interpolation
6. **Phase Reconstruction**: Griffin-Lim algorithm restores phase coherence
7. **Output**: Normalized to 0.95 peak, saved as 16-bit PCM WAV

---

## Testing

```bash
# Run all tests
poe test

# Run with coverage
pytest tests/ --cov=voico --cov-report=term-missing

# Full quality check pipeline
poe check
```

**Test coverage: 91%** across 94 tests covering core logic, analysis, DSP, I/O, and CLI.

---

## CI/CD

| Workflow | Trigger | Purpose |
|---|---|---|
| `check.yml` | Pull request | Lint, compile, test (Python 3.8-3.13) |
| `publish.yml` | Release | Publish to PyPI |
| `codeql.yml` | Push/PR | Security scanning |
| `dependabot-auto.yml` | Dependabot | Auto-merge dependency updates |

---

## License & Usage

This project is released under the **Universal Copyleft Source License (UCSL-1.0)**.

* **Free for Open Source, Personal & Commercial Use:** You may use, modify, distribute, and monetize this software, provided your project is fully open-source and complies with UCSL-1.0 terms.
* **Strict Copyleft:** Integrating this code into your application requires releasing your entire source code under the same UCSL-1.0 terms.
* **Proprietary License:** Contact the author for a separate commercial license for closed-source use.

See [LICENSE](./LICENSE) for full terms.
