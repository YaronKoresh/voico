# 🎙️ Modular Voice Converter

A professional, modular Python framework for voice conversion and audio manipulation.

---

## ✨ Features

* **Pitch Shifting**: High-quality pitch transposition (semitones).
* **Formant Shifting**: Independent control over vocal tract size (timbre).
* **Auto-Matching (Cloning)**: Automatically analyzes a target voice and calculates the necessary pitch/formant shifts to mimic it.
* **Phase Reconstruction**: Uses Griffin-Lim for high-fidelity output.
* **Quality Presets**: From "Turbo" (real-time capable) to "Master" (high-precision offline rendering).
* **Modular Design**: Clean separation of Analysis, DSP, and Core logic.

---

## ⚙️ Installation

### Prerequisites

* Python 3.8+
* Pip

### Dependencies

Install the required packages:

```bash
pip install .
```

---

## 🚀 Usage

Run the tool via the `voico` command.

### 1. Manual Mode

Manually adjust pitch (semitones) and formant (scale factor).

```bash
# Shift pitch UP by 2 semitones, make voice "smaller" (1.1x)
voico input.wav -p 2.0 -f 1.1

# Shift pitch DOWN by 5 semitones, make voice "deeper" (0.9x)
voico input.wav -p -5.0 -f 0.9
```

### 2. Auto-Match (Voice Cloning) Mode

Provide a target audio file, and the system will calculate the difference.

```bash
voico source_voice.wav -t target_voice.wav
```

### 3. Quality Settings

Control the trade-off between speed and quality using the `-q` flag.
**Options:** `turbo`, `fast`, `balanced` (default), `high`, `ultra`, `master`.

```bash
voico input.wav -p 2.0 -q master
```

---

## 🔧 Architecture Overview

The system is built on a **Pipeline** pattern:

1. **Input**: Audio is loaded and normalized.
2. **Analysis (Optional/Auto)**:
* `PitchDetector`: Extracts fundamental frequency.
* `FormantAnalyzer`: Uses Linear Predictive Coding (LPC) to find vocal tract resonances.

3. **Matching (Auto Mode)**: `VoiceMatcher` compares Source vs. Target profiles.
4. **DSP**:
* `SpectralShifter`: Resamples audio (pitch) and warps spectral envelope (formants).
* `PhaseReconstructor`: Rebuilds phase coherence lost during spectral warping.

5. **Output**: Audio is normalized and saved as 16-bit PCM WAV.

---

**License:** [UCSL-1.0](./LICENSE).
