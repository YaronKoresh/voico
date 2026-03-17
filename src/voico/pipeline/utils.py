import numpy as np


def compute_snr(original: np.ndarray, processed: np.ndarray) -> float:
    n = min(len(original), len(processed))
    if n == 0:
        return 0.0
    orig = original[:n].astype(np.float64)
    proc = processed[:n].astype(np.float64)
    signal_power = np.mean(orig**2)
    noise_power = np.mean((orig - proc) ** 2)
    if noise_power < 1e-10:
        return 60.0
    return float(10.0 * np.log10(max(signal_power / noise_power, 1e-10)))


def compute_spectral_centroid(audio: np.ndarray, sr: int, n_fft: int) -> float:
    if len(audio) < n_fft:
        return 0.0
    spectrum = np.abs(np.fft.rfft(audio[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    total = np.sum(spectrum)
    if total < 1e-10:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)
