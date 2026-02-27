import numpy as np
import pytest


@pytest.fixture
def sine_wave_440hz() -> np.ndarray:
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def sine_wave_220hz() -> np.ndarray:
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * 220 * t).astype(np.float32)


@pytest.fixture
def white_noise() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(44100).astype(np.float32)


@pytest.fixture
def silence() -> np.ndarray:
    return np.zeros(44100, dtype=np.float32)


@pytest.fixture
def sample_rate() -> int:
    return 44100
