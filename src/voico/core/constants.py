from typing import ClassVar, Tuple


class AudioConstants:
    MIN_F0_HZ = 50.0
    MAX_F0_HZ = 600.0
    DEFAULT_N_FFT = 2048

    FORMANT_ANALYSIS_SR = 10000
    LPC_ORDER_LOW_PITCH = 16
    PITCH_THRESHOLD_LOW = 120.0

    EPSILON = 1e-10

    DEFAULT_FORMANT_FREQS: ClassVar[Tuple[int, ...]] = (
        500,
        1500,
        2500,
        3500,
        4500,
    )
    DEFAULT_FORMANT_BANDWIDTHS: ClassVar[Tuple[int, ...]] = (
        80,
        100,
        120,
        150,
        200,
    )
    MAX_FORMANT_BANDWIDTH = 400
