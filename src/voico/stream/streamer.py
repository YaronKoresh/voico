import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Iterator, Optional

import numpy as np
import scipy.signal

from ..core.config import ConversionQuality, QualitySettings
from ..core.constants import AudioConstants


class VoiceStreamProcessor:
    def __init__(
        self,
        sample_rate: int = 44100,
        pitch_shift: float = 0.0,
        formant_shift: float = 1.0,
        quality: ConversionQuality = ConversionQuality.FAST,
    ):
        self.sample_rate = sample_rate
        self.pitch_shift = pitch_shift
        self.formant_shift = formant_shift
        self._settings = QualitySettings.from_preset(quality)
        self._n_fft = AudioConstants.DEFAULT_N_FFT
        self._hop_length = self._n_fft // self._settings.hop_divisor
        self._window = scipy.signal.get_window("hann", self._n_fft)
        self._buffer = np.zeros(self._n_fft * 2, dtype=np.float32)
        self._output_buffer = np.zeros(self._n_fft * 2, dtype=np.float32)
        self._buffer_pos = 0

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk = chunk.astype(np.float32)
        n = len(chunk)
        output_samples = np.zeros(n, dtype=np.float32)
        pos = 0

        while pos < n:
            space = self._hop_length - self._buffer_pos % self._hop_length
            take = min(space, n - pos)
            write_start = self._buffer_pos % len(self._buffer)
            write_end = min(write_start + take, len(self._buffer))
            actual_take = write_end - write_start
            self._buffer[write_start:write_end] = chunk[pos:pos + actual_take]
            self._buffer_pos += actual_take
            pos += actual_take

            if self._buffer_pos > 0 and self._buffer_pos % self._hop_length == 0:
                buf_start = max(0, self._buffer_pos - self._n_fft)
                frame = self._buffer[buf_start % len(self._buffer):
                                     buf_start % len(self._buffer) + min(self._n_fft, len(self._buffer))]
                if len(frame) < self._n_fft:
                    frame = np.pad(frame, (0, self._n_fft - len(frame)))
                processed = self._process_frame(frame[:self._n_fft])
                out_start = max(0, pos - len(processed))
                out_len = min(len(processed), n - out_start)
                output_samples[out_start:out_start + out_len] += processed[:out_len] * 0.5

        return output_samples

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        windowed = frame * self._window.astype(np.float32)

        if abs(self.pitch_shift) < 0.01 and abs(self.formant_shift - 1.0) < 0.01:
            return windowed

        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        if abs(self.formant_shift - 1.0) >= 0.01:
            n_bins = len(magnitude)
            target_bins = np.clip(
                np.arange(n_bins) * self.formant_shift, 0, n_bins - 1
            )
            magnitude = np.interp(
                np.arange(n_bins), target_bins, magnitude
            )

        if abs(self.pitch_shift) >= 0.01:
            factor = 2.0 ** (self.pitch_shift / 12.0)
            n_bins = len(magnitude)
            src_bins = np.arange(n_bins) / factor
            valid = src_bins < n_bins
            new_mag = np.zeros(n_bins, dtype=np.float32)
            new_mag[valid] = np.interp(
                src_bins[valid], np.arange(n_bins), magnitude
            )
            magnitude = new_mag

        reconstructed = magnitude * np.exp(1j * phase)
        return np.fft.irfft(reconstructed).astype(np.float32)[:len(frame)]

    def flush(self) -> np.ndarray:
        remaining = np.zeros(self._n_fft, dtype=np.float32)
        result = self._process_frame(remaining)
        self._buffer[:] = 0
        self._output_buffer[:] = 0
        self._buffer_pos = 0
        return result

    def stream(
        self,
        audio_iterator: Iterator[np.ndarray],
    ) -> Iterator[np.ndarray]:
        for chunk in audio_iterator:
            output = self.process_chunk(chunk)
            if np.any(output != 0):
                yield output
        final = self.flush()
        if np.any(final != 0):
            yield final

    async def astream(
        self,
        audio_iterator: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        async for chunk in audio_iterator:
            output = await loop.run_in_executor(
                executor, self.process_chunk, chunk
            )
            if np.any(output != 0):
                yield output
        final = await loop.run_in_executor(executor, self.flush)
        if np.any(final != 0):
            yield final
