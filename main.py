"""
Extracts waveform data from audio files and saves it as a JSON file.
"""

import argparse
import json
import logging
import os
from enum import IntEnum
from typing import List, Tuple, Union

import numpy as np
from pydub import AudioSegment

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class BitDepth(IntEnum):
    """Supported bit depths for waveform output."""
    EIGHT = 8
    SIXTEEN = 16


class WaveformBuffer:
    """Stores waveform data as min/max sample pairs per channel per pixel."""

    MAX_CHANNELS = 24

    def __init__(self):
        self.sample_rate: int = 0
        self.samples_per_pixel: int = 0
        self.bits: int = BitDepth.SIXTEEN
        self.channels: int = 1
        self.data: List[Tuple[int, int]] = []  # List of (min, max) pairs

    def set_sample_rate(self, sample_rate: int) -> None:
        """Set the audio sample rate."""
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        self.sample_rate = sample_rate

    def set_samples_per_pixel(self, samples_per_pixel: int) -> None:
        """Set the number of audio samples per pixel."""
        if samples_per_pixel < 2:
            raise ValueError("Samples per pixel must be at least 2")
        self.samples_per_pixel = samples_per_pixel

    def set_channels(self, channels: int) -> None:
        """Set the number of audio channels."""
        if not 1 <= channels <= self.MAX_CHANNELS:
            raise ValueError(f"Channels must be between 1 and {self.MAX_CHANNELS}")
        self.channels = channels

    def get_size(self) -> int:
        """Return the number of points (pixels)."""
        return len(self.data) // self.channels

    def append_samples(self, min_value: int, max_value: int) -> None:
        """Append a min/max pair to the buffer."""
        self.data.append((min_value, max_value))

    def save_as_json(self, filename: str, bits: BitDepth = BitDepth.SIXTEEN) -> None:
        """Save waveform data as a formatted JSON file."""
        size = self.get_size()
        divisor = 256 if bits == BitDepth.EIGHT else 1

        try:
            data = {
                "version": 2,
                "channels": self.channels,
                "sample_rate": self.sample_rate,
                "samples_per_pixel": self.samples_per_pixel,
                "bits": bits,
                "length": size,
                "data": [int(v) // divisor for pair in self.data for v in pair],  # Flatten min/max pairs
            }
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)  # Pretty-print with 4-space indentation
                f.write("\n")  # Optional: ensure a trailing newline
            logger.info(f"Waveform data saved to '{filename}'")
        except Exception as e:
            raise IOError(f"Failed to save JSON file: {e}")


class ScaleFactor:
    """Abstract base class for determining samples per pixel scaling."""

    def get_samples_per_pixel(self, sample_rate: int) -> int:
        """Calculate the number of samples per pixel."""
        raise NotImplementedError("Subclasses must implement this method")


class SamplesPerPixelScaleFactor(ScaleFactor):
    """Scale factor based on a fixed samples-per-pixel value."""

    def __init__(self, samples_per_pixel: int):
        if samples_per_pixel < 2:
            raise ValueError("Samples per pixel must be at least 2")
        self.samples_per_pixel = samples_per_pixel

    def get_samples_per_pixel(self, sample_rate: int) -> int:
        return self.samples_per_pixel


class PixelsPerSecondScaleFactor(ScaleFactor):
    """Scale factor based on pixels per second."""

    def __init__(self, pixels_per_second: int):
        if pixels_per_second <= 0:
            raise ValueError("Pixels per second must be positive")
        self.pixels_per_second = pixels_per_second

    def get_samples_per_pixel(self, sample_rate: int) -> int:
        return max(2, sample_rate // self.pixels_per_second)


class DurationScaleFactor(ScaleFactor):
    """Scale factor based on duration and width in pixels."""

    def __init__(self, start_time: float, end_time: float, width_pixels: int):
        if end_time < start_time:
            raise ValueError(f"End time ({end_time}) must be greater than start time ({start_time})")
        if width_pixels < 1:
            raise ValueError("Width in pixels must be at least 1")
        self.start_time = start_time
        self.end_time = end_time
        self.width_pixels = width_pixels

    def get_samples_per_pixel(self, sample_rate: int) -> int:
        duration = self.end_time - self.start_time
        total_samples = int(duration * sample_rate)
        return max(2, total_samples // self.width_pixels)


class WaveformGenerator:
    """Generates waveform data from audio samples."""

    def __init__(self, buffer: WaveformBuffer, split_channels: bool, scale_factor: ScaleFactor):
        self.buffer = buffer
        self.split_channels = split_channels
        self.scale_factor = scale_factor
        self.channels = 0
        self.output_channels = 0
        self.samples_per_pixel = 0
        self.min_sample = -32768  # Configurable for different bit depths if needed
        self.max_sample = 32767

    def init(self, sample_rate: int, channels: int) -> None:
        """Initialize the waveform generator."""
        if not 1 <= channels <= WaveformBuffer.MAX_CHANNELS:
            raise ValueError(f"Unsupported number of channels: {channels}")
        self.channels = channels
        self.samples_per_pixel = self.scale_factor.get_samples_per_pixel(sample_rate)
        self.output_channels = channels if self.split_channels else 1

        self.buffer.set_samples_per_pixel(self.samples_per_pixel)
        self.buffer.set_sample_rate(sample_rate)
        self.buffer.set_channels(self.output_channels)

        logger.info(f"Generating waveform: samples_per_pixel={self.samples_per_pixel}, "
                    f"input_channels={self.channels}, output_channels={self.output_channels}")

    def _clamp_sample(self, sample: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """Clamp sample values to valid range."""
        return np.clip(sample, self.min_sample, self.max_sample)

    def process(self, input_buffer: np.ndarray) -> None:
        """Process audio samples using vectorized operations."""
        num_samples = len(input_buffer)
        num_points = num_samples // self.samples_per_pixel

        if num_points == 0:
            return

        # Trim and reshape for processing
        trimmed = input_buffer[:num_points * self.samples_per_pixel]
        reshaped = trimmed.reshape(num_points, self.samples_per_pixel, self.channels)

        if self.output_channels == 1:
            # Mix channels into mono
            mixed = reshaped.mean(axis=2, dtype=np.int32)  # Avoid overflow
            clamped = self._clamp_sample(mixed)
            min_vals = clamped.min(axis=1)
            max_vals = clamped.max(axis=1)
            for i in range(num_points):
                self.buffer.append_samples(int(min_vals[i]), int(max_vals[i]))
        else:
            # Process each channel separately
            clamped = self._clamp_sample(reshaped)
            min_vals = clamped.min(axis=1)
            max_vals = clamped.max(axis=1)
            for i in range(num_points):
                for ch in range(self.channels):
                    self.buffer.append_samples(int(min_vals[i, ch]), int(max_vals[i, ch]))

        # Process remainder recursively
        remainder = input_buffer[num_points * self.samples_per_pixel:]
        if len(remainder) > 0:
            self.process(remainder)

    def done(self) -> None:
        """Finalize waveform generation."""
        logger.info(f"Generated {self.buffer.get_size()} points")


def read_audio_file(filename: str, show_info: bool = True) -> AudioSegment:
    """Read an audio file and return the audio data."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Input file '{filename}' not found")

    try:
        audio = AudioSegment.from_file(filename)
        if audio.frame_count() == 0:
            raise ValueError("Audio file is empty")

        if show_info:
            logger.info(f"Input file: {filename}")
            logger.info(f"Channels: {audio.channels}")
            logger.info(f"Sample rate: {audio.frame_rate} Hz")
            logger.info(f"Duration: {len(audio) / 1000:.2f} seconds")
        return audio
    except Exception as e:
        raise IOError(f"Failed to read audio file '{filename}': {e}")


def generate_waveform_data(audio: AudioSegment, scale_factor: ScaleFactor, split_channels: bool) -> WaveformBuffer:
    """Generate waveform data from an audio file."""
    buffer = WaveformBuffer()
    generator = WaveformGenerator(buffer, split_channels, scale_factor)

    # Initialize generator
    generator.init(audio.frame_rate, audio.channels)

    # Convert audio samples to numpy array
    samples = np.array(audio.get_array_of_samples())
    if audio.sample_width != 2:  # Adjust if not 16-bit
        max_value = 2 ** (8 * audio.sample_width - 1) - 1
        samples = (samples / max_value * 32767).astype(np.int16)

    # Reshape for multi-channel
    samples = samples.reshape(-1, audio.channels) if audio.channels > 1 else samples.reshape(-1, 1)

    # Process samples
    generator.process(samples)
    generator.done()

    return buffer


def create_scale_factor(args: argparse.Namespace) -> ScaleFactor:
    """Create a scale factor based on command-line arguments."""
    if args.pixels_per_second:
        return PixelsPerSecondScaleFactor(args.pixels_per_second)
    elif args.end is not None:
        return DurationScaleFactor(args.start, args.end, args.width)
    return SamplesPerPixelScaleFactor(args.zoom)  # Default


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract waveform data from audio files and save as JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file", help="Input audio file (e.g., .mp3, .wav)")
    parser.add_argument("output_file", help="Output JSON file")

    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument(
        "--zoom", "-z", type=int, default=256, help="Samples per pixel (e.g., 256 for standard resolution)"
    )
    scale_group.add_argument(
        "--pixels-per-second", type=int, help="Pixels per second (e.g., 100)"
    )
    scale_group.add_argument(
        "--end", "-e", type=float, help="End time in seconds (use with --start and --width)"
    )

    parser.add_argument("--start", "-s", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--width", "-w", type=int, default=800, help="Width in pixels (for --end)")
    parser.add_argument("--split-channels", action="store_true",
                        help="Keep channels separate instead of mixing to mono")
    parser.add_argument(
        "--bits", "-b", type=int, choices=[BitDepth.EIGHT, BitDepth.SIXTEEN], default=BitDepth.SIXTEEN,
        help="Output bit depth (8 or 16)"
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress messages")
    parser.add_argument("--version", action="version", version="Waveform Extractor 1.0")

    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    args = parse_arguments()

    # Configure logging
    if args.quiet:
        logger.setLevel(logging.WARNING)

    try:
        # Read audio
        audio = read_audio_file(args.input_file)

        # Create scale factor
        scale_factor = create_scale_factor(args)

        # Generate waveform
        buffer = generate_waveform_data(audio, scale_factor, args.split_channels)

        # Save output
        buffer.save_as_json(args.output_file, BitDepth(args.bits))
        return 0

    except (ValueError, IOError, FileNotFoundError) as e:
        logger.error(str(e))
        return 1


if __name__ == "__main__":
    exit(main())
