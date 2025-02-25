"""
Extracts waveform data from audio files and saves it as a JSON file.
"""

import argparse
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import numpy as np
from pydub import AudioSegment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class WaveformBuffer:
    """
    Stores the minimum and maximum sample values for each pixel.
    """
    MAX_CHANNELS = 24

    def __init__(self):
        self.sample_rate: int = 0
        self.samples_per_pixel: int = 0
        self.bits: int = 16
        self.channels: int = 1
        self.data: List[int] = []  # List of min/max pairs for each channel and pixel

    def set_sample_rate(self, sample_rate: int) -> None:
        """Set the sample rate of the audio."""
        self.sample_rate = sample_rate

    def set_samples_per_pixel(self, samples_per_pixel: int) -> None:
        """Set the number of audio samples per pixel."""
        self.samples_per_pixel = samples_per_pixel

    def set_channels(self, channels: int) -> None:
        """Set the number of audio channels."""
        self.channels = channels

    def get_size(self) -> int:
        """Return the number of points (pixels)."""
        if not self.data:
            return 0
        return len(self.data) // (2 * self.channels)

    def append_samples(self, min_value: int, max_value: int) -> None:
        """Append a min/max pair to the buffer."""
        self.data.append(min_value)
        self.data.append(max_value)

    def get_min_sample(self, channel: int, index: int) -> int:
        """Get the minimum sample value for a channel at the specified index."""
        offset = (index * self.channels + channel) * 2
        return self.data[offset]

    def get_max_sample(self, channel: int, index: int) -> int:
        """Get the maximum sample value for a channel at the specified index."""
        offset = (index * self.channels + channel) * 2 + 1
        return self.data[offset]

    def set_samples(self, channel: int, index: int, min_value: int, max_value: int) -> None:
        """Set the min/max pair for a channel at the specified index."""
        offset = (index * self.channels + channel) * 2
        self.data[offset] = min_value
        self.data[offset + 1] = max_value

    def save_as_json(self, filename: str, bits: int = 16) -> bool:
        """Save the waveform data as a JSON file."""
        size = self.get_size()
        version = 2

        try:
            data = {
                "version": version,
                "channels": self.channels,
                "sample_rate": self.sample_rate,
                "samples_per_pixel": self.samples_per_pixel,
                "bits": bits,
                "length": size,
                "data": []
            }

            divisor = 256 if bits == 8 else 1

            # Convert numpy types to native Python integers for JSON serialization
            for value in self.data:
                data["data"].append(int(value) // divisor)

            with open(filename, 'w') as f:
                json.dump(data, f)
                f.write('\n')

            return True
        except Exception as e:
            logger.error(f"Error saving JSON file: {str(e)}")
            return False


class ScaleFactor(ABC):
    """Abstract base class for determining the samples per pixel scaling."""

    @abstractmethod
    def get_samples_per_pixel(self, sample_rate: int) -> int:
        """Calculate the number of samples per pixel."""
        pass


class SamplesPerPixelScaleFactor(ScaleFactor):
    """Scale factor based on samples per pixel."""

    def __init__(self, samples_per_pixel: int):
        if samples_per_pixel < 2:
            raise ValueError("Invalid samples per pixel: must be at least 2")
        self.samples_per_pixel = samples_per_pixel

    def get_samples_per_pixel(self, sample_rate: int) -> int:
        return self.samples_per_pixel


class PixelsPerSecondScaleFactor(ScaleFactor):
    """Scale factor based on pixels per second."""

    def __init__(self, pixels_per_second: int):
        if pixels_per_second <= 0:
            raise ValueError("Invalid pixels per second: must be greater than zero")
        self.pixels_per_second = pixels_per_second

    def get_samples_per_pixel(self, sample_rate: int) -> int:
        return sample_rate // self.pixels_per_second


class DurationScaleFactor(ScaleFactor):
    """Scale factor based on duration and width in pixels."""

    def __init__(self, start_time: float, end_time: float, width_pixels: int):
        if end_time < start_time:
            raise ValueError(f"Invalid end time, must be greater than {start_time}")
        if width_pixels < 1:
            raise ValueError("Invalid image width: minimum 1")

        self.start_time = start_time
        self.end_time = end_time
        self.width_pixels = width_pixels

    def get_samples_per_pixel(self, sample_rate: int) -> int:
        seconds = self.end_time - self.start_time
        width_samples = int(seconds * sample_rate)
        return max(2, width_samples // self.width_pixels)  # Ensure minimum of 2


class WaveformGenerator:
    """
    Generates waveform data from audio samples.
    """
    # Constants
    MAX_SAMPLE = 32767
    MIN_SAMPLE = -32768

    def __init__(self, buffer: WaveformBuffer, split_channels: bool, scale_factor: ScaleFactor):
        self.buffer = buffer
        self.scale_factor = scale_factor
        self.split_channels = split_channels
        self.channels = 0
        self.output_channels = 0
        self.samples_per_pixel = 0
        self.count = 0
        self.min: List[int] = []  # Min values per channel
        self.max: List[int] = []  # Max values per channel

    def init(self, sample_rate: int, channels: int, frame_count: int = 0, buffer_size: int = 0) -> bool:
        """Initialize the generator."""
        if channels < 1 or channels > WaveformBuffer.MAX_CHANNELS:
            logger.error(f"Cannot generate waveform data from audio file with {channels} channels")
            return False

        self.channels = channels
        self.samples_per_pixel = self.scale_factor.get_samples_per_pixel(sample_rate)

        if self.samples_per_pixel < 2:
            logger.error("Invalid zoom: minimum 2")
            return False

        self.output_channels = channels if self.split_channels else 1

        self.buffer.set_samples_per_pixel(self.samples_per_pixel)
        self.buffer.set_sample_rate(sample_rate)
        self.buffer.set_channels(self.output_channels)

        logger.info(f"Generating waveform data...")
        logger.info(f"Samples per pixel: {self.samples_per_pixel}")
        logger.info(f"Input channels: {self.channels}")
        logger.info(f"Output channels: {self.output_channels}")

        # Initialize min/max arrays
        self.min = [self.MAX_SAMPLE] * self.output_channels
        self.max = [self.MIN_SAMPLE] * self.output_channels
        self.reset()

        return True

    def reset(self) -> None:
        """Reset the min/max values and sample counter."""
        for channel in range(self.output_channels):
            self.min[channel] = self.MAX_SAMPLE
            self.max[channel] = self.MIN_SAMPLE

        self.count = 0

    def process(self, input_buffer: np.ndarray) -> bool:
        """Process audio samples and update the waveform buffer."""
        for i in range(len(input_buffer)):
            if self.output_channels == 1:
                # Mix all channels together - prevent overflow by using int32
                sample = int(sum(int(s) for s in input_buffer[i]) // self.channels)

                # Clamp the sample value
                sample = max(self.MIN_SAMPLE, min(self.MAX_SAMPLE, sample))

                self.min[0] = min(self.min[0], sample)
                self.max[0] = max(self.max[0], sample)
            else:
                # Process each channel separately
                for channel in range(self.channels):
                    sample = int(input_buffer[i][channel])

                    # Clamp the sample value
                    sample = max(self.MIN_SAMPLE, min(self.MAX_SAMPLE, sample))

                    self.min[channel] = min(self.min[channel], sample)
                    self.max[channel] = max(self.max[channel], sample)

            self.count += 1

            if self.count == self.samples_per_pixel:
                for channel in range(self.output_channels):
                    self.buffer.append_samples(self.min[channel], self.max[channel])
                self.reset()

        return True

    def done(self) -> None:
        """Finalize the waveform generation."""
        if self.count > 0:
            for channel in range(self.output_channels):
                self.buffer.append_samples(self.min[channel], self.max[channel])
            self.reset()

        logger.info(f"Generated {self.buffer.get_size()} points")


def read_audio_file(filename: str, show_info: bool = True) -> Tuple[Optional[AudioSegment], bool]:
    """Read an audio file and return the audio data."""
    try:
        if filename == '-' or filename == '':
            logger.error("Reading from stdin not supported in this simplified version")
            return None, False

        # Determine file type from extension
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext == '.mp3':
            audio = AudioSegment.from_mp3(filename)
        elif file_ext in ('.wav', '.wave'):
            audio = AudioSegment.from_wav(filename)
        elif file_ext in ('.ogg', '.oga'):
            audio = AudioSegment.from_ogg(filename)
        elif file_ext == '.flac':
            audio = AudioSegment.from_file(filename, "flac")
        else:
            # Try to load the file anyway
            audio = AudioSegment.from_file(filename)

        if show_info:
            logger.info(f"Input file: {filename}")
            logger.info(f"Channels: {audio.channels}")
            logger.info(f"Sample rate: {audio.frame_rate} Hz")
            logger.info(f"Duration: {len(audio) / 1000:.2f} seconds")

        return audio, True
    except Exception as e:
        logger.error(f"Failed to read file: {filename}")
        logger.error(str(e))
        return None, False


def generate_waveform_data(audio: AudioSegment, scale_factor: ScaleFactor, split_channels: bool) -> Optional[
    WaveformBuffer]:
    """Generate waveform data from an audio file."""
    buffer = WaveformBuffer()

    # Create waveform generator
    generator = WaveformGenerator(buffer, split_channels, scale_factor)

    # Initialize the generator
    if not generator.init(audio.frame_rate, audio.channels):
        return None

    # Get the audio samples as a numpy array
    samples = np.array(audio.get_array_of_samples())

    # Convert to 16-bit signed integers if needed
    if audio.sample_width != 2:  # If not 16-bit audio
        # Scale to 16-bit range
        max_sample_value = 2 ** (8 * audio.sample_width - 1) - 1
        samples = (samples / max_sample_value * 32767).astype(np.int16)

    # Reshape the samples array for multi-channel audio
    if audio.channels > 1:
        samples = samples.reshape(-1, audio.channels)
    else:
        samples = samples.reshape(-1, 1)

    # Process the samples
    generator.process(samples)

    # Finalize the waveform generation
    generator.done()

    return buffer


def create_scale_factor(args: argparse.Namespace) -> ScaleFactor:
    """Create appropriate scale factor based on command line args."""
    if args.zoom:
        return SamplesPerPixelScaleFactor(args.zoom)
    elif args.pixels_per_second:
        return PixelsPerSecondScaleFactor(args.pixels_per_second)
    elif args.end:
        return DurationScaleFactor(args.start, args.end, args.width)
    else:
        # Default
        return SamplesPerPixelScaleFactor(256)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract waveform data from audio files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input_file', help='Input audio file')
    parser.add_argument('output_file', help='Output JSON file')

    # Scale options - mutually exclusive
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument('--zoom', '-z', type=int, default=256,
                             help='Samples per pixel')
    scale_group.add_argument('--pixels-per-second', type=int,
                             help='Pixels per second')
    scale_group.add_argument('--end', '-e', type=float,
                             help='End time (seconds)')

    # Other options
    parser.add_argument('--start', '-s', type=float, default=0.0,
                        help='Start time (seconds)')
    parser.add_argument('--width', '-w', type=int, default=800,
                        help='Width in pixels (for --end option)')
    parser.add_argument('--split-channels', action='store_true',
                        help='Split channels instead of mixing to mono')
    parser.add_argument('--bits', '-b', type=int, choices=[8, 16], default=16,
                        help='Resolution (8 or 16 bits)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress messages')

    return parser.parse_args()


def main() -> int:
    """Main function."""
    args = parse_arguments()

    # Configure logging
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Validate input file
    if not os.path.exists(args.input_file):
        logger.error(f"Error: Input file '{args.input_file}' not found")
        return 1

    # Read the audio file
    audio, success = read_audio_file(args.input_file)
    if not success or audio is None:
        return 1

    # Create appropriate scale factor
    try:
        scale_factor = create_scale_factor(args)
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Generate waveform data
    buffer = generate_waveform_data(audio, scale_factor, args.split_channels)
    if buffer is None:
        return 1

    # Save the waveform data
    if not buffer.save_as_json(args.output_file, args.bits):
        logger.error(f"Error: Failed to save waveform data to '{args.output_file}'")
        return 1

    logger.info(f"Waveform data saved to '{args.output_file}'")
    return 0


if __name__ == "__main__":
    exit(main())
