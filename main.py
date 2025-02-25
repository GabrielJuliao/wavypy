#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
from pydub import AudioSegment
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any


class WaveformBuffer:
    """
    Stores the minimum and maximum sample values for each pixel.
    Equivalent to the C++ WaveformBuffer class.
    """

    def __init__(self):
        self.sample_rate = 0
        self.samples_per_pixel = 0
        self.bits = 16
        self.channels = 1
        self.data = []  # List of min/max pairs for each channel and pixel

    def set_sample_rate(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate

    def set_samples_per_pixel(self, samples_per_pixel: int) -> None:
        self.samples_per_pixel = samples_per_pixel

    def set_channels(self, channels: int) -> None:
        self.channels = channels

    def get_size(self) -> int:
        """Return the number of points (pixels)"""
        if not self.data:
            return 0
        return len(self.data) // (2 * self.channels)

    def append_samples(self, min_value: int, max_value: int) -> None:
        """Append a min/max pair to the buffer"""
        self.data.append(min_value)
        self.data.append(max_value)

    def get_min_sample(self, channel: int, index: int) -> int:
        """Get the minimum sample value for a channel at the specified index"""
        offset = (index * self.channels + channel) * 2
        return self.data[offset]

    def get_max_sample(self, channel: int, index: int) -> int:
        """Get the maximum sample value for a channel at the specified index"""
        offset = (index * self.channels + channel) * 2 + 1
        return self.data[offset]

    def set_samples(self, channel: int, index: int, min_value: int, max_value: int) -> None:
        """Set the min/max pair for a channel at the specified index"""
        offset = (index * self.channels + channel) * 2
        self.data[offset] = min_value
        self.data[offset + 1] = max_value

    def save_as_json(self, filename: str, bits: int = 16) -> bool:
        """Save the waveform data as a JSON file"""
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
            print(f"Error saving JSON file: {str(e)}")
            return False


class ScaleFactor:
    """Base class for determining the samples per pixel scaling"""

    def get_samples_per_pixel(self, sample_rate: int) -> int:
        raise NotImplementedError("Subclasses must implement this method")


class SamplesPerPixelScaleFactor(ScaleFactor):
    """Scale factor based on samples per pixel"""

    def __init__(self, samples_per_pixel: int):
        self.samples_per_pixel = samples_per_pixel

    def get_samples_per_pixel(self, sample_rate: int) -> int:
        return self.samples_per_pixel


class PixelsPerSecondScaleFactor(ScaleFactor):
    """Scale factor based on pixels per second"""

    def __init__(self, pixels_per_second: int):
        self.pixels_per_second = pixels_per_second
        if self.pixels_per_second <= 0:
            raise ValueError("Invalid pixels per second: must be greater than zero")

    def get_samples_per_pixel(self, sample_rate: int) -> int:
        return sample_rate // self.pixels_per_second


class DurationScaleFactor(ScaleFactor):
    """Scale factor based on duration and width in pixels"""

    def __init__(self, start_time: float, end_time: float, width_pixels: int):
        self.start_time = start_time
        self.end_time = end_time
        self.width_pixels = width_pixels

        if end_time < start_time:
            raise ValueError(f"Invalid end time, must be greater than {start_time}")

        if width_pixels < 1:
            raise ValueError("Invalid image width: minimum 1")

    def get_samples_per_pixel(self, sample_rate: int) -> int:
        seconds = self.end_time - self.start_time
        width_samples = int(seconds * sample_rate)
        return width_samples // self.width_pixels


class WaveformGenerator:
    """
    Generates waveform data from audio samples.
    Equivalent to the C++ WaveformGenerator class.
    """

    def __init__(self, buffer: WaveformBuffer, split_channels: bool, scale_factor: ScaleFactor):
        self.buffer = buffer
        self.scale_factor = scale_factor
        self.split_channels = split_channels
        self.channels = 0
        self.output_channels = 0
        self.samples_per_pixel = 0
        self.count = 0
        self.min = []  # Min values per channel
        self.max = []  # Max values per channel

    def init(self, sample_rate: int, channels: int, frame_count: int = 0, buffer_size: int = 0) -> bool:
        """Initialize the generator"""
        MAX_CHANNELS = 24  # Same as the C++ version

        if channels < 1 or channels > MAX_CHANNELS:
            print(f"Cannot generate waveform data from audio file with {channels} channels")
            return False

        self.channels = channels
        self.samples_per_pixel = self.scale_factor.get_samples_per_pixel(sample_rate)

        if self.samples_per_pixel < 2:
            print("Invalid zoom: minimum 2")
            return False

        self.output_channels = channels if self.split_channels else 1

        self.buffer.set_samples_per_pixel(self.samples_per_pixel)
        self.buffer.set_sample_rate(sample_rate)
        self.buffer.set_channels(self.output_channels)

        print(f"Generating waveform data...")
        print(f"Samples per pixel: {self.samples_per_pixel}")
        print(f"Input channels: {self.channels}")
        print(f"Output channels: {self.output_channels}")

        # Initialize min/max arrays
        MAX_SAMPLE = 32767
        MIN_SAMPLE = -32768

        self.min = [MAX_SAMPLE] * self.output_channels
        self.max = [MIN_SAMPLE] * self.output_channels
        self.reset()

        return True

    def reset(self) -> None:
        """Reset the min/max values and sample counter"""
        MAX_SAMPLE = 32767
        MIN_SAMPLE = -32768

        for channel in range(self.output_channels):
            self.min[channel] = MAX_SAMPLE
            self.max[channel] = MIN_SAMPLE

        self.count = 0

    def process(self, input_buffer: np.ndarray) -> bool:
        """Process audio samples and update the waveform buffer"""
        MAX_SAMPLE = 32767
        MIN_SAMPLE = -32768

        for i in range(len(input_buffer)):
            if self.output_channels == 1:
                # Mix all channels together - prevent overflow by using int32
                sample = int(sum(int(s) for s in input_buffer[i]) // self.channels)

                # Clamp the sample value
                if sample > MAX_SAMPLE:
                    sample = MAX_SAMPLE
                elif sample < MIN_SAMPLE:
                    sample = MIN_SAMPLE

                if sample < self.min[0]:
                    self.min[0] = sample

                if sample > self.max[0]:
                    self.max[0] = sample
            else:
                # Process each channel separately
                for channel in range(self.channels):
                    sample = input_buffer[i][channel]

                    # Clamp the sample value
                    if sample > MAX_SAMPLE:
                        sample = MAX_SAMPLE
                    elif sample < MIN_SAMPLE:
                        sample = MIN_SAMPLE

                    if sample < self.min[channel]:
                        self.min[channel] = sample

                    if sample > self.max[channel]:
                        self.max[channel] = sample

            self.count += 1

            if self.count == self.samples_per_pixel:
                for channel in range(self.output_channels):
                    self.buffer.append_samples(self.min[channel], self.max[channel])

                self.reset()

        return True

    def done(self) -> None:
        """Finalize the waveform generation"""
        if self.count > 0:
            for channel in range(self.output_channels):
                self.buffer.append_samples(self.min[channel], self.max[channel])

            self.reset()

        print(f"Generated {self.buffer.get_size()} points")


def read_mp3_file(filename: str, show_info: bool = True) -> Tuple[AudioSegment, bool]:
    """Read an MP3 file and return the audio data"""
    try:
        if filename == '-' or filename == '':
            # Read from stdin - not implemented in this simplified version
            print("Reading from stdin not supported in this simplified version")
            return None, False

        audio = AudioSegment.from_mp3(filename)

        if show_info:
            print(f"Input file: {filename}")
            print(f"Channels: {audio.channels}")
            print(f"Sample rate: {audio.frame_rate} Hz")
            print(f"Duration: {len(audio) / 1000:.2f} seconds")

        return audio, True
    except Exception as e:
        print(f"Failed to read file: {filename}")
        print(f"{str(e)}")
        return None, False


def generate_waveform_data(audio: AudioSegment, scale_factor: ScaleFactor, split_channels: bool) -> Optional[
    WaveformBuffer]:
    """Generate waveform data from an audio file"""
    buffer = WaveformBuffer()

    # Create waveform generator
    generator = WaveformGenerator(buffer, split_channels, scale_factor)

    # Initialize the generator
    if not generator.init(audio.frame_rate, audio.channels):
        return None

    # Get the audio samples as a numpy array
    samples = np.array(audio.get_array_of_samples())

    # Convert to 16-bit signed integers
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


def main():
    """Main function with hardcoded options"""
    parser = argparse.ArgumentParser(description='Extract waveform data from MP3 files')
    parser.add_argument('input_file', help='Input MP3 file')
    parser.add_argument('output_file', help='Output JSON file')
    parser.add_argument('--zoom', type=int, default=256, help='Samples per pixel')
    parser.add_argument('--split-channels', action='store_true', help='Split channels')
    parser.add_argument('--bits', type=int, default=16, choices=[8, 16], help='Resolution (8 or 16 bits)')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return 1

    # Read the MP3 file
    audio, success = read_mp3_file(args.input_file)
    if not success:
        return 1

    # Create scale factor
    scale_factor = SamplesPerPixelScaleFactor(args.zoom)

    # Generate waveform data
    buffer = generate_waveform_data(audio, scale_factor, args.split_channels)
    if buffer is None:
        return 1

    # Save the waveform data
    if not buffer.save_as_json(args.output_file, args.bits):
        print(f"Error: Failed to save waveform data to '{args.output_file}'")
        return 1

    print(f"Waveform data saved to '{args.output_file}'")
    return 0


if __name__ == "__main__":
    exit(main())
