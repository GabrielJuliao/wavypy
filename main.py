"""
Extracts waveform data from audio files with frequency band splitting and saves it as a JSON file.
"""

import argparse
import json
import logging
import os
from enum import IntEnum
from typing import List, Tuple, Union, Dict, Optional

import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, sosfilt, sosfilt_zi

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class BitDepth(IntEnum):
    """Supported bit depths for waveform output."""
    EIGHT = 8
    SIXTEEN = 16


class FilterType(IntEnum):
    """Types of filters for frequency band splitting."""
    LOWPASS = 0
    HIGHPASS = 1
    BANDPASS = 2
    BANDSTOP = 3


class FilterProfile:
    """Defines a filter profile for frequency band splitting."""

    def __init__(
            self,
            filter_type: FilterType,
            cutoff_freq: Union[float, Tuple[float, float]],
            slope_db_oct: float = 12.0,
            order: Optional[int] = None
    ):
        """
        Initialize a filter profile.

        Args:
            filter_type: Type of filter (lowpass, highpass, bandpass, bandstop)
            cutoff_freq: Cutoff frequency in Hz, or tuple of (low, high) for bandpass/bandstop
            slope_db_oct: Filter slope in dB/octave (6, 12, 18, 24, etc.)
            order: Filter order (calculated from slope if not specified)
        """
        self.filter_type = filter_type
        self.cutoff_freq = cutoff_freq
        self.slope_db_oct = slope_db_oct

        # Calculate filter order from slope if not specified
        if order is None:
            # Each 6dB/octave corresponds to a first-order filter
            self.order = max(1, int(slope_db_oct / 6.0))
        else:
            self.order = order


class FrequencyBand:
    """Defines a frequency band with a name and filtering parameters."""

    def __init__(self, name: str, filter_profiles: List[FilterProfile]):
        """
        Initialize a frequency band.

        Args:
            name: Name of the band (e.g., "low", "mid", "high")
            filter_profiles: List of filter profiles to apply
        """
        self.name = name
        self.filter_profiles = filter_profiles


class WaveformBuffer:
    """Stores waveform data as min/max sample pairs per channel per pixel."""

    MAX_CHANNELS = 24

    def __init__(self):
        self.sample_rate: int = 0
        self.samples_per_pixel: int = 0
        self.bits: int = BitDepth.SIXTEEN
        self.channels: int = 1
        self.band_names: List[str] = []
        self.data: Dict[str, List[Tuple[int, int]]] = {}  # Dict of band -> list of (min, max) pairs

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

    def set_bands(self, band_names: List[str]) -> None:
        """Set the frequency bands."""
        self.band_names = band_names
        for band in band_names:
            self.data[band] = []

    def get_size(self, band: str = None) -> int:
        """Return the number of points (pixels) for a specific band or the first band."""
        if band is None:
            band = self.band_names[0] if self.band_names else "fullrange"
        return len(self.data.get(band, [])) // self.channels

    def append_samples(self, band: str, min_value: int, max_value: int) -> None:
        """Append a min/max pair to the buffer for a specific band."""
        if band not in self.data:
            self.data[band] = []
        self.data[band].append((min_value, max_value))

    def save_as_json(self, filename: str, bits: BitDepth = BitDepth.SIXTEEN) -> None:
        """Save waveform data as a formatted JSON file."""
        size = self.get_size()
        divisor = 256 if bits == BitDepth.EIGHT else 1

        try:
            bands_data = {}
            for band in self.data:
                bands_data[band] = [int(v) // divisor for pair in self.data[band] for v in pair]

            data = {
                "version": 3,  # Updated version to indicate band support
                "channels": self.channels,
                "sample_rate": self.sample_rate,
                "samples_per_pixel": self.samples_per_pixel,
                "bits": bits,
                "length": size,
                "bands": self.band_names if self.band_names else ["fullrange"],
                "data": bands_data
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


class BandSplitter:
    """Handles the splitting of audio into frequency bands."""

    def __init__(self, sample_rate: int, bands: List[FrequencyBand]):
        """
        Initialize the band splitter.

        Args:
            sample_rate: Audio sample rate in Hz
            bands: List of frequency bands to split into
        """
        self.sample_rate = sample_rate
        self.bands = bands
        self.filters_cache = {}  # Cache for filter coefficients

    def _create_filter(self, profile: FilterProfile) -> Tuple:
        """
        Create a filter based on the provided profile.

        Returns:
            Tuple of (sos, zi) where sos is second-order filter sections and
            zi is the initial filter state.
        """
        nyquist = 0.5 * self.sample_rate

        # Convert cutoff frequencies to normalized frequency (0 to 1)
        if isinstance(profile.cutoff_freq, tuple):
            # Bandpass or bandstop with two cutoff frequencies
            normalized_cutoff = (profile.cutoff_freq[0] / nyquist, profile.cutoff_freq[1] / nyquist)
        else:
            # Lowpass or highpass with one cutoff frequency
            normalized_cutoff = profile.cutoff_freq / nyquist

        # Create the Butterworth filter
        btype = FilterType(profile.filter_type).name.lower()
        sos = butter(profile.order, normalized_cutoff, btype=btype, output='sos')

        # Create initial filter state
        zi = sosfilt_zi(sos)
        # For a single section filter, zi needs to be reshaped correctly
        if sos.ndim == 2 and sos.shape[0] == 1:
            zi = zi.reshape(1, 2)

        return sos, zi

    def process_sample(self, samples: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split the audio samples into frequency bands.

        Args:
            samples: Audio samples as numpy array [samples, channels]

        Returns:
            Dictionary of band name -> filtered samples
        """
        results = {}

        # Process each band
        for band in self.bands:
            filtered_samples = samples.copy()

            # Apply each filter profile in this band
            for profile in band.filter_profiles:
                # Create or retrieve filter coefficients
                filter_key = (band.name, profile.filter_type,
                              str(profile.cutoff_freq), profile.order)
                if filter_key not in self.filters_cache:
                    self.filters_cache[filter_key] = self._create_filter(profile)

                sos, zi = self.filters_cache[filter_key]

                # Apply the filter
                for channel in range(filtered_samples.shape[1]):
                    # Create a fresh initial state for each channel to avoid filter state interference
                    channel_zi = zi.copy()
                    filtered_samples[:, channel], _ = sosfilt(
                        sos, filtered_samples[:, channel], zi=channel_zi
                    )

            results[band.name] = filtered_samples

        return results


class WaveformGenerator:
    """Generates waveform data from audio samples."""

    def __init__(self, buffer: WaveformBuffer, split_channels: bool, scale_factor: ScaleFactor,
                 bands: Optional[List[FrequencyBand]] = None):
        self.buffer = buffer
        self.split_channels = split_channels
        self.scale_factor = scale_factor
        self.bands = bands
        self.band_splitter = None
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

        # Initialize band splitter if bands are defined
        if self.bands:
            band_names = [band.name for band in self.bands]
            self.buffer.set_bands(band_names)
            self.band_splitter = BandSplitter(sample_rate, self.bands)
        else:
            self.buffer.set_bands(["fullrange"])

        logger.info(f"Generating waveform: samples_per_pixel={self.samples_per_pixel}, "
                    f"input_channels={self.channels}, output_channels={self.output_channels}, "
                    f"bands={len(self.bands) if self.bands else 1}")

    def _clamp_sample(self, sample: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """Clamp sample values to valid range."""
        return np.clip(sample, self.min_sample, self.max_sample)

    def _process_band(self, band_name: str, input_buffer: np.ndarray) -> None:
        """Process audio samples for a specific band."""
        num_samples = len(input_buffer)
        num_points = num_samples // self.samples_per_pixel

        if num_points == 0:
            return

        # Trim and reshape for processing
        trimmed = input_buffer[:num_points * self.samples_per_pixel]
        reshaped = trimmed.reshape(num_points, self.samples_per_pixel, self.channels)

        # Apply band-specific normalization to compensate for filter effects
        # These are approximate scaling factors based on typical filter responses
        normalization_factor = 1.0
        # if band_name == "low":
        #     normalization_factor = 1.0  # Boost low frequencies
        # elif band_name == "low_mid":
        #     normalization_factor = 1.0  # Significant boost for narrow band
        # elif band_name == "mid":
        #     normalization_factor = 1.0  # Moderate boost
        # elif band_name == "high":
        #     normalization_factor = 1.0  # Boost to compensate for narrow band
        # elif band_name == "ultra":
        #     normalization_factor = 1.0  # Significant boost for very high frequencies

        # Apply normalization
        reshaped = reshaped * normalization_factor

        if self.output_channels == 1:
            # Mix channels into mono
            mixed = reshaped.mean(axis=2, dtype=np.int32)  # Avoid overflow
            clamped = self._clamp_sample(mixed)
            min_vals = clamped.min(axis=1)
            max_vals = clamped.max(axis=1)
            for i in range(num_points):
                self.buffer.append_samples(band_name, int(min_vals[i]), int(max_vals[i]))
        else:
            # Process each channel separately
            clamped = self._clamp_sample(reshaped)
            min_vals = clamped.min(axis=1)
            max_vals = clamped.max(axis=1)
            for i in range(num_points):
                for ch in range(self.channels):
                    self.buffer.append_samples(band_name, int(min_vals[i, ch]), int(max_vals[i, ch]))

        # Process remainder recursively
        remainder = input_buffer[num_points * self.samples_per_pixel:]
        if len(remainder) > 0:
            self._process_band(band_name, remainder)

    def process(self, input_buffer: np.ndarray) -> None:
        """Process audio samples using vectorized operations."""
        if self.band_splitter:
            # Split into bands and process each band
            band_samples = self.band_splitter.process_sample(input_buffer)
            for band_name, samples in band_samples.items():
                self._process_band(band_name, samples)
        else:
            # Process full range
            self._process_band("fullrange", input_buffer)

    def done(self) -> None:
        """Finalize waveform generation."""
        band_info = []
        for band_name in self.buffer.band_names:
            band_info.append(f"{band_name}:{self.buffer.get_size(band_name)}")

        logger.info(f"Generated points: {', '.join(band_info)}")


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


def parse_frequency_bands(bands_arg: List[str]) -> List[FrequencyBand]:
    """
    Parse frequency band arguments into FrequencyBand objects.

    Expected format for each band:
    "name:filter_type:freq:slope[,filter_type:freq:slope,...]"

    Examples:
    - "low:lowpass:500:12"
    - "mid:bandpass:500-5000:24"
    - "high:highpass:5000:12"
    - "custom:lowpass:1000:12,highpass:200:12" (combines multiple filters)
    """
    result = []

    for band_spec in bands_arg:
        parts = band_spec.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid band specification: {band_spec}")

        name = parts[0]
        filter_specs = parts[1].split(",")
        filter_profiles = []

        for filter_spec in filter_specs:
            filter_parts = filter_spec.split(":")
            if len(filter_parts) < 3:
                raise ValueError(f"Invalid filter specification: {filter_spec}")

            filter_type_str = filter_parts[0].upper()
            if not hasattr(FilterType, filter_type_str):
                raise ValueError(f"Invalid filter type: {filter_type_str}")

            filter_type = getattr(FilterType, filter_type_str)

            # Handle frequency range for bandpass/bandstop
            freq_part = filter_parts[1]
            if "-" in freq_part and filter_type in (FilterType.BANDPASS, FilterType.BANDSTOP):
                freq_range = freq_part.split("-")
                if len(freq_range) != 2:
                    raise ValueError(f"Invalid frequency range: {freq_part}")
                cutoff_freq = (float(freq_range[0]), float(freq_range[1]))
            else:
                cutoff_freq = float(freq_part)

            # Get slope in dB/octave
            slope_db_oct = float(filter_parts[2])

            # Optional filter order
            order = None
            if len(filter_parts) > 3:
                order = int(filter_parts[3])

            filter_profiles.append(FilterProfile(
                filter_type=filter_type,
                cutoff_freq=cutoff_freq,
                slope_db_oct=slope_db_oct,
                order=order
            ))

        result.append(FrequencyBand(name=name, filter_profiles=filter_profiles))

    return result


def generate_waveform_data(audio: AudioSegment, scale_factor: ScaleFactor,
                           split_channels: bool, bands: Optional[List[FrequencyBand]] = None) -> WaveformBuffer:
    """Generate waveform data from an audio file."""
    buffer = WaveformBuffer()
    generator = WaveformGenerator(buffer, split_channels, scale_factor, bands)

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

    # Add frequency band options
    parser.add_argument(
        "--bands", "-f", nargs="+",
        help="Frequency bands to split into. Format: 'name:filter_type:freq:slope[,filter_type:freq:slope,...]'"
             " Example: 'low:lowpass:500:12' or 'mid:bandpass:500-5000:24'"
    )

    # Add predefined band presets
    parser.add_argument(
        "--band-preset", choices=["standard", "detailed", "club"],
        help="Use a predefined frequency band preset"
    )

    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress messages")
    parser.add_argument("--version", action="version", version="Waveform Extractor 2.0")

    return parser.parse_args()


def get_band_preset(preset_name: str) -> List[FrequencyBand]:
    """Get a predefined frequency band preset."""
    presets = {
        "standard": [
            "low:lowpass:250:12",
            "mid:bandpass:250-4000:12",
            "high:highpass:4000:12"
        ],
        "detailed": [
            "sub:lowpass:60:24",
            "low:bandpass:60-250:24",
            "lowmid:bandpass:250-500:24",
            "mid:bandpass:500-2000:24",
            "highmid:bandpass:2000-4000:24",
            "high:bandpass:4000-10000:24",
            "ultra:highpass:10000:24"
        ],
        "club": [
            "low:lowpass:250:12",
            "low_mid:highpass:250:12,lowpass:500:12",
            "mid:highpass:250:12,lowpass:1200:12",
            "high:highpass:2000:12,lowpass:3000:12",
        ]
    }

    if preset_name not in presets:
        raise ValueError(f"Unknown band preset: {preset_name}")

    return parse_frequency_bands(presets[preset_name])


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

        # Parse frequency bands
        bands = None
        if args.band_preset:
            bands = get_band_preset(args.band_preset)
        elif args.bands:
            bands = parse_frequency_bands(args.bands)

        # Generate waveform
        buffer = generate_waveform_data(audio, scale_factor, args.split_channels, bands)

        # Save output
        buffer.save_as_json(args.output_file, BitDepth(args.bits))
        return 0

    except (ValueError, IOError, FileNotFoundError) as e:
        logger.error(str(e))
        return 1


if __name__ == "__main__":
    exit(main())
