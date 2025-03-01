"""
Extracts waveform data from audio files with frequency band splitting and saves it as a JSON file.
Samples are stored as base64 encoded strings with configurable compression (gzip by default).
Uses an object-oriented approach with a Waveform class.
"""

import argparse
import json
import logging
import os
import base64
import gzip
import struct
from enum import IntEnum
from typing import List, Tuple, Union, Dict, Optional, Any

import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, sosfilt, sosfilt_zi

# Global band presets
band_presets = {
    "standard": [
        "low:lowpass:250:12",
        "mid:bandpass:250-4000:12",
        "high:highpass:4000:12"
    ],
    "detailed": [
        "sub:lowpass:60:24",
        "low:bandpass:60-250:24",
        "low_mid:bandpass:250-500:24",
        "mid:bandpass:500-2000:24",
        "high_mid:bandpass:2000-4000:24",
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


class SampleFormat(IntEnum):
    """Format for storing samples."""
    BASE64_JSON = 0  # Base64 encoded JSON array
    BASE64_BINARY = 1  # Base64 encoded binary data


class CompressionType(IntEnum):
    """Types of compression for sample data."""
    NONE = 0
    GZIP = 1

    # Reserved for future compression types
    # BZIP2 = 2
    # ZSTD = 3
    # LZ4 = 4

    @classmethod
    def from_string(cls, compression_str: str) -> 'CompressionType':
        """Convert a string to a CompressionType enum value."""
        mapping = {
            "none": cls.NONE,
            "gzip": cls.GZIP,
            # Add future compression types here
        }
        if compression_str.lower() not in mapping:
            raise ValueError(f"Unknown compression type: {compression_str}. "
                             f"Available types: {', '.join(mapping.keys())}")
        return mapping[compression_str.lower()]

    @classmethod
    def to_string(cls, compression_type: 'CompressionType') -> str:
        """Convert a CompressionType enum value to a string."""
        mapping = {
            cls.NONE: "none",
            cls.GZIP: "gzip",
            # Add future compression types here
        }
        return mapping.get(compression_type, "unknown")


class FilterProfile:
    """Defines a filter profile for frequency band splitting."""

    def __init__(
            self,
            filter_type: FilterType,
            cutoff_freq: Union[float, Tuple[float, float]],
            slope_db_oct: float = 12.0,
            order: Optional[int] = None
    ):
        self.filter_type = filter_type
        self.cutoff_freq = cutoff_freq
        self.slope_db_oct = slope_db_oct
        self.order = max(1, int(slope_db_oct / 6.0)) if order is None else order


class FrequencyBand:
    """Defines a frequency band with a name and filtering parameters."""

    def __init__(self, name: str, filter_profiles: List[FilterProfile]):
        self.name = name
        self.filter_profiles = filter_profiles


class Band:
    """Represents a frequency band in a waveform."""

    def __init__(self, name: str, sample_format: SampleFormat,
                 compression: CompressionType = CompressionType.GZIP):
        self.name = name
        self.frequency_range = []  # Will be set later if provided
        self.sample_format = sample_format
        self.compression = compression
        self.samples_buffer = []  # Temporary buffer for collecting samples
        self.samples = ""  # Final encoded string

    def append_samples(self, min_value: int, max_value: int) -> None:
        """Append a min/max pair to the buffer."""
        self.samples_buffer.extend([min_value, max_value])

    def get_size(self) -> int:
        """Return the number of points (pixels) in this band."""
        return len(self.samples_buffer) // 2  # Each point is a min/max pair

    def encode_samples(self, bits: BitDepth) -> None:
        """Encode the samples buffer to a base64 string."""
        # Scale samples based on bit depth
        divisor = 256 if bits == BitDepth.EIGHT else 1
        samples = [int(v) // divisor for v in self.samples_buffer]

        # Prepare data based on format
        if self.sample_format == SampleFormat.BASE64_JSON:
            # Encode as JSON array
            data = json.dumps(samples).encode('utf-8')
        else:  # SampleFormat.BASE64_BINARY
            # Encode as raw binary data
            fmt = 'b' if bits == BitDepth.EIGHT else 'h'  # 'b' for 8-bit, 'h' for 16-bit
            data = struct.pack(f'<{len(samples)}{fmt}', *samples)

        # Apply compression if requested
        if self.compression == CompressionType.GZIP:
            data = gzip.compress(data)

        # Base64 encode the final data
        self.samples = base64.b64encode(data).decode('utf-8')

        # Clear the buffer to free memory
        self.samples_buffer = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert band to dictionary for JSON serialization."""
        compression_str = CompressionType.to_string(self.compression)
        format_str = "base64_json" if self.sample_format == SampleFormat.BASE64_JSON else "base64_binary"

        return {
            "name": self.name,
            "frequency_range": self.frequency_range,
            "compression": compression_str,
            "sample_format": format_str,
            "samples": self.samples
        }


class Waveform:
    """Represents a complete waveform with all its data and metadata."""

    MAX_CHANNELS = 24

    def __init__(self, sample_format: SampleFormat = SampleFormat.BASE64_BINARY,
                 compression: CompressionType = CompressionType.GZIP):
        # Basic properties
        self.version = "1.0.0"
        self.sample_rate = 0
        self.samples_per_pixel = 0
        self.bits = BitDepth.EIGHT
        self.channels = 1
        self.duration = 0.0
        self.type = "fullrange"  # or "multiband"

        # Format and compression settings
        self.sample_format = sample_format
        self.compression = compression

        # Data storage
        self.bands = {}  # Dictionary of Band objects keyed by name

        # Metadata
        self.metadata = {
            "source": "raw",
            "tags": {}
        }

    def set_sample_rate(self, sample_rate: int) -> None:
        """Set the sample rate of the waveform."""
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        self.sample_rate = sample_rate

    def set_samples_per_pixel(self, samples_per_pixel: int) -> None:
        """Set the number of samples per pixel for the waveform."""
        if samples_per_pixel < 2:
            raise ValueError("Samples per pixel must be at least 2")
        self.samples_per_pixel = samples_per_pixel

    def set_channels(self, channels: int) -> None:
        """Set the number of channels in the waveform."""
        if not 1 <= channels <= self.MAX_CHANNELS:
            raise ValueError(f"Channels must be between 1 and {self.MAX_CHANNELS}")
        self.channels = channels

    def set_duration(self, total_samples: int) -> None:
        """Set duration based on total samples and sample rate."""
        if self.sample_rate > 0:
            self.duration = total_samples / self.sample_rate

    def set_bands(self, band_names: List[str]) -> None:
        """Initialize the bands for this waveform."""
        self.type = "multiband" if band_names and len(band_names) > 1 else "fullrange"
        self.bands = {}

        # Create a Band object for each band
        for name in band_names:
            self.bands[name] = Band(name, self.sample_format, self.compression)

    def append_samples(self, band_name: str, min_value: int, max_value: int) -> None:
        """Append a min/max sample pair to the specified band."""
        if band_name in self.bands:
            self.bands[band_name].append_samples(min_value, max_value)

    def get_size(self) -> int:
        """Return the number of points (pixels) in the waveform."""
        if not self.bands:
            return 0
        return next(iter(self.bands.values())).get_size()

    def encode_samples(self) -> None:
        """Encode all band samples to their final format."""
        for band in self.bands.values():
            band.encode_samples(self.bits)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the waveform to a dictionary for JSON serialization."""
        compression_str = CompressionType.to_string(self.compression)

        result = {
            "version": self.version,
            "channels": self.channels,
            "sample_rate": self.sample_rate,
            "bits_per_sample": self.bits,
            "duration": self.duration,
            "samples_per_pixel": self.samples_per_pixel,
            "type": self.type,
            "data": {},
            "metadata": {
                "compression": compression_str,
                "source": self.metadata["source"],
                "tags": self.metadata["tags"]
            }
        }

        # Add band data based on waveform type
        if self.type == "multiband":
            result["data"]["multiband"] = {
                "bands": [band.to_dict() for band in self.bands.values()]
            }
        else:
            # There should be only one band for fullrange
            fullrange_band = self.bands.get("fullrange")
            if fullrange_band:
                result["data"]["fullrange"] = {
                    "compression": CompressionType.to_string(self.compression),
                    "sample_format": "base64_json" if self.sample_format == SampleFormat.BASE64_JSON else "base64_binary",
                    "samples": fullrange_band.samples
                }

        return result

    def save_as_json(self, filename: str, bits: BitDepth = BitDepth.SIXTEEN) -> None:
        """Save waveform data as a formatted JSON file."""
        self.bits = bits

        try:
            # Encode all samples before saving
            self.encode_samples()

            # Convert to dictionary and save
            data = self.to_dict()
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
                f.write("\n")
            logger.info(f"Waveform data saved to '{filename}'")
        except Exception as e:
            raise IOError(f"Failed to save JSON file: {e}")


class ScaleFactor:
    """Abstract base class for determining samples per pixel scaling."""

    def get_samples_per_pixel(self, sample_rate: int) -> int:
        raise NotImplementedError("Subclasses must implement this method")


class SamplesPerPixelScaleFactor(ScaleFactor):
    def __init__(self, samples_per_pixel: int):
        if samples_per_pixel < 2:
            raise ValueError("Samples per pixel must be at least 2")
        self.samples_per_pixel = samples_per_pixel

    def get_samples_per_pixel(self, sample_rate: int) -> int:
        return self.samples_per_pixel


class PixelsPerSecondScaleFactor(ScaleFactor):
    def __init__(self, pixels_per_second: int):
        if pixels_per_second <= 0:
            raise ValueError("Pixels per second must be positive")
        self.pixels_per_second = pixels_per_second

    def get_samples_per_pixel(self, sample_rate: int) -> int:
        return max(2, sample_rate // self.pixels_per_second)


class DurationScaleFactor(ScaleFactor):
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
    def __init__(self, sample_rate: int, bands: List[FrequencyBand]):
        self.sample_rate = sample_rate
        self.bands = bands
        self.filters_cache = {}

    def _create_filter(self, profile: FilterProfile) -> Tuple:
        nyquist = 0.5 * self.sample_rate
        if isinstance(profile.cutoff_freq, tuple):
            normalized_cutoff = (profile.cutoff_freq[0] / nyquist, profile.cutoff_freq[1] / nyquist)
        else:
            normalized_cutoff = profile.cutoff_freq / nyquist
        btype = FilterType(profile.filter_type).name.lower()
        sos = butter(profile.order, normalized_cutoff, btype=btype, output='sos')
        zi = sosfilt_zi(sos)
        if sos.ndim == 2 and sos.shape[0] == 1:
            zi = zi.reshape(1, 2)
        return sos, zi

    def process_sample(self, samples: np.ndarray) -> Dict[str, np.ndarray]:
        results = {}
        for band in self.bands:
            filtered_samples = samples.copy()
            for profile in band.filter_profiles:
                filter_key = (band.name, profile.filter_type, str(profile.cutoff_freq), profile.order)
                if filter_key not in self.filters_cache:
                    self.filters_cache[filter_key] = self._create_filter(profile)
                sos, zi = self.filters_cache[filter_key]
                for channel in range(filtered_samples.shape[1]):
                    channel_zi = zi.copy()
                    filtered_samples[:, channel], _ = sosfilt(sos, filtered_samples[:, channel], zi=channel_zi)
            results[band.name] = filtered_samples
        return results


class WaveformGenerator:
    def __init__(self, waveform: Waveform, split_channels: bool, scale_factor: ScaleFactor,
                 bands: Optional[List[FrequencyBand]] = None):
        self.waveform = waveform
        self.split_channels = split_channels
        self.scale_factor = scale_factor
        self.bands = bands
        self.band_splitter = None
        self.channels = 0
        self.output_channels = 0
        self.samples_per_pixel = 0
        self.min_sample = -32768
        self.max_sample = 32767

    def init(self, sample_rate: int, channels: int) -> None:
        if not 1 <= channels <= Waveform.MAX_CHANNELS:
            raise ValueError(f"Unsupported number of channels: {channels}")
        self.channels = channels
        self.samples_per_pixel = self.scale_factor.get_samples_per_pixel(sample_rate)
        self.output_channels = channels if self.split_channels else 1

        self.waveform.set_samples_per_pixel(self.samples_per_pixel)
        self.waveform.set_sample_rate(sample_rate)
        self.waveform.set_channels(self.output_channels)

        if self.bands:
            band_names = [band.name for band in self.bands]
            self.waveform.set_bands(band_names)
            self.band_splitter = BandSplitter(sample_rate, self.bands)
        else:
            self.waveform.set_bands(["fullrange"])

        logger.info(f"Generating waveform: samples_per_pixel={self.samples_per_pixel}, "
                    f"input_channels={self.channels}, output_channels={self.output_channels}, "
                    f"bands={len(self.bands) if self.bands else 1}")

    def _clamp_sample(self, sample: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        return np.clip(sample, self.min_sample, self.max_sample)

    def _process_band(self, band_name: str, input_buffer: np.ndarray) -> None:
        num_samples = len(input_buffer)
        num_points = num_samples // self.samples_per_pixel
        if num_points == 0:
            return
        trimmed = input_buffer[:num_points * self.samples_per_pixel]
        reshaped = trimmed.reshape(num_points, self.samples_per_pixel, self.channels)
        normalization_factor = 1.0
        reshaped = reshaped * normalization_factor
        if self.output_channels == 1:
            mixed = reshaped.mean(axis=2, dtype=np.int32)
            clamped = self._clamp_sample(mixed)
            min_vals = clamped.min(axis=1)
            max_vals = clamped.max(axis=1)
            for i in range(num_points):
                self.waveform.append_samples(band_name, int(min_vals[i]), int(max_vals[i]))
        else:
            clamped = self._clamp_sample(reshaped)
            min_vals = clamped.min(axis=1)
            max_vals = clamped.max(axis=1)
            for i in range(num_points):
                for ch in range(self.channels):
                    self.waveform.append_samples(band_name, int(min_vals[i, ch]), int(max_vals[i, ch]))
        remainder = input_buffer[num_points * self.samples_per_pixel:]
        if len(remainder) > 0:
            self._process_band(band_name, remainder)

    def process(self, input_buffer: np.ndarray) -> None:
        self.waveform.set_duration(len(input_buffer))
        if self.band_splitter:
            band_samples = self.band_splitter.process_sample(input_buffer)
            for band_name, samples in band_samples.items():
                self._process_band(band_name, samples)
        else:
            self._process_band("fullrange", input_buffer)

    def done(self) -> None:
        band_info = []
        for band_name, band in self.waveform.bands.items():
            band_info.append(f"{band_name}:{band.get_size()}")
        logger.info(f"Generated points: {', '.join(band_info)}")


def read_audio_file(filename: str, show_info: bool = True) -> AudioSegment:
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
            freq_part = filter_parts[1]
            if "-" in freq_part and filter_type in (FilterType.BANDPASS, FilterType.BANDSTOP):
                freq_range = freq_part.split("-")
                if len(freq_range) != 2:
                    raise ValueError(f"Invalid frequency range: {freq_part}")
                cutoff_freq = (float(freq_range[0]), float(freq_range[1]))
            else:
                cutoff_freq = float(freq_part)
            slope_db_oct = float(filter_parts[2])
            order = int(filter_parts[3]) if len(filter_parts) > 3 else None
            filter_profiles.append(FilterProfile(filter_type, cutoff_freq, slope_db_oct, order))
        result.append(FrequencyBand(name, filter_profiles))
    return result


def generate_waveform_data(audio: AudioSegment, scale_factor: ScaleFactor, split_channels: bool,
                           bands: Optional[List[FrequencyBand]] = None,
                           sample_format: SampleFormat = SampleFormat.BASE64_BINARY,
                           compression: CompressionType = CompressionType.GZIP) -> Waveform:
    # Create a new Waveform object
    waveform = Waveform(sample_format, compression)

    # Set up the generator to process audio data
    generator = WaveformGenerator(waveform, split_channels, scale_factor, bands)
    generator.init(audio.frame_rate, audio.channels)

    # Convert audio data to numpy array for processing
    samples = np.array(audio.get_array_of_samples())
    if audio.sample_width != 2:
        max_value = 2 ** (8 * audio.sample_width - 1) - 1
        samples = (samples / max_value * 32767).astype(np.int16)

    # Reshape samples according to channel count
    samples = samples.reshape(-1, audio.channels) if audio.channels > 1 else samples.reshape(-1, 1)

    # Process audio data into waveform
    generator.process(samples)
    generator.done()

    return waveform


def create_scale_factor(args: argparse.Namespace) -> ScaleFactor:
    if args.pixels_per_second:
        return PixelsPerSecondScaleFactor(args.pixels_per_second)
    elif args.end is not None:
        return DurationScaleFactor(args.start, args.end, args.width)
    return SamplesPerPixelScaleFactor(args.zoom)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract waveform data from audio files and save as JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file", help="Input audio file (e.g., .mp3, .wav)")
    parser.add_argument("output_file", help="Output JSON file")
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument("--zoom", "-z", type=int, default=256, help="Samples per pixel")
    scale_group.add_argument("--pixels-per-second", type=int, help="Pixels per second")
    scale_group.add_argument("--end", "-e", type=float, help="End time in seconds")
    parser.add_argument("--start", "-s", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--width", "-w", type=int, default=800, help="Width in pixels")
    parser.add_argument("--split-channels", action="store_true", help="Keep channels separate")
    parser.add_argument("--bits", "-b", type=int, choices=[BitDepth.EIGHT, BitDepth.SIXTEEN], default=BitDepth.EIGHT,
                        help="Output bit depth")
    parser.add_argument("--bands", "-f", nargs="+",
                        help="Frequency bands: 'name:filter_type:freq:slope[,...]'")
    parser.add_argument("--band-preset", choices=band_presets.keys(), help="Use a predefined band preset")
    parser.add_argument("--sample-format", type=str, choices=["json", "binary"], default="binary",
                        help="Sample encoding format (json or binary)")
    parser.add_argument("--compression-type", "-c", type=str, default="gzip",
                        choices=["none", "gzip"],
                        help="Compression type for sample data")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress messages")
    parser.add_argument("--version", action="version", version="Waveform Extractor 2.0")
    return parser.parse_args()


def get_band_preset(preset_name: str) -> List[FrequencyBand]:
    if preset_name not in band_presets:
        raise ValueError(f"Unknown band preset: {preset_name}")
    return parse_frequency_bands(band_presets[preset_name])


def main() -> int:
    args = parse_arguments()
    if args.quiet:
        logger.setLevel(logging.WARNING)
    try:
        # Read the audio file
        audio = read_audio_file(args.input_file)

        # Create scale factor based on command line arguments
        scale_factor = create_scale_factor(args)

        # Determine which frequency bands to use
        bands = get_band_preset(args.band_preset) if args.band_preset else \
            parse_frequency_bands(args.bands) if args.bands else None

        # Determine sample format and compression from arguments
        sample_format = SampleFormat.BASE64_JSON if args.sample_format == "json" else SampleFormat.BASE64_BINARY
        compression = CompressionType.from_string(args.compression_type)

        # Generate the waveform data
        waveform = generate_waveform_data(
            audio, scale_factor, args.split_channels, bands, sample_format, compression
        )

        # Save the waveform to JSON
        waveform.save_as_json(args.output_file, BitDepth(args.bits))

        return 0
    except (ValueError, IOError, FileNotFoundError) as e:
        logger.error(str(e))
        return 1


if __name__ == "__main__":
    exit(main())
