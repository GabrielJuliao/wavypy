# WavyPy

Extract waveform data from audio files with ease and precision.

WavyPy is a Python tool designed to process audio files (e.g., MP3, WAV) and extract waveform data, saving it as a
structured JSON file. Whether you're visualizing audio for a web app, analyzing sound patterns, or debugging audio
processing pipelines, WavyPy offers a fast, flexible, and reliable solution.

---

## Features

- **High Performance**: Utilizes NumPy for optimized waveform processing.
- **Flexible Scaling**: Choose between samples per pixel, pixels per second, or duration-based scaling.
- **Multi-Channel Support**: Process stereo or mono audio with options to split or mix channels.
- **Readable Output**: Saves waveform data in a well-structured JSON format.
- **Command-Line Interface**: Simple CLI with detailed help and options.
- **Extensible**: Modular design for easy customization and integration.

---

## Installation

### Prerequisites

- Python 3.11+
- pip (Python package manager)

### Install Dependencies

Clone the repository and install the required packages:

```bash
git clone https://github.com/GabrielJuliao/wavypy.git
cd wavypy
pip install -r requirements.txt
```

---

## Usage

Run the script from the command line with an input audio file and output JSON file:

```bash
python waveform_extractor.py input.mp3 output.json
```

### Options

| Flag                     | Description                                 | Default | Example                       |
|--------------------------|---------------------------------------------|---------|-------------------------------|
| `-z, --zoom`             | Samples per pixel                           | 256     | `--zoom 512`                  |
| `--pixels-per-second`    | Pixels per second                           | -       | `--pixels-per-second 100`     |
| `-s, --start`            | Start time (seconds)                        | 0.0     | `--start 2.5`                 |
| `-e, --end`              | End time (seconds)                          | -       | `--end 10.0`                  |
| `-w, --width`            | Width in pixels                             | 800     | `--width 1000`                |
| `--split-channels`       | Keep channels separate (vs. mixing to mono) | False   | `--split-channels`            |
| `-b, --bits`             | Output bit depth (8 or 16)                  | 8       | `--bits 16`                   |
| `-f, --bands`            | Frequency bands                             | -       | `--bands bass:lowpass:200:12` |
| `--band-preset`          | Use a predefined band preset                | -       | `--band-preset detailed`         |
| `--sample-format`        | Sample encoding format (json or binary)     | binary  | `--sample-format json`        |
| `-c, --compression-type` | Compression type for sample data            | gzip    | `--compression-type none`     |
| `-q, --quiet`            | Suppress progress messages                  | False   | `--quiet`                     |
| `--version`              | Show version                                | -       | `--version`                   |

### Examples

#### 1. Basic Extraction (Mono, 8-bit)

```bash
python waveform_extractor.py song.mp3 waveform.json
```

#### 2. High-Resolution Extraction

```bash
python waveform_extractor.py podcast.wav output.json --zoom 512
```

#### 3. Specific Duration with Split Channels

```bash
python waveform_extractor.py stereo.wav result.json --start 0 --end 10 --width 1000 --split-channels
```

#### 4. Pixels per Second

```bash
python waveform_extractor.py audio.mp3 data.json --pixels-per-second 50
```

#### 5. Frequency Band Processing

```bash
python waveform_extractor.py music.wav bands.json --bands bass:lowpass:200:12 mid:bandpass:1000:6
```

#### 6. Using a Band Preset

```bash
python waveform_extractor.py speech.wav preset.json --band-preset detailed
```

---

## Output Format

The tool generates a JSON file with structured waveform data. Example (`output.json`):

```json
{
  "version": "1.0.0",
  "channels": 1,
  "sample_rate": 44100,
  "bits_per_sample": 16,
  "duration": 0.0,
  "samples_per_pixel": 256,
  "type": "fullrange",
  "data": {
    "fullrange": {
      "compression": "none",
      "sample_format": "base64_json",
      "samples": ""
    },
    "multiband": {
      "bands": [
        {
          "name": "low",
          "frequency_range": [
            20,
            200
          ],
          "compression": "gzip",
          "sample_format": "base64_binary",
          "samples": ""
        }
      ]
    }
  },
  "metadata": {
    "compression": "none",
    "source": "raw",
    "tags": {}
  }
}
```

### Output Fields

- **`version`**: Metadata version for compatibility.
- **`channels`**: Number of audio channels.
- **`sample_rate`**: Audio sample rate (Hz).
- **`bits_per_sample`**: Bit depth per sample.
- **`duration`**: Duration of the audio (seconds).
- **`samples_per_pixel`**: Number of samples per visualized pixel.
- **`type`**: Waveform type (`fullrange` or `multiband`).
- **`data`**: Contains waveform data, supporting both full-range and multi-band formats.
- **`metadata`**: Additional metadata such as compression and source type.

---

## How It Works

1. **Audio Loading**: Reads audio files using `pydub`.
2. **Waveform Processing**: Computes min/max sample values per pixel using optimized NumPy operations.
3. **Scaling**: Adjusts resolution based on user-defined scaling (e.g., samples per pixel).
4. **Frequency Band Filtering**: Applies user-defined or preset frequency bands.
5. **Output**: Saves results as a formatted JSON file.

---

## Contributing

We welcome contributions! To get started:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add amazing feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/amazing-feature
   ```
5. Open a Pull Request.

Please include tests and update documentation as needed.

---

## License

This project is licensed under the GPL License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with [NumPy](https://numpy.org/) and [pydub](https://github.com/jiaaro/pydub).
- Inspired by audio visualization tools everywhere.

---