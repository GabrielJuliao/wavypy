# Waveform Extractor

Extract waveform data from audio files with ease and precision.

Waveform Extractor is a Python tool that processes audio files (e.g., MP3, WAV) and extracts waveform data, saving it as
a structured JSON file. Whether you're visualizing audio for a web app, analyzing sound patterns, or debugging audio
processing pipelines, this tool provides a fast, flexible, and reliable solution.

---

## Features

- **High Performance**: Uses NumPy for optimized waveform processing.
- **Flexible Scaling**: Choose between samples per pixel, pixels per second, or duration-based scaling.
- **Multi-Channel Support**: Process stereo or mono audio, with options to split or mix channels.
- **Readable Output**: Saves waveform data in a well-structured JSON format.
- **Command-Line Interface**: Simple CLI with detailed help and options.
- **Extensible**: Modular design for easy customization or integration.

---

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Install Dependencies

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/waveform-extractor.git
cd waveform-extractor
pip install -r requirements.txt
```

#### `requirements.txt`

```txt
numpy>=1.21.0
pydub>=0.25.1
```

---

## Usage

Run the script from the command line with an input audio file and output JSON file:

```bash
python waveform_extractor.py input.mp3 output.json
```

### Options

| Flag                  | Description                                 | Default | Example                   |
|-----------------------|---------------------------------------------|---------|---------------------------|
| `-z, --zoom`          | Samples per pixel                           | 256     | `--zoom 512`              |
| `--pixels-per-second` | Pixels per second                           | -       | `--pixels-per-second 100` |
| `-s, --start`         | Start time (seconds)                        | 0.0     | `--start 2.5`             |
| `-e, --end`           | End time (seconds)                          | -       | `--end 10.0`              |
| `-w, --width`         | Width in pixels (with `--end`)              | 800     | `--width 1000`            |
| `--split-channels`    | Keep channels separate (vs. mixing to mono) | False   | `--split-channels`        |
| `-b, --bits`          | Output bit depth (8 or 16)                  | 16      | `--bits 8`                |
| `-q, --quiet`         | Suppress progress messages                  | False   | `--quiet`                 |
| `--version`           | Show version                                | -       | `--version`               |

### Examples

#### 1. Basic Extraction (Mono, 16-bit)

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

---

## Output Format

The tool generates a JSON file with structured waveform data. Example (`output.json`):

```json
{
  "version": 2,
  "channels": 1,
  "sample_rate": 44100,
  "samples_per_pixel": 256,
  "bits": 16,
  "length": 2,
  "data": [
    -32768,
    32767,
    -12345,
    12345
  ]
}
```

### Output Fields

- **`version`**: Format version (currently 2).
- **`channels`**: Number of audio channels.
- **`sample_rate`**: Audio sample rate (Hz).
- **`samples_per_pixel`**: Samples per data point.
- **`bits`**: Bit depth of output values.
- **`length`**: Number of data points per channel.
- **`data`**: Array of min/max pairs (flattened).

---

## How It Works

1. **Audio Loading**: Reads audio files using `pydub`.
2. **Waveform Processing**: Computes min/max sample values per pixel using optimized NumPy operations.
3. **Scaling**: Adjusts resolution based on user-defined scaling (e.g., samples per pixel).
4. **Output**: Saves results as a formatted JSON file.

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with [NumPy](https://numpy.org/) and [pydub](https://github.com/jiaaro/pydub).
- Inspired by audio visualization tools everywhere.

---