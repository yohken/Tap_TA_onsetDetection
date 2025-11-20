# MFA Onset Detection Pipeline

This document provides detailed documentation for the MFA-based onset detection pipeline.

## Overview

The MFA onset detection pipeline provides a comprehensive solution for detecting /t/ burst onsets in Japanese speech recordings (e.g., "ta" syllables). It combines two complementary detection methods:

1. **MFA-based detection**: Uses Montreal Forced Aligner (MFA) TextGrid annotations to identify phone segments and detect burst onsets using high-frequency RMS envelope analysis
2. **Hilbert-based detection**: Uses the Fujii method with Hilbert envelope for onset detection without requiring phonetic annotations

The pipeline processes multiple WAV files, generates comparison visualizations, and exports results in multiple formats with full parameter documentation for reproducibility.

## Features

- ✅ **Multi-file processing**: Process multiple WAV files in batch mode
- ✅ **Automatic MFA alignment**: Optional automatic generation of TextGrid files via MFA
- ✅ **Dual detection methods**: Compare MFA and Hilbert-based results side-by-side
- ✅ **Comprehensive visualization**: Plot waveform, envelopes, and onset markers
- ✅ **Multiple export formats**: CSV and JSON with full parameter details
- ✅ **Error handling**: Graceful error handling with detailed logging
- ✅ **Reproducibility**: All parameters logged with timestamps for reproducibility
- ✅ **Customizable parameters**: Override all default detection parameters

## Installation

### Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### Optional: Install MFA (for automatic TextGrid generation)

If you want to use automatic MFA alignment, install Montreal Forced Aligner:

```bash
# Via conda (recommended)
conda install -c conda-forge montreal-forced-aligner

# Or follow instructions at:
# https://montreal-forced-aligner.readthedocs.io/
```

Download required acoustic model and dictionary:

```bash
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

## Usage

### Basic Usage

```bash
# Process a single file (Hilbert detection only)
python mfa_onset_pipeline.py speech.wav

# Process multiple files
python mfa_onset_pipeline.py file1.wav file2.wav file3.wav
```

### With MFA Alignment

```bash
# Automatically run MFA alignment before detection
python mfa_onset_pipeline.py speech.wav --run-mfa

# Specify text content for alignment
python mfa_onset_pipeline.py speech.wav --run-mfa --text-content "ta ta ta"

# Use different MFA models
python mfa_onset_pipeline.py speech.wav \
    --run-mfa \
    --mfa-model japanese_mfa \
    --mfa-dictionary japanese_mfa
```

### Custom Parameters

```bash
# Customize MFA detection parameters
python mfa_onset_pipeline.py speech.wav \
    --mfa-high-freq 2500 \
    --mfa-frame-length 10 \
    --mfa-hop-length 2 \
    --mfa-threshold 3.0

# Customize Hilbert detection parameters
python mfa_onset_pipeline.py speech.wav \
    --hilbert-sr 44100 \
    --hilbert-hp-cutoff 600 \
    --hilbert-threshold 0.15 \
    --hilbert-lookback 50 \
    --hilbert-min-interval 100

# Specify output directory
python mfa_onset_pipeline.py speech.wav -o my_results/

# Verbose output for debugging
python mfa_onset_pipeline.py speech.wav -v
```

### Complete Example

```bash
# Process multiple files with MFA alignment and custom parameters
python mfa_onset_pipeline.py ta1.wav ta2.wav ta3.wav \
    --run-mfa \
    --text-content "ta" \
    --mfa-high-freq 2500 \
    --hilbert-hp-cutoff 600 \
    -o results_custom/ \
    -v
```

## Parameters

### MFA Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mfa-high-freq` | 2000.0 | High-frequency minimum for RMS envelope (Hz) |
| `--mfa-frame-length` | 5.0 | Frame length for RMS computation (ms) |
| `--mfa-hop-length` | 1.0 | Hop length for RMS computation (ms) |
| `--mfa-threshold` | 2.0 | Differential threshold (std multiplier) |

### Hilbert Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hilbert-sr` | 48000 | Target sampling rate (Hz) |
| `--hilbert-hp-cutoff` | 500.0 | High-pass filter cutoff (Hz) |
| `--hilbert-threshold` | 0.1 | Onset threshold ratio (0.1 = 10% of peak) |
| `--hilbert-lookback` | 74 | Lookback points (~2ms at 48kHz) |
| `--hilbert-min-interval` | 50.0 | Minimum interval between onsets (ms) |

### MFA Alignment Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--run-mfa` | False | Run MFA alignment to generate TextGrid |
| `--text-content` | "ta" | Text content for MFA alignment |
| `--mfa-model` | english_us_arpa | MFA acoustic model name |
| `--mfa-dictionary` | english_us_arpa | MFA dictionary name |

### General Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-o, --output-dir` | ./mfa_onset_results | Output directory for results |
| `-v, --verbose` | False | Enable verbose logging |

## Output Files

For each input WAV file `example.wav`, the pipeline generates:

### 1. Comparison Plot (`example_comparison.png`)

A three-panel plot showing:
- **Top panel**: Waveform with onset markers (red=MFA, blue=Hilbert, green=Hilbert peaks)
- **Middle panel**: MFA high-frequency RMS envelope with onset markers
- **Bottom panel**: Hilbert envelope with onset markers

### 2. CSV Results (`example_results.csv`)

Tab-separated CSV file with columns:
- `method`: Detection method (MFA or Hilbert)
- `onset_time_s`: Onset time in seconds
- `peak_time_s`: Peak time in seconds (for Hilbert only)
- `file`: Source filename

Example:
```csv
method,onset_time_s,peak_time_s,file
MFA,0.35,,example.wav
MFA,0.85,,example.wav
Hilbert,0.30,0.31,example.wav
Hilbert,0.80,0.81,example.wav
```

### 3. JSON Results (`example_results.json`)

Complete results with full parameter details:
```json
{
  "file": "example.wav",
  "timestamp": "2025-11-20T04:40:36.902442",
  "mfa_detection": {
    "onsets": [0.35, 0.85],
    "count": 2,
    "parameters": {
      "high_freq_min": 2000.0,
      "frame_length_ms": 5.0,
      "hop_length_ms": 1.0,
      "diff_threshold_std": 2.0
    },
    "error": null
  },
  "hilbert_detection": {
    "onsets": [0.30, 0.80],
    "peaks": [0.31, 0.81],
    "count": 2,
    "parameters": {
      "target_sr": 48000,
      "hp_cutoff": 500.0,
      "threshold_ratio": 0.1,
      "lookback_points": 74,
      "min_interval_ms": 50.0
    },
    "error": null
  }
}
```

### 4. Processing Summary (`processing_summary.txt`)

Text report with:
- Processing timestamp
- All parameter values used
- Per-file results summary
- Paths to output files

### 5. Processing Log (`pipeline_YYYYMMDD_HHMMSS.log`)

Detailed log of all processing steps, warnings, and errors.

## Detection Methods

### MFA-based Detection

The MFA method detects /t/ burst onsets using:

1. **TextGrid annotation**: Uses MFA phone-level segmentation to identify /t/ segments
2. **High-frequency RMS**: Computes RMS envelope for frequencies ≥2000 Hz (captures burst energy)
3. **Onset detection**: Within each /t/ segment, detects the earliest onset in the high-frequency envelope
4. **Frame-based**: Uses 5ms frames with 1ms hop for temporal resolution

**Advantages**:
- Phonetically informed (knows where /t/ segments are)
- Focuses on high-frequency burst energy specific to /t/
- Good for multi-syllable utterances

**Requirements**:
- TextGrid file with phone-level annotations
- Accurate MFA alignment

### Hilbert-based Detection (Fujii Method)

The Hilbert method uses the Fujii onset detection algorithm:

1. **High-pass filtering**: Removes low-frequency noise (default: 500 Hz)
2. **Hilbert envelope**: Computes instantaneous amplitude envelope
3. **Peak detection**: Finds local maxima in envelope
4. **Onset search**: For each peak, searches backward to find 10% threshold crossing
5. **Linear interpolation**: Sub-sample precision for onset timing

**Advantages**:
- No annotations required
- Sub-sample timing precision
- Adaptive per-event threshold
- Established method in literature

**Best for**:
- Percussive sounds (taps, clicks)
- When TextGrid is unavailable
- Comparative validation

## Comparison and Analysis

The pipeline facilitates comparison between the two methods:

### Visual Comparison
The comparison plot shows both detection results simultaneously, making it easy to:
- Identify agreement/disagreement between methods
- Assess timing differences
- Evaluate detection quality visually

### Quantitative Comparison
The CSV export enables:
- Computing timing differences: `hilbert_onset - mfa_onset`
- Calculating detection precision and recall
- Statistical analysis of onset timing

### Example Analysis Workflow

```python
import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('example_results.csv')

# Separate methods
mfa = df[df['method'] == 'MFA']['onset_time_s'].values
hilbert = df[df['method'] == 'Hilbert']['onset_time_s'].values

# Match closest onsets (within 50ms)
threshold = 0.05  # 50ms
matches = []
for m_onset in mfa:
    diffs = np.abs(hilbert - m_onset)
    if np.min(diffs) < threshold:
        h_onset = hilbert[np.argmin(diffs)]
        matches.append((m_onset, h_onset, h_onset - m_onset))

# Analyze timing differences
if matches:
    timing_diffs = [m[2] for m in matches]
    print(f"Mean timing difference: {np.mean(timing_diffs)*1000:.2f} ms")
    print(f"Std timing difference: {np.std(timing_diffs)*1000:.2f} ms")
```

## Error Handling

The pipeline includes robust error handling:

### MFA Errors
- **MFA not installed**: Logs warning, continues with Hilbert detection only
- **MFA alignment fails**: Logs error, skips MFA detection for that file
- **TextGrid generation fails**: Logs warning, continues processing other files

### File Errors
- **WAV file not found**: Logs error, skips to next file
- **Invalid WAV format**: Logs error, continues processing
- **Corrupt audio**: Logs error with details

### Detection Errors
- **No onsets detected**: Logs warning, exports empty results
- **TextGrid parsing error**: Logs error, falls back to Hilbert only

All errors are logged to:
1. Console output (errors and warnings)
2. Log file (all details)
3. Processing summary (error count)

## Best Practices

### For /t/ Burst Detection

1. **MFA alignment**:
   - Use appropriate MFA model for your language
   - Ensure text content matches audio
   - Check TextGrid quality before processing

2. **Parameter tuning**:
   - Start with defaults
   - Adjust `--mfa-high-freq` if bursts are very high/low frequency
   - Adjust `--hilbert-hp-cutoff` based on background noise
   - Increase `--mfa-threshold` if detecting too many false positives

3. **Validation**:
   - Always check comparison plots visually
   - Listen to audio at detected onset times
   - Compare with manual annotations if available

### For Reproducibility

1. **Save parameters**: JSON export includes all parameters used
2. **Version control**: Log git commit hash in summary
3. **Document**: Add notes about data source and processing goals
4. **Archive**: Keep all output files together

## Troubleshooting

### Common Issues

**Problem**: No MFA onsets detected
- **Solution**: Check that TextGrid file was generated correctly
- **Solution**: Verify phone label matches (default: "t")
- **Solution**: Try adjusting `--mfa-threshold` parameter

**Problem**: Too many false positive Hilbert onsets
- **Solution**: Increase `--hilbert-threshold` (e.g., 0.15 or 0.2)
- **Solution**: Increase `--hilbert-min-interval` to avoid rapid duplicates
- **Solution**: Adjust `--hilbert-hp-cutoff` to filter noise

**Problem**: MFA alignment fails
- **Solution**: Check that MFA is installed correctly
- **Solution**: Verify model and dictionary are downloaded
- **Solution**: Check that text content makes sense for audio
- **Solution**: Try with simpler text (e.g., single word)

**Problem**: Timing mismatch between methods
- **Expected**: Methods use different algorithms and may differ by 5-20ms
- **Check**: Large differences (>50ms) may indicate parameter issues
- **Solution**: Visual inspection of plots to diagnose

## Python API

The pipeline can also be used programmatically:

```python
from pathlib import Path
from mfa_onset_pipeline import MFAOnsetPipeline, MFAParameters, HilbertParameters

# Custom parameters
mfa_params = MFAParameters(
    high_freq_min=2500.0,
    diff_threshold_std=3.0
)

hilbert_params = HilbertParameters(
    hp_cutoff=600.0,
    threshold_ratio=0.15
)

# Create pipeline
pipeline = MFAOnsetPipeline(
    mfa_params=mfa_params,
    hilbert_params=hilbert_params,
    output_dir=Path('./my_results')
)

# Process files
wav_files = [Path('file1.wav'), Path('file2.wav')]
results = pipeline.process_multiple_files(wav_files, run_mfa=False)

# Access results
for result in results:
    print(f"File: {result['file']}")
    print(f"Hilbert onsets: {result['hilbert_result'].onset_times}")
```

## References

- Montreal Forced Aligner: https://montreal-forced-aligner.readthedocs.io/
- Fujii, S., et al. (2011). "The effect of tempo on timing accuracy in sensorimotor synchronization"
- For MFA citation: McAuliffe, M., et al. (2017). "Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi"

## Support

For issues or questions:
1. Check this documentation
2. Review the processing log for error details
3. Try with verbose mode (`-v`) for more information
4. Open an issue on GitHub with log file and example data

## License

See repository license.
