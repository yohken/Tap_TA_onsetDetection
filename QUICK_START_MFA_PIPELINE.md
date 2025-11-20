# Quick Start: MFA Onset Detection Pipeline

## Installation

```bash
# Clone the repository
git clone https://github.com/yohken/Tap_TA_onsetDetection.git
cd Tap_TA_onsetDetection

# Install dependencies
pip install -r requirements.txt

# Optional: Install MFA for automatic TextGrid generation
# See: https://montreal-forced-aligner.readthedocs.io/
conda install -c conda-forge montreal-forced-aligner
```

## Usage

### Step 1: Generate Example Files (Optional)

```bash
# Create synthetic example audio files with /t/ bursts
python example_mfa_pipeline.py
```

This creates 3 example WAV files: `example_speech_1.wav`, `example_speech_2.wav`, `example_speech_3.wav`

### Step 2: Process Audio Files

```bash
# Basic usage (MFA alignment runs automatically if MFA is installed)
python mfa_onset_pipeline.py example_speech_*.wav

# Without MFA alignment (if TextGrid files already exist)
python mfa_onset_pipeline.py example_speech_*.wav --no-mfa

# Custom output directory
python mfa_onset_pipeline.py example_speech_*.wav -o my_results/
```

### Step 3: View Results

After processing, check the output directory (default: `mfa_onset_results/`):

```
mfa_onset_results/
├── example_speech_1_comparison.png    # Comparison plot
├── example_speech_1_results.csv       # Detection results (CSV)
├── example_speech_1_results.json      # Full results with parameters (JSON)
├── example_speech_2_comparison.png
├── example_speech_2_results.csv
├── example_speech_2_results.json
├── example_speech_3_comparison.png
├── example_speech_3_results.csv
├── example_speech_3_results.json
├── processing_summary.txt             # Summary report
└── pipeline_YYYYMMDD_HHMMSS.log      # Detailed log
```

## Output Example

### Comparison Plot

The pipeline generates a three-panel comparison plot for each WAV file:

![Example Output](https://github.com/user-attachments/assets/8b6a1f99-f762-4dec-8d1a-215585947430)

- **Top panel**: Waveform with onset markers (blue dashed = Hilbert onsets, green dotted = peaks)
- **Middle panel**: MFA high-frequency RMS envelope (shows "not available" if MFA not run)
- **Bottom panel**: Hilbert envelope with onset and peak markers

### CSV Results

```csv
method,onset_time_s,peak_time_s,file
Hilbert,0.5030422034628117,0.515125,example_speech_1.wav
Hilbert,1.203004660322513,1.2155,example_speech_1.wav
Hilbert,1.9029636569364679,1.915125,example_speech_1.wav
Hilbert,2.602956647225807,2.614625,example_speech_1.wav
```

### JSON Results

```json
{
  "file": "example_speech_1.wav",
  "timestamp": "2025-11-20T04:49:56.983046",
  "mfa_detection": {
    "onsets": [],
    "count": 0,
    "parameters": null,
    "error": "MFA skipped"
  },
  "hilbert_detection": {
    "onsets": [0.503, 1.203, 1.903, 2.603],
    "peaks": [0.515, 1.216, 1.915, 2.615],
    "count": 4,
    "parameters": {
      "target_sr": 48000,
      "hp_cutoff": 500.0,
      "threshold_ratio": 0.1,
      "lookback_points": 74,
      "min_interval_ms": 50.0
    }
  }
}
```

## Customizing Parameters

```bash
# Adjust MFA detection parameters
python mfa_onset_pipeline.py file.wav \
    --mfa-high-freq 2500 \
    --mfa-frame-length 10 \
    --mfa-threshold 3.0

# Adjust Hilbert detection parameters
python mfa_onset_pipeline.py file.wav \
    --hilbert-hp-cutoff 600 \
    --hilbert-threshold 0.15 \
    --hilbert-min-interval 100

# Combined with verbose output
python mfa_onset_pipeline.py file.wav \
    --hilbert-hp-cutoff 600 \
    -o results/ \
    -v
```

## Getting Help

```bash
# Show all available options
python mfa_onset_pipeline.py --help

# Read detailed documentation
cat MFA_PIPELINE_README.md

# View implementation summary
cat IMPLEMENTATION_MFA_PIPELINE.md
```

## Common Use Cases

### 1. Process Multiple Speech Recordings

```bash
python mfa_onset_pipeline.py speech/*.wav -o speech_results/
```

### 2. Use with Existing TextGrid Files

```bash
# Place TextGrid files in the same directory as WAV files with matching names
# Example: file.wav and file.TextGrid
python mfa_onset_pipeline.py file.wav
```

### 3. Fine-tune Detection for Noisy Audio

```bash
python mfa_onset_pipeline.py noisy.wav \
    --hilbert-hp-cutoff 700 \
    --hilbert-threshold 0.15
```

## Troubleshooting

**Problem**: No onsets detected
- Try lowering `--hilbert-threshold` (e.g., 0.05)
- Check if high-pass cutoff is appropriate for your audio

**Problem**: Too many false positives
- Try increasing `--hilbert-threshold` (e.g., 0.2)
- Increase `--hilbert-min-interval` to avoid rapid duplicates

**Problem**: MFA alignment fails
- Check that MFA is installed: `mfa version`
- Verify text content matches audio: `--text-content "ta ta ta"`
- Try simpler text or single words

For more help, see `MFA_PIPELINE_README.md`
