# Tap_TA_onsetDetection

A Python module for onset detection in audio signals, specifically designed for three types of audio:
1. **Metronome click tracks** - theoretical grid positions (no detection needed)
2. **Finger tap recordings** - detect percussive tap onsets
3. **Sung Japanese syllable "ta"** - detect the /t/ burst onset using MFA TextGrid annotations

## Features

- **MFA-based Onset Detection Pipeline**: Comprehensive pipeline combining Montreal Forced Aligner (MFA) TextGrid annotations with Hilbert-based detection for /t/ burst onset analysis
- **GUI Application**: Easy-to-use graphical interface with file selection dialogs and automatic plotting
- **Envelope Comparison Framework**: Systematic comparison of onset detection variants with quantitative metrics
- Clean, well-documented Python code using standard DSP techniques (no machine learning)
- Python 3.10+ with full type hints
- Uses industry-standard libraries: numpy, scipy, librosa, textgrid, matplotlib

## Installation

```bash
pip install -r requirements.txt
```

Note: tkinter is usually included with Python. If not, install it:
- Ubuntu/Debian: `sudo apt-get install python3-tk`
- macOS: Included with Python
- Windows: Included with Python

## Quick Start

### MFA-based Onset Detection Pipeline (NEW)

Automatically detect /t/ burst onsets using both MFA TextGrid annotations and Hilbert-based detection:

```bash
# Process single file (MFA alignment runs automatically)
python mfa_onset_pipeline.py speech.wav

# Process multiple files (MFA alignment runs automatically)
python mfa_onset_pipeline.py file1.wav file2.wav file3.wav

# Process without MFA (if TextGrid files already exist)
python mfa_onset_pipeline.py file1.wav --no-mfa

# Process with custom parameters
python mfa_onset_pipeline.py speech.wav \
    --mfa-high-freq 2500 \
    --hilbert-hp-cutoff 600 \
    --hilbert-threshold 0.15 \
    -o results/

# Custom text content for MFA alignment (advanced)
python mfa_onset_pipeline.py speech.wav \
    --text-content "ta ta ta" \
    --mfa-model english_us_arpa \
    --mfa-dictionary english_us_arpa
```

**What it does**:
- Accepts WAV files containing singing audio (Japanese "ta" syllables)
- Automatically runs MFA alignment to generate TextGrid files
- Detects /t/ burst onsets using MFA TextGrid annotations (high-frequency RMS envelope method)
- Detects onsets using Hilbert-based method (Fujii method)
- Creates comparison plots showing both methods with waveform, envelopes, and markers
- Exports results to CSV and JSON with full parameter details
- Generates processing log and summary report
- Handles errors gracefully and continues processing remaining files

**Output**:
- `*_comparison.png`: Comparison plot with both detection methods
- `*_results.csv`: Detection results in CSV format (method, onset_time, peak_time)
- `*_results.json`: Full results with parameters in JSON format
- `processing_summary.txt`: Summary report for all processed files
- `pipeline_*.log`: Detailed processing log

**Default Parameters**:
- Text content: "ta" (automatically set for /t/ burst detection)
- MFA: high_freq_min=2000 Hz, frame=5ms, hop=1ms, threshold_std=2.0
- Hilbert: sr=48000 Hz, hp_cutoff=500 Hz, threshold_ratio=0.1, lookback_points=74, min_interval=50ms

All parameters can be customized via command-line arguments. See `python mfa_onset_pipeline.py --help` for full documentation.

### Envelope Comparison

Compare different envelope configurations and detection parameters:

```bash
# Basic comparison
python compare_envelopes.py --wav audio.wav --export_plots

# Test different HPF cutoffs
python compare_envelopes.py --wav audio.wav \
    --hpf_cutoffs 0,300,500,1000 \
    --export_plots --out_dir results

# Compare smoothing effects  
python compare_envelopes.py --wav audio.wav \
    --smooth_ms 0,0.1,0.5,1.0 \
    --hpf_cutoffs 300 \
    --export_plots --out_dir results
```

**What it does**:
- Generates multiple envelope variants (Hilbert smoothed/unsmoothed, RMS-based)
- Tests different HPF cutoffs, smoothing windows, and detection parameters
- Produces quantitative metrics for each variant (rise time, slope, event count)
- Creates multi-panel comparison plots
- Exports results to CSV with git commit hash for reproducibility

**Use cases**:
- Choose optimal parameters for your audio type
- Understand how parameters affect detection
- Document and reproduce detection configurations
- Compare different envelope methods side-by-side

See [COMPARISON_README.md](COMPARISON_README.md) for detailed documentation on:
- Metrics explanation (rise time, slope, etc.)
- Parameter effects and trade-offs
- Usage examples for different audio types
- Interpreting results and selecting best configuration

### GUI Application (Recommended)

Launch the graphical interface:

```bash
python onset_detection_gui.py
```

The GUI provides:
- File selection dialogs for easy audio file selection
- Automatic visualization of detection results
- Support for all three detection methods
- Real-time status updates and results display

See [GUI_README.md](GUI_README.md) for detailed GUI documentation.

### Command-Line Interface

### 1. Generate Click Track Onsets (Theoretical)

```python
import onset_detection

# Generate 8 clicks at 120 BPM
onsets = onset_detection.get_click_onsets_from_bpm(bpm=120, n_clicks=8)
print(onsets)  # [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
```

### 2. Detect Tap Onsets from Audio

```python
# Original method: Detect taps using RMS envelope
tap_onsets = onset_detection.detect_tap_onsets_from_audio(
    'recording.wav',
    hp_cutoff=500.0,           # High-pass filter at 500 Hz
    diff_threshold_std=2.0,     # Threshold: mean + 2*std
    min_interval_ms=50.0        # Minimum 50ms between taps
)
print(f"Detected {len(tap_onsets)} taps at: {tap_onsets}")

# New method: Detect taps using Hilbert envelope (48kHz, with lookback criterion)
tap_onsets = onset_detection.detect_tap_onsets_from_audio_hilbert(
    'recording.wav',
    target_sr=48000,           # Resample to 48kHz
    hp_cutoff=500.0,           # High-pass filter at 500 Hz
    threshold_ratio=0.1,        # 10% of peak amplitude
    lookback_points=74,         # ~2ms lookback at 48kHz
    min_interval_ms=50.0        # Minimum 50ms between taps
)
print(f"Detected {len(tap_onsets)} taps at: {tap_onsets}")
```

### 2b. Detect Metronome Onsets from Audio

```python
# Detect metronome clicks using Hilbert envelope (48kHz)
metronome_onsets = onset_detection.detect_metronome_onsets_from_audio(
    'metronome.wav',
    target_sr=48000,           # Resample to 48kHz
    threshold_ratio=0.1,        # 10% of peak amplitude
    min_interval_ms=50.0        # Minimum 50ms between clicks
)
print(f"Detected {len(metronome_onsets)} clicks at: {metronome_onsets}")
```

### 3. Detect /t/ Burst Onsets with MFA TextGrid

```python
# Detect /t/ bursts using Montreal Forced Aligner TextGrid
t_burst_onsets = onset_detection.detect_t_burst_onsets_from_mfa(
    'speech.wav',
    'speech.TextGrid',
    tier_name='phones',          # Tier containing phone segments
    phone_label='t',             # Look for 't' segments
    high_freq_min=2000.0         # Use high-frequency energy (2+ kHz)
)
print(f"Detected {len(t_burst_onsets)} /t/ bursts")
```

## API Reference

### Core Functions

#### `compute_rms_envelope()`
Compute short-time RMS envelope with optional band-pass filtering.

#### `compute_hilbert_envelope()`
Compute Hilbert envelope: E(t) = sqrt(x(t)^2 + x_hat(t)^2) using Hilbert transform. Provides instantaneous amplitude of the signal.

#### `detect_onsets_from_envelope()`
Detect onsets from an envelope using derivative-based peak picking.

#### `get_click_onsets_from_bpm()`
Generate theoretical click track onset times from BPM and subdivision.

#### `detect_metronome_onsets_from_audio()`
Detect metronome onsets using Hilbert envelope at 48kHz sampling rate. Onset defined as when envelope exceeds 10% of peak amplitude for each sound burst.

#### `detect_tap_onsets_from_audio()`
Detect tap onsets from audio file using high-pass filtered RMS envelope (original method).

#### `detect_tap_onsets_from_audio_hilbert()`
Detect tap onsets using Hilbert envelope at 48kHz with high-pass filtering. Onset defined as when amplitude exceeds 10% of peak, with 74-point (~2ms) lookback criterion to ensure quiet period before onset.

#### `detect_t_burst_onsets_from_mfa()`
Detect /t/ burst onsets using MFA TextGrid and high-frequency energy analysis.

#### `plot_envelope_with_onsets()`
Visualization helper for debugging onset detection.

**Interactive Features (GUI Mode):**
- **Multiple File Selection**: Select multiple WAV files to process sequentially
- **HPF Frequency Slider**: Adjust High-Pass Filter cutoff frequency (100-2000 Hz) after initial detection
- **Re-detect Button**: Recompute onset detection with new HPF frequency without restarting
- **Marker Deletion**: Cmd+Shift+Click (Mac) or Ctrl+Shift+Click (Windows/Linux) to delete false positive markers
- **Export Button**: Save onset/peak data to CSV via file dialog
- **Next Button**: Navigate to next file in batch or open new file dialog
- **Onset Count Display**: Real-time display of number of detected onsets
- **X-axis Zoom**: Use mouse wheel (or trackpad pinch) to zoom in/out on the time axis
  - Scroll UP or pinch OUT to zoom in
  - Scroll DOWN or pinch IN to zoom out
  - Zoom is centered on your mouse cursor position
  - Both plots zoom together (synchronized X-axis)
  - Zoom is preserved when deleting markers

**NOTE:** For tap/click re-detection that complies with the Fujii method (10% threshold, backward search, linear interpolation), use `onset_hilbert` module instead:
- `onset_hilbert.plot_waveform_and_envelope_interactive()` for interactive re-detection
- `onset_hilbert.detect_tap_onsets_and_peaks()` or `detect_click_onsets_and_peaks()` for detection
- See `onset_hilbert_README.md` for full documentation

For detailed parameter documentation, see the docstrings in `onset_detection.py` and `onset_hilbert.py`.

## Demo

Run the built-in demo:

```bash
python onset_detection.py
```

Note: The demo requires test audio files. See the `__main__` block in `onset_detection.py` for examples.

## Algorithm Details

### Fujii Method (Recommended for Tap/Click Detection)

The **Fujii method** is the primary onset detection method implemented in `onset_hilbert.py`. It provides:

1. **High-pass filtering**: Remove low-frequency noise with zero-phase Butterworth filter
2. **Hilbert envelope**: Compute instantaneous amplitude E(t) = |hilbert(y)|
3. **Peak detection**: Find local maxima in envelope representing sound bursts
4. **10% threshold per peak**: For each peak with amplitude Amax, set threshold = 0.1 × Amax
5. **Backward search**: From each peak, search backward to find where envelope first crosses threshold
6. **Linear interpolation**: For sub-sample precision, interpolate between samples at threshold crossing
7. **Output**: Both onset time (10% crossing) and peak time

**Key advantages:**
- Sub-sample precision (< 0.1ms error at 48kHz)
- Per-event adaptive threshold (not affected by global amplitude variations)
- Zero-phase filtering preserves timing information
- Operates on continuous waveform (no frame-based artifacts)

**For re-detection with HPF changes, always use `onset_hilbert` module to ensure Fujii method compliance.**

### Metronome Onset Detection (New Hilbert-based Method)
1. Load audio and resample to 48,000 Hz
2. Compute Hilbert envelope: E(t) = sqrt(x(t)^2 + x_hat(t)^2)
3. Find prominent peaks in envelope (sound bursts)
4. For each peak, search backward to find onset point
5. Onset defined as when envelope exceeds 10% of peak amplitude

### Tap Onset Detection (Original RMS Method)
1. Load audio and apply high-pass filter (default: 500 Hz)
2. Compute short-time RMS envelope
3. Calculate first-order difference to detect energy rises
4. Apply threshold: mean + k*std of positive differences
5. Find peaks exceeding threshold with minimum spacing

**NOTE:** This RMS method does NOT include linear interpolation and is not Fujii-compliant. Use `onset_hilbert` for Fujii method.

### Tap Onset Detection (New Hilbert-based Method)
1. Load audio and resample to 48,000 Hz
2. Apply high-pass filter (default: 500 Hz) as preprocessing
3. Compute Hilbert envelope: E(t) = sqrt(x(t)^2 + x_hat(t)^2)
4. Find prominent peaks in envelope (sound bursts)
5. For each peak, search backward to find onset point
6. Onset defined as when envelope exceeds 10% of peak amplitude
7. Valid only if envelope was below threshold for 74 points (~2ms) before onset

### /t/ Burst Detection
1. Load audio and MFA TextGrid
2. For each phone segment labeled as 't':
   - Extract audio segment
   - Compute high-frequency RMS envelope (>2000 Hz)
   - Detect earliest onset within the segment
   - This captures the burst release after closure

## Requirements

- Python 3.10+
- numpy >= 1.21.0
- scipy >= 1.7.0
- librosa >= 0.9.0
- textgrid >= 1.5.0
- matplotlib >= 3.4.0

## License

See repository license.

## Contributing

Contributions welcome! Please ensure code follows the existing style with type hints and comprehensive docstrings.