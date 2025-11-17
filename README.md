# Tap_TA_onsetDetection

A Python module for onset detection in audio signals, specifically designed for three types of audio:
1. **Metronome click tracks** - theoretical grid positions (no detection needed)
2. **Finger tap recordings** - detect percussive tap onsets
3. **Sung Japanese syllable "ta"** - detect the /t/ burst onset using MFA TextGrid annotations

## Features

- **GUI Application**: Easy-to-use graphical interface with file selection dialogs and automatic plotting
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
- **HPF Frequency Slider**: Adjust High-Pass Filter cutoff frequency (100-2000 Hz) after initial detection
- **Re-detect Button**: Recompute onset detection with new HPF frequency without restarting
- **Onset Count Display**: Real-time display of number of detected onsets
- **X-axis Zoom**: Use mouse wheel (or trackpad pinch) to zoom in/out on the time axis
  - Scroll UP or pinch OUT to zoom in
  - Scroll DOWN or pinch IN to zoom out
  - Zoom is centered on your mouse cursor position
  - Both plots zoom together (synchronized X-axis)

For detailed parameter documentation, see the docstrings in `onset_detection.py`.

## Demo

Run the built-in demo:

```bash
python onset_detection.py
```

Note: The demo requires test audio files. See the `__main__` block in `onset_detection.py` for examples.

## Algorithm Details

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