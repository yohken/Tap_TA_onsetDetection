# Tap_TA_onsetDetection

A Python module for onset detection in audio signals, specifically designed for three types of audio:
1. **Metronome click tracks** - theoretical grid positions (no detection needed)
2. **Finger tap recordings** - detect percussive tap onsets
3. **Sung Japanese syllable "ta"** - detect the /t/ burst onset using MFA TextGrid annotations

## Features

- Clean, well-documented Python code using standard DSP techniques (no machine learning)
- Python 3.10+ with full type hints
- Uses industry-standard libraries: numpy, scipy, librosa, textgrid, matplotlib

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Click Track Onsets (Theoretical)

```python
import onset_detection

# Generate 8 clicks at 120 BPM
onsets = onset_detection.get_click_onsets_from_bpm(bpm=120, n_clicks=8)
print(onsets)  # [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
```

### 2. Detect Tap Onsets from Audio

```python
# Detect taps from a WAV file
tap_onsets = onset_detection.detect_tap_onsets_from_audio(
    'recording.wav',
    hp_cutoff=500.0,           # High-pass filter at 500 Hz
    diff_threshold_std=2.0,     # Threshold: mean + 2*std
    min_interval_ms=50.0        # Minimum 50ms between taps
)
print(f"Detected {len(tap_onsets)} taps at: {tap_onsets}")
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

#### `detect_onsets_from_envelope()`
Detect onsets from an envelope using derivative-based peak picking.

#### `get_click_onsets_from_bpm()`
Generate theoretical click track onset times from BPM and subdivision.

#### `detect_tap_onsets_from_audio()`
Detect tap onsets from audio file using high-pass filtered RMS envelope.

#### `detect_t_burst_onsets_from_mfa()`
Detect /t/ burst onsets using MFA TextGrid and high-frequency energy analysis.

#### `plot_envelope_with_onsets()`
Visualization helper for debugging onset detection.

For detailed parameter documentation, see the docstrings in `onset_detection.py`.

## Demo

Run the built-in demo:

```bash
python onset_detection.py
```

Note: The demo requires test audio files. See the `__main__` block in `onset_detection.py` for examples.

## Algorithm Details

### Tap Onset Detection
1. Load audio and apply high-pass filter (default: 500 Hz)
2. Compute short-time RMS envelope
3. Calculate first-order difference to detect energy rises
4. Apply threshold: mean + k*std of positive differences
5. Find peaks exceeding threshold with minimum spacing

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