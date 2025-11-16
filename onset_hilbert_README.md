# onset_hilbert.py - Unified Onset + Peak Detection Module

A Python module implementing Fujii-style onset and peak detection for metronome clicks and finger taps using Hilbert envelope analysis.

## Features

- **Unified algorithm** for both metronome clicks and finger taps
- **Hilbert envelope** analysis (not RMS-based)
- **10% threshold method**: onset = when envelope first exceeds 10% of local peak amplitude
- **Sub-sample precision**: linear interpolation for accurate onset timing
- **Zero-phase filtering**: configurable Butterworth high-pass filter
- **CSV export**: save onset and peak times with pandas
- **Visualization**: plot waveform, envelope, and detected events
- **Interactive tuning**: manually adjust HPF cutoff and see results
- **GUI helpers**: file dialogs for easy WAV selection and CSV saving

## Installation

```bash
pip install numpy scipy soundfile matplotlib pandas
```

Note: `tkinter` is required for GUI dialogs (usually included with Python).

## Quick Start

### 1. Detect Click Onsets and Peaks

```python
import onset_hilbert

# Detect metronome clicks (default HPF = 1000 Hz)
onset_times, peak_times = onset_hilbert.detect_click_onsets_and_peaks(
    'metronome.wav',
    hp_cutoff_hz=1000.0,
    threshold_ratio=0.1,
    min_distance_ms=100.0
)

print(f"Detected {len(onset_times)} clicks")
print(f"Onset times [s]: {onset_times}")
print(f"Peak times [s]:  {peak_times}")
```

### 2. Detect Tap Onsets and Peaks

```python
# Detect finger taps (default HPF = 300 Hz)
onset_times, peak_times = onset_hilbert.detect_tap_onsets_and_peaks(
    'taps.wav',
    hp_cutoff_hz=300.0,
    threshold_ratio=0.1,
    min_distance_ms=100.0
)
```

### 3. Save Results to CSV

```python
# Save with label for identification
onset_hilbert.save_onsets_and_peaks_csv(
    'results.csv',
    onset_times,
    peak_times,
    label='click'
)
```

CSV format:
```
index,onset_sec,peak_sec,label
0,0.299,0.300,click
1,0.799,0.800,click
2,1.299,1.300,click
```

### 4. Visualize Results

```python
import soundfile as sf

# Load audio
data, sr = sf.read('metronome.wav')

# Apply filtering and envelope
y_filt = onset_hilbert.highpass_filter(data, sr, 1000.0)
env = onset_hilbert.hilbert_envelope(y_filt, sr, smooth_ms=0.5)

# Detect
onset_times, peak_times = onset_hilbert.detect_onsets_and_peaks_from_envelope(
    env, sr
)

# Plot
onset_hilbert.plot_waveform_and_envelope(
    y_filt, sr, env, onset_times, peak_times,
    hp_cutoff_hz=1000.0,
    title='Click Detection Results'
)
```

### 5. Interactive HPF Tuning

```python
# Interactively adjust the high-pass filter cutoff
onset_hilbert.interactive_hpf_tuning(
    'recording.wav',
    initial_hp_cutoff_hz=300.0,
    is_click=False
)
# This will show plots and prompt for new cutoff values
# Press ENTER to exit
```

### 6. GUI Helpers

```python
# Click detection with GUI dialogs
onset_hilbert.run_click_detection_with_dialog(
    default_hp_cutoff_hz=1000.0
)

# Tap detection with GUI dialogs
onset_hilbert.run_tap_detection_with_dialog(
    default_hp_cutoff_hz=300.0
)
```

## API Reference

### Core Functions

#### `highpass_filter(y, sr, cutoff_hz, order=4)`

Apply zero-phase Butterworth high-pass filter.

**Parameters:**
- `y` (ndarray): Mono audio signal
- `sr` (int): Sampling rate [Hz]
- `cutoff_hz` (float | None): HPF cutoff in Hz. If None or ≤0, returns y unchanged
- `order` (int): Filter order (default: 4)

**Returns:** Filtered signal (same shape as y)

---

#### `hilbert_envelope(y, sr, smooth_ms=0.5)`

Compute Hilbert envelope with optional smoothing.

**Parameters:**
- `y` (ndarray): Mono audio signal
- `sr` (int): Sampling rate [Hz]
- `smooth_ms` (float | None): Moving-average window in milliseconds. If None or ≤0, no smoothing

**Returns:** Envelope array (same length as y)

---

#### `detect_onsets_and_peaks_from_envelope(env, sr, threshold_ratio=0.1, min_distance_ms=100.0, global_min_height_ratio=0.2)`

Core detection algorithm using Fujii-style 10% threshold method.

**Algorithm:**
1. Find peaks in envelope using `scipy.signal.find_peaks`
2. For each peak, search backward to find where envelope first exceeds 10% of peak amplitude
3. Linearly interpolate for sub-sample precision
4. Return onset and peak times in seconds

**Parameters:**
- `env` (ndarray): Hilbert envelope
- `sr` (int): Sampling rate [Hz]
- `threshold_ratio` (float): Local onset threshold as fraction of peak amplitude (default: 0.1)
- `min_distance_ms` (float): Minimum distance between peaks in ms (default: 100.0)
- `global_min_height_ratio` (float): Global height threshold relative to env.max() (default: 0.2)

**Returns:** Tuple of (onset_times, peak_times) in seconds

---

#### `detect_click_onsets_and_peaks(wav_path, hp_cutoff_hz=1000.0, ...)`

Detect onsets and peaks for metronome clicks.

**Default HPF:** 1000 Hz (suitable for high-frequency metronome clicks)

---

#### `detect_tap_onsets_and_peaks(wav_path, hp_cutoff_hz=300.0, ...)`

Detect onsets and peaks for finger taps.

**Default HPF:** 300 Hz (suitable for lower-frequency taps)

---

#### `save_onsets_and_peaks_csv(out_path, onset_times, peak_times, label=None)`

Save detected times to CSV using pandas.

**CSV Columns:**
- `index`: Integer index (0, 1, 2, ...)
- `onset_sec`: Onset times [s]
- `peak_sec`: Peak times [s]
- `label`: Optional string label

---

#### `plot_waveform_and_envelope(y, sr, env, onset_times, peak_times, hp_cutoff_hz=None, title="")`

Plot waveform, envelope, and detected events.

**Legend shows:**
- Waveform
- Hilbert envelope (with HPF cutoff if applicable)
- Onsets (green dashed lines)
- Peaks (red dotted lines)

---

#### `interactive_hpf_tuning(wav_path, initial_hp_cutoff_hz=300.0, is_click=False, ...)`

REPL-style interactive HPF tuning.

**Usage:**
1. Shows plot with current HPF cutoff
2. Prompts for new cutoff value
3. Press ENTER to exit

---

#### `run_click_detection_with_dialog(default_hp_cutoff_hz=1000.0, ...)`

GUI helper for click detection using tkinter file dialogs.

---

#### `run_tap_detection_with_dialog(default_hp_cutoff_hz=300.0, ...)`

GUI helper for tap detection using tkinter file dialogs.

## Algorithm Details

### Fujii-Style Onset Detection

The method is based on Fujii et al.'s approach:

1. **High-pass filtering**: Remove low-frequency noise (configurable cutoff)
2. **Hilbert transform**: Compute instantaneous envelope E(t) = |hilbert(y)|
3. **Peak detection**: Find local maxima in envelope (sound bursts)
4. **Onset localization**: For each peak with amplitude Amax:
   - Threshold = 0.1 × Amax (10%)
   - Search backward from peak to find threshold crossing
   - Linearly interpolate between samples for sub-sample precision
5. **Output**: Both onset time (10% crossing) and peak time

### Key Differences from RMS-Based Methods

- Uses **Hilbert envelope** (instantaneous amplitude) instead of short-time RMS
- **No frame-based analysis** - operates on continuous waveform
- **Per-event adaptive threshold** (10% of local peak, not global threshold)
- **Sub-sample precision** via linear interpolation
- **Zero-phase filtering** preserves timing information

### Default Parameters

| Parameter | Clicks | Taps | Notes |
|-----------|--------|------|-------|
| HPF cutoff | 1000 Hz | 300 Hz | Clicks are higher frequency |
| Threshold ratio | 0.1 (10%) | 0.1 (10%) | Fujii standard |
| Min distance | 100 ms | 100 ms | Typical tap/click interval |
| Smoothing | 0.5 ms | 0.5 ms | Minimal smoothing |

## Requirements

- Python 3.10+
- numpy >= 1.21.0
- scipy >= 1.7.0
- soundfile >= 0.10.0
- matplotlib >= 3.4.0
- pandas >= 1.3.0
- tkinter (usually included with Python)

## Testing

Run the test suite:

```bash
python -m unittest test_onset_hilbert -v
```

All 17 tests should pass with sub-millisecond accuracy.

## Example Output

```
Creating test WAV with clicks at: [0.3, 0.8, 1.3, 1.8]
Detected 4 events:
Onset times [s]: [0.29915949 0.7991815  1.29915052 1.79917764]
Peak times [s]:  [0.3 0.8 1.3 1.8]

Accuracy check:
  Click 1: expected 0.300s, detected onset 0.299s (error: 0.8ms)
           peak at 0.300s, onset-to-peak: 0.8ms
  Click 2: expected 0.800s, detected onset 0.799s (error: 0.8ms)
           peak at 0.800s, onset-to-peak: 0.8ms
```

## References

This implementation follows the Fujii-style onset detection method:
- Hilbert envelope for instantaneous amplitude
- 10% threshold relative to local peak
- Sub-sample interpolation for precision

## License

See repository license.
