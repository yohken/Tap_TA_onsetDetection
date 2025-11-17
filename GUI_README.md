# Onset Detection GUI

A graphical user interface for the onset detection module.

## Features

- **File Selection Dialog**: Select audio files (and TextGrid files) using native file dialogs
- **Automatic Plotting**: Results are automatically visualized using matplotlib
- **Interactive HPF Control**: Adjust High-Pass Filter frequency after initial detection with a slider and re-detect button
- **Interactive Plot Zoom**: Use mouse wheel to zoom in/out on the time axis for detailed analysis
- **Three Detection Methods**:
  1. Tap Onset Detection - Select a WAV file to detect percussive taps
  2. /t/ Burst Onset Detection - Select WAV and TextGrid files to detect /t/ bursts
  3. Click Track Generation - Generate theoretical click positions with custom BPM

## Usage

Run the GUI application:

```bash
python onset_detection_gui.py
```

### Tap Onset Detection

1. Click "Detect Tap Onsets"
2. Select a WAV file containing tap recordings
3. View the results in the text area
4. A plot window will open showing the waveform, envelope, and detected onsets
5. **Adjust the HPF frequency slider** (100-2000 Hz) to change the filter cutoff
6. **Click the "Re-detect" button** to recompute onsets with the new frequency
7. **Use mouse wheel to zoom in/out on the X-axis** for detailed inspection

### /t/ Burst Onset Detection

1. Click "Detect /t/ Burst Onsets"
2. Select a WAV file containing speech with /t/ sounds
3. Select the corresponding TextGrid file from MFA
4. View the results in the text area
5. A plot window will open showing the detection results
6. **Adjust the HPF frequency slider** (100-2000 Hz) to change the filter cutoff
7. **Click the "Re-detect" button** to recompute onsets with the new frequency
8. **Use mouse wheel to zoom in/out on the X-axis** for detailed inspection

### Click Track Generation

1. Click "Generate Click Track"
2. Enter BPM, number of clicks, and subdivision
3. View the generated onset times

## Requirements

- Python 3.10+
- tkinter (usually included with Python)
- All dependencies from requirements.txt

## Screenshots

The GUI provides:
- Simple, clean interface with large buttons
- Real-time status updates
- Scrollable results area
- Automatic plot generation with interactive controls

## Notes

- The GUI uses the standard tkinter library (included with Python)
- Plots are displayed using matplotlib's default backend
- All detection algorithms are from the `onset_detection` module
- The original CLI interface in `onset_detection.py` remains available

## Interactive Plot Features

When a plot window opens, you have access to the following interactive features:

### HPF Frequency Control (New!)
- **Slider**: Adjust the High-Pass Filter cutoff frequency from 100 Hz to 2000 Hz in 50 Hz steps
- **Re-detect Button**: Click to recompute onset detection with the new HPF frequency
- **Onset Count Display**: Shows the number of detected onsets, updates automatically after re-detection
- **Real-time Updates**: The waveform, envelope, and onset markers all update when you re-detect

### Zoom and Navigation
- **Zoom In/Out**: Scroll the mouse wheel UP (or pinch OUT on trackpad) to zoom in, DOWN (or pinch IN) to zoom out
- **Zoom Center**: The zoom will be centered on your mouse cursor position
- **X-axis Only**: Only the time axis (X-axis) zooms; the Y-axis remains auto-scaled
- **Synchronized**: Both waveform and envelope plots zoom together
- **Navigation**: Standard matplotlib toolbar buttons are also available for pan/zoom

### Use Cases for HPF Adjustment
- **Low Frequency Noise**: Increase HPF cutoff to filter out rumble, handling noise, or low-frequency interference
- **Fine-tuning**: Experiment with different cutoff frequencies to optimize detection for your specific audio
- **Comparison**: Quickly compare detection results across different HPF settings without restarting the application
