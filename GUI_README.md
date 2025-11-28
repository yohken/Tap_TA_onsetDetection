# Onset Detection GUI

A graphical user interface for the onset detection module.

## Features

- **File Selection Dialog**: Select audio files using native file dialogs
- **Automatic Plotting**: Results are automatically visualized using matplotlib
- **Interactive HPF Control**: Adjust High-Pass Filter frequency after initial detection with a slider and re-detect button
- **Interactive Plot Zoom**: Use mouse wheel to zoom in/out on the time axis for detailed analysis
- **Two Detection Methods**:
  1. Tap Onset Detection - Select a WAV file to detect percussive taps
  2. Voice Segment Detection - Detect voice segments (e.g., "ta" syllables) with feature point extraction

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

### Multiple File Selection (New!)
- **Multi-select**: In the initial file dialog, you can now select multiple WAV files at once
- **File Navigation**: Use the "Next" button in the plot window to move to the next file
- **Single File Mode**: If you selected only one file, the "Next" button will show a new file dialog

### HPF Frequency Control
- **Slider**: Adjust the High-Pass Filter cutoff frequency from 100 Hz to 2000 Hz in 50 Hz steps
- **Re-detect Button**: Click to recompute onset detection with the new parameters
- **Onset Count Display**: Shows the number of detected onsets, updates automatically after re-detection
- **Real-time Updates**: The waveform, envelope, and onset markers all update when you re-detect

### Detection Threshold Control
- **Slider**: Adjust the detection threshold from 1% to 50% in 1% steps
- **Default**: 10% (Fujii method default)
- **Lower threshold**: More sensitive detection, may increase false positives
- **Higher threshold**: Less sensitive detection, may miss quieter onsets

### Marker Deletion (New!)
- **Delete Markers**: Hold Cmd+Shift (Mac) or Ctrl+Shift (Windows/Linux) and click near a marker to delete it
- **Maintains Zoom**: When you delete a marker, the current zoom level is preserved
- **Precise Selection**: Click within 50ms of a marker to delete it
- **Paired Deletion**: Deleting an onset also deletes its corresponding peak, and vice versa

### Data Export (New!)
- **Export Button**: Click the "Export" button to save the current onset and peak data
- **CSV Format**: Data is saved as a CSV file with columns: index, onset_sec, peak_sec, label
- **File Dialog**: Choose the save location and filename through a file dialog
- **Git Integration**: The CSV includes metadata for reproducibility

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

## Workflow Examples

### Processing Multiple Files
1. Click "Detect Tap Onsets"
2. Select multiple WAV files in the file dialog (Ctrl+Click or Cmd+Click to select multiple)
3. Review the first file's detection results
4. Adjust HPF frequency if needed and click "Re-detect"
5. Delete any false positive markers using Cmd+Shift+Click (Mac) or Ctrl+Shift+Click (Windows/Linux)
6. Click "Export" to save the data for this file
7. Click "Next" to move to the next file in your selection
8. Repeat steps 4-7 for each file

### Refining Detection Results
1. After initial detection, use the zoom (mouse wheel) to inspect the waveform closely
2. If you spot false positives (incorrectly detected markers), use Cmd+Shift+Click to delete them
3. If needed, adjust the HPF slider and click "Re-detect" to try different parameters
4. Once satisfied, click "Export" to save the refined results
