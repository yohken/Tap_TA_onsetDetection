"""
GUI application for onset detection with file selection and result plotting.

This module provides a graphical user interface for:
1. Tap onset detection from WAV files
2. Voice segment detection (e.g., "ta" syllables) with feature point extraction

Features:
- File selection dialogs for audio files
- Voice segment detection based on amplitude/RMS threshold
- Feature point extraction for each voice segment:
  - t_start: consonant onset (Fujii 10% method)
  - t_peak: burst maximum point
  - a_start: vowel onset (periodicity-based voicing detection)
  - a_peak: first stable periodic peak
  - a_end: segment end
- Automatic visualization of detection results with color-coded markers
- User-friendly interface using tkinter

Target Python version: 3.10+
Dependencies: tkinter (standard library), onset_detection module,
              librosa, numpy, scipy, matplotlib, ta_onset_analysis
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import librosa
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import onset_detection
import onset_hilbert
import ta_onset_analysis
import os
import numpy as np
import scipy.signal
import pandas as pd

# Configure matplotlib backend for GUI
# TkAgg will be used automatically when available


# ==============================================================================
# CONFIGURABLE PARAMETERS FOR VOICE SEGMENT DETECTION
# ==============================================================================

# Amplitude threshold for voice segment detection (ratio of max amplitude)
# A segment is detected when amplitude exceeds this ratio of the maximum
AMPLITUDE_THRESHOLD_RATIO = 0.05  # 5% of max amplitude

# Minimum segment duration in seconds
# Segments shorter than this are ignored (noise rejection)
MIN_SEGMENT_DURATION_SEC = 0.05  # 50ms

# Minimum silence duration between segments in seconds
# Used to separate distinct voice segments
MIN_SILENCE_DURATION_SEC = 0.03  # 30ms

# RMS window size in milliseconds for envelope computation
RMS_FRAME_LENGTH_MS = 5.0

# RMS hop size in milliseconds
RMS_HOP_LENGTH_MS = 1.0

# ==============================================================================


def save_ta_features_csv(
    out_path: str,
    features_list: list[dict[str, float]],
) -> None:
    """
    Save detected TA feature points to a CSV file.

    Columns:
        index: integer index (0, 1, 2, ...)
        t_start: t_start times [s]
        t_peak: t_peak times [s]
        a_start: a_start times [s]
        a_peak: a_peak times [s]
        a_end: a_end times [s]

    Args:
        out_path: path to CSV file.
        features_list: list of feature dictionaries.
    
    Raises:
        OSError: If the file cannot be written (permission denied, disk full, etc.).
    """
    # Build DataFrame
    data_dict = {
        'index': [],
        't_start': [],
        't_peak': [],
        'a_start': [],
        'a_peak': [],
        'a_end': [],
    }
    
    for i, features in enumerate(features_list):
        data_dict['index'].append(i)
        data_dict['t_start'].append(features.get('t_start', np.nan))
        data_dict['t_peak'].append(features.get('t_peak', np.nan))
        data_dict['a_start'].append(features.get('a_start', np.nan))
        data_dict['a_peak'].append(features.get('a_peak', np.nan))
        data_dict['a_end'].append(features.get('a_end', np.nan))
    
    df = pd.DataFrame(data_dict)
    
    # Save to CSV
    df.to_csv(out_path, index=False)


def save_ta_features_csv_with_retry(
    out_path: str,
    features_list: list[dict[str, float]],
    *,
    parent_window: object | None = None,
) -> tuple[bool, str | None]:
    """
    Save TA feature points to a CSV file with error handling and retry.
    
    If a file I/O error occurs (permission denied, disk full, etc.), the user is
    informed of the error and prompted to select a new save location.
    
    Args:
        out_path: initial path to CSV file.
        features_list: list of feature dictionaries.
        parent_window: optional parent tkinter window for dialogs.
    
    Returns:
        Tuple of (success, actual_path):
            - success: True if file was saved successfully, False if cancelled.
            - actual_path: The path where file was saved, or None if cancelled.
    """
    current_path = out_path
    
    while True:
        try:
            save_ta_features_csv(current_path, features_list)
            return (True, current_path)
        except OSError as e:
            # Create a root window if none provided
            created_root = None
            try:
                if parent_window is None:
                    created_root = tk.Tk()
                    created_root.withdraw()
                    created_root.attributes('-topmost', True)
                
                # Inform user of the error
                error_message = (
                    f"ファイルの書き込みに失敗しました。\n\n"
                    f"エラー: {str(e)}\n"
                    f"パス: {current_path}\n\n"
                    f"別の保存場所を選択しますか？"
                )
                
                retry = messagebox.askyesno(
                    "ファイル書き込みエラー",
                    error_message,
                    icon='warning'
                )
                
                if not retry:
                    # User chose not to retry
                    return (False, None)
                
                # Get default filename from current path
                default_name = os.path.basename(current_path)
                
                # Show save file dialog for new location
                new_path = filedialog.asksaveasfilename(
                    title="新しい保存場所を選択",
                    defaultextension=".csv",
                    initialfile=default_name,
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                )
                
                if not new_path:
                    # User cancelled the dialog
                    return (False, None)
                
                current_path = new_path
            finally:
                # Clean up root window if we created it
                if created_root is not None:
                    created_root.destroy()


def detect_voice_segments(
    y: np.ndarray,
    sr: int,
    *,
    amplitude_threshold_ratio: float = AMPLITUDE_THRESHOLD_RATIO,
    min_segment_duration_sec: float = MIN_SEGMENT_DURATION_SEC,
    min_silence_duration_sec: float = MIN_SILENCE_DURATION_SEC,
) -> list[tuple[int, int]]:
    """
    Detect voice segments from audio based on amplitude threshold.
    
    A voice segment is a continuous region where the amplitude envelope
    exceeds a threshold value. Segments are separated by silence periods.
    
    Args:
        y: Audio signal (mono, float).
        sr: Sampling rate in Hz.
        amplitude_threshold_ratio: Threshold as fraction of max amplitude (default: 0.05).
        min_segment_duration_sec: Minimum segment duration in seconds (default: 0.05).
        min_silence_duration_sec: Minimum silence duration between segments (default: 0.03).
    
    Returns:
        List of (start_sample, end_sample) tuples for each detected segment.
    """
    if len(y) == 0:
        return []
    
    # Compute amplitude envelope using RMS
    frame_length = int(RMS_FRAME_LENGTH_MS * sr / 1000)
    hop_length = int(RMS_HOP_LENGTH_MS * sr / 1000)
    
    env = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Calculate threshold
    max_env = np.max(env)
    if max_env == 0:
        return []
    
    threshold = amplitude_threshold_ratio * max_env
    
    # Find frames above threshold
    above_threshold = env > threshold
    
    # Convert min durations to frames
    min_segment_frames = int(min_segment_duration_sec * sr / hop_length)
    min_silence_frames = int(min_silence_duration_sec * sr / hop_length)
    
    # Find segment boundaries
    segments = []
    in_segment = False
    segment_start = 0
    silence_count = 0
    
    for i, is_above in enumerate(above_threshold):
        if is_above:
            if not in_segment:
                # Start of new segment
                in_segment = True
                segment_start = i
            silence_count = 0
        else:
            if in_segment:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    # End of segment
                    segment_end = i - silence_count
                    # Check minimum duration
                    if segment_end - segment_start >= min_segment_frames:
                        # Convert frame indices to sample indices
                        start_sample = segment_start * hop_length
                        end_sample = min((segment_end + 1) * hop_length, len(y))
                        segments.append((start_sample, end_sample))
                    in_segment = False
                    silence_count = 0
    
    # Handle last segment if still in segment
    if in_segment:
        segment_end = len(above_threshold) - 1
        if segment_end - segment_start >= min_segment_frames:
            start_sample = segment_start * hop_length
            end_sample = min((segment_end + 1) * hop_length, len(y))
            segments.append((start_sample, end_sample))
    
    return segments


def extract_feature_points(
    y: np.ndarray,
    sr: int,
    segment_start: int,
    segment_end: int,
    *,
    hpf_cutoff: float = 300.0,
    threshold_ratio: float = 0.1,
    f0_min: float = 50.0,
    f0_max: float = 500.0,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 2.0,
    min_voiced_frames: int = 3,
) -> dict[str, float]:
    """
    Extract feature points for a single voice segment (e.g., "ta" syllable).
    
    This function uses scientifically-grounded methods for detecting:
    
    Feature points:
    1. t_start: Consonant onset /t/ - Fujii 10% method with backward search + interpolation
    2. t_peak: Burst maximum of /t/ - High-frequency envelope maximum
    3. a_start: Vowel onset /a/ - Periodicity-based voicing detection
    4. a_peak: First stable periodic peak after a_start
    5. a_end: End of vowel /a/ (segment end)
    
    T_start (Fujii 10% method):
    - Compute high-frequency envelope (HPF → Hilbert envelope → smoothing)
    - Let E_peak = max envelope value within segment
    - Threshold = 0.1 * E_peak (10%)
    - Search backward from peak until envelope falls below threshold
    - Linear interpolation for exact crossing point
    
    A_start (Periodicity-based voicing detection):
    - Bandpass filter focusing on F0 band (50-500 Hz)
    - Short-time analysis with RMS and periodicity (autocorrelation)
    - Adaptive thresholds using median/MAD
    - A_start = first frame that is voiced and stays voiced for ≥3-5 frames
    
    A_peak:
    - First stable periodic peak after A_start
    - Low-pass filter to focus on F0 band
    - Find first local maximum after A_start
    
    Temporal ordering constraints:
    - t_start <= t_peak
    - t_start <= a_start
    - a_start <= a_peak <= a_end
    
    Args:
        y: Full audio signal.
        sr: Sampling rate in Hz.
        segment_start: Start sample index of the segment.
        segment_end: End sample index of the segment.
        hpf_cutoff: High-pass filter cutoff for T detection (Hz).
        threshold_ratio: Fujii threshold ratio (default: 0.1 = 10%).
        f0_min: Minimum F0 for voicing detection (Hz).
        f0_max: Maximum F0 for voicing detection (Hz).
        frame_length_ms: Frame length for voicing analysis (ms).
        hop_length_ms: Hop length for voicing analysis (ms).
        min_voiced_frames: Minimum consecutive voiced frames for A_start.
    
    Returns:
        Dictionary with feature point times in seconds:
        {'t_start': float, 't_peak': float, 'a_start': float, 'a_peak': float, 'a_end': float}
    """
    return ta_onset_analysis.extract_ta_feature_points(
        y, sr, segment_start, segment_end,
        hpf_cutoff=hpf_cutoff,
        threshold_ratio=threshold_ratio,
        f0_min=f0_min,
        f0_max=f0_max,
        frame_length_ms=frame_length_ms,
        hop_length_ms=hop_length_ms,
        min_voiced_frames=min_voiced_frames,
    )


def detect_voice_segments_with_features(
    wav_path: str,
    *,
    amplitude_threshold_ratio: float = AMPLITUDE_THRESHOLD_RATIO,
    min_segment_duration_sec: float = MIN_SEGMENT_DURATION_SEC,
    min_silence_duration_sec: float = MIN_SILENCE_DURATION_SEC,
) -> tuple[np.ndarray, int, list[dict[str, float]]]:
    """
    Detect voice segments and extract feature points from a WAV file.
    
    Args:
        wav_path: Path to the WAV file.
        amplitude_threshold_ratio: Threshold as fraction of max amplitude.
        min_segment_duration_sec: Minimum segment duration in seconds.
        min_silence_duration_sec: Minimum silence duration between segments.
    
    Returns:
        Tuple of (audio_signal, sample_rate, list_of_feature_dicts)
        Each feature dict contains: {'t_start', 't_peak', 'a_start', 'a_peak', 'a_end'}
    """
    # Load audio
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    
    # Detect voice segments
    segments = detect_voice_segments(
        y, sr,
        amplitude_threshold_ratio=amplitude_threshold_ratio,
        min_segment_duration_sec=min_segment_duration_sec,
        min_silence_duration_sec=min_silence_duration_sec,
    )
    
    # Extract feature points for each segment
    features_list = []
    for start, end in segments:
        features = extract_feature_points(y, sr, start, end)
        features_list.append(features)
    
    return y, sr, features_list


def plot_voice_segments_interactive(
    wav_path: str,
    y: np.ndarray,
    sr: int,
    features_list: list[dict[str, float]],
    *,
    initial_threshold_ratio: float = AMPLITUDE_THRESHOLD_RATIO,
    title: str = "",
) -> None:
    """
    Interactive plot of voice segments with feature points.
    
    Displays:
    - Amplitude waveform with segment markers
    - RMS envelope with feature point markers
    - Optional spectrogram overlay (PRAAT-style, toggle with button)
    
    Feature points are color-coded:
    - t_start (consonant onset): Green solid line
    - t_peak (burst peak): Red dashed line
    - a_start (vowel onset): Blue dotted line
    - a_peak (periodic peak): Cyan dash-dot line
    - a_end (segment end): Magenta solid line
    
    Args:
        wav_path: Path to the WAV file (for re-detection).
        y: Audio signal.
        sr: Sampling rate.
        features_list: List of feature dictionaries from detect_voice_segments_with_features.
        initial_threshold_ratio: Initial amplitude threshold ratio.
        title: Optional plot title.
    """
    # Compute RMS envelope
    frame_length = int(RMS_FRAME_LENGTH_MS * sr / 1000)
    hop_length = int(RMS_HOP_LENGTH_MS * sr / 1000)
    env = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop_length)
    
    # Compute spectrogram for overlay
    S_db, spec_times, spec_freqs = onset_hilbert.compute_spectrogram(y, sr)
    
    # Create figure
    fig = plt.figure(figsize=(14, 8))
    
    # Create subplots
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=2, sharex=ax1)
    
    # Create a secondary y-axis for frequency display (for spectrogram)
    ax1_spec = ax1.twinx()
    ax1_spec.set_ylabel('Frequency (Hz)')
    ax1_spec.set_ylim(spec_freqs[0], spec_freqs[-1])
    ax1_spec.set_visible(False)  # Initially hidden
    
    # State for spectrogram visibility
    spectrogram_state = {'visible': False, 'image': None}
    
    # Plot waveform
    time_axis = np.arange(len(y)) / sr
    waveform_line, = ax1.plot(time_axis, y, alpha=0.5, linewidth=0.5, color='blue', label='Waveform')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(title if title else f'Voice Segment Detection - {os.path.basename(wav_path)}')
    ax1.grid(True, alpha=0.3)
    
    # Plot RMS envelope
    ax2.plot(times, env, label='RMS Envelope', linewidth=1.5, color='orange')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('RMS Envelope')
    ax2.grid(True, alpha=0.3)
    
    # Store marker lines for updating
    marker_lines_ax1 = []
    marker_lines_ax2 = []
    
    # Define marker styles for each feature point
    marker_styles = {
        't_start': {'color': 'green', 'linestyle': '-', 'linewidth': 2, 'label': 't-start (consonant onset)'},
        't_peak': {'color': 'red', 'linestyle': '--', 'linewidth': 2, 'label': 't-peak (burst max)'},
        'a_start': {'color': 'blue', 'linestyle': ':', 'linewidth': 2, 'label': 'a-start (vowel onset)'},
        'a_peak': {'color': 'cyan', 'linestyle': '-.', 'linewidth': 2, 'label': 'a-peak (periodic peak)'},
        'a_end': {'color': 'magenta', 'linestyle': '-', 'linewidth': 2, 'label': 'a-end (segment end)'},
    }
    
    def draw_markers(features_list_to_draw):
        """Draw feature point markers on both axes."""
        # Clear existing markers
        for line in marker_lines_ax1:
            line.remove()
        marker_lines_ax1.clear()
        for line in marker_lines_ax2:
            line.remove()
        marker_lines_ax2.clear()
        
        # Track which labels have been added
        added_labels = set()
        
        # Draw markers for each segment
        for features in features_list_to_draw:
            for key, style in marker_styles.items():
                if key in features:
                    time_val = features[key]
                    label = style['label'] if key not in added_labels else None
                    
                    line1 = ax1.axvline(
                        x=time_val,
                        color=style['color'],
                        linestyle=style['linestyle'],
                        linewidth=style['linewidth'],
                        alpha=0.7,
                        label=label
                    )
                    marker_lines_ax1.append(line1)
                    
                    line2 = ax2.axvline(
                        x=time_val,
                        color=style['color'],
                        linestyle=style['linestyle'],
                        linewidth=style['linewidth'],
                        alpha=0.7
                    )
                    marker_lines_ax2.append(line2)
                    
                    added_labels.add(key)
        
        # Update legend
        ax1.legend(loc='upper right', fontsize=8)
    
    # Initial marker drawing
    draw_markers(features_list)
    
    # Status text
    status_text = ax1.text(
        0.02, 0.98,
        f'Segments detected: {len(features_list)}',
        transform=ax1.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10
    )
    
    # Add sliders
    plt.subplots_adjust(bottom=0.18)
    
    # Threshold slider (shortened to make room for buttons)
    ax_slider = plt.axes([0.15, 0.10, 0.30, 0.03])
    slider_threshold = Slider(
        ax_slider,
        'Threshold (%)',
        1.0,
        30.0,
        valinit=initial_threshold_ratio * 100,
        valstep=0.5
    )
    
    # Button layout - position buttons in a row
    button_width = 0.10
    button_height = 0.04
    button_y = 0.10
    button_start_x = 0.50
    button_spacing = 0.02
    
    # Re-detect button
    ax_button = plt.axes([button_start_x, button_y, button_width, button_height])
    button_redetect = Button(ax_button, 'Re-detect')
    
    # Spectrogram toggle button
    ax_spectrogram = plt.axes([button_start_x + button_width + button_spacing, button_y, button_width, button_height])
    button_spectrogram = Button(ax_spectrogram, 'Spectrogram')
    
    # Export button
    ax_export = plt.axes([button_start_x + 2 * (button_width + button_spacing), button_y, button_width, button_height])
    button_export = Button(ax_export, 'Export')
    
    def toggle_spectrogram(event):
        """Toggle spectrogram overlay visibility."""
        if spectrogram_state['visible']:
            # Hide spectrogram
            if spectrogram_state['image'] is not None:
                spectrogram_state['image'].remove()
                spectrogram_state['image'] = None
            ax1_spec.set_visible(False)
            waveform_line.set_alpha(0.5)
            spectrogram_state['visible'] = False
            button_spectrogram.label.set_text('Spectrogram')
        else:
            # Show spectrogram
            extent = [spec_times[0], spec_times[-1], spec_freqs[0], spec_freqs[-1]]
            
            # Display spectrogram on ax1 using imshow with transparency
            spectrogram_state['image'] = ax1.imshow(
                S_db,
                aspect='auto',
                origin='lower',
                extent=extent,
                cmap='magma',
                alpha=0.6,
                interpolation='bilinear',
                zorder=0  # Put behind the waveform
            )
            ax1_spec.set_visible(True)
            ax1_spec.set_ylim(spec_freqs[0], spec_freqs[-1])
            waveform_line.set_alpha(0.8)  # Make waveform more visible
            spectrogram_state['visible'] = True
            button_spectrogram.label.set_text('Hide Spec')
        
        fig.canvas.draw_idle()
    
    button_spectrogram.on_clicked(toggle_spectrogram)
    
    # Current state
    state = {
        'features_list': features_list,
        'threshold_ratio': initial_threshold_ratio,
    }
    
    def on_redetect(event):
        """Handle re-detect button click."""
        new_threshold = slider_threshold.val / 100.0
        state['threshold_ratio'] = new_threshold
        
        # Re-detect segments
        segments = detect_voice_segments(
            y, sr,
            amplitude_threshold_ratio=new_threshold,
        )
        
        # Re-extract features
        new_features_list = []
        for start, end in segments:
            features = extract_feature_points(y, sr, start, end)
            new_features_list.append(features)
        
        state['features_list'] = new_features_list
        
        # Update markers
        draw_markers(new_features_list)
        
        # Update status
        status_text.set_text(f'Segments detected: {len(new_features_list)}')
        
        fig.canvas.draw_idle()
    
    button_redetect.on_clicked(on_redetect)
    
    def on_export(event):
        """Handle export button click."""
        # Create a hidden tkinter root window for file dialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Get default filename
        default_name = os.path.splitext(os.path.basename(wav_path))[0] + '_ta_features.csv'
        
        # Show save file dialog
        file_path = filedialog.asksaveasfilename(
            title="Export TA Feature Points",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        root.destroy()
        
        if file_path:
            # Save data with error handling and retry
            success, actual_path = save_ta_features_csv_with_retry(
                file_path,
                state['features_list']
            )
            if success:
                print(f"TA features exported to: {actual_path}")
            else:
                print("Export cancelled.")
    
    button_export.on_clicked(on_export)
    
    # Add zoom functionality
    def on_scroll(event):
        """Handle mouse wheel scroll for X-axis zoom."""
        if event.inaxes not in (ax1, ax2):
            return
        
        ax = event.inaxes
        cur_xlim = ax.get_xlim()
        xdata = event.xdata
        
        if xdata is None:
            return
        
        zoom_factor = 1.2
        if event.button == 'up':
            scale_factor = 1 / zoom_factor
        elif event.button == 'down':
            scale_factor = zoom_factor
        else:
            return
        
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        new_xlim = [xdata - new_width * (1 - relx), xdata + new_width * relx]
        ax.set_xlim(new_xlim)
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    plt.show()


class OnsetDetectionGUI:
    """Main GUI application for onset detection."""
    
    def __init__(self, root: tk.Tk):
        """Initialize the GUI application.
        
        Args:
            root: The root tkinter window.
        """
        self.root = root
        self.root.title("Onset Detection Tool")
        self.root.geometry("800x600")
        
        # Create main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for responsiveness
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title label
        title_label = ttk.Label(
            main_frame, 
            text="Onset Detection Tool",
            font=('Helvetica', 16, 'bold')
        )
        title_label.grid(row=0, column=0, pady=20)
        
        # Description label
        desc_label = ttk.Label(
            main_frame,
            text="Select an onset detection method and choose your audio file(s)",
            wraplength=700
        )
        desc_label.grid(row=1, column=0, pady=10)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=20)
        
        # Tap detection button
        self.tap_button = ttk.Button(
            button_frame,
            text="Detect Tap Onsets",
            command=self.detect_tap_onsets,
            width=30
        )
        self.tap_button.grid(row=0, column=0, pady=10, padx=10)
        
        # Voice segment detection button (for "ta" syllables with feature points)
        self.voice_segment_button = ttk.Button(
            button_frame,
            text="Detect Voice Segments (ta)",
            command=self.detect_voice_segments,
            width=30
        )
        self.voice_segment_button.grid(row=1, column=0, pady=10, padx=10)
        
        # Status label
        self.status_label = ttk.Label(
            main_frame,
            text="Ready",
            font=('Helvetica', 10),
            foreground='green'
        )
        self.status_label.grid(row=3, column=0, pady=10)
        
        # Results text area
        result_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        result_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        main_frame.rowconfigure(4, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        # Text widget with scrollbar
        self.result_text = tk.Text(result_frame, height=10, wrap=tk.WORD)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text['yscrollcommand'] = scrollbar.set
    
    def update_status(self, message: str, color: str = 'black'):
        """Update the status label.
        
        Args:
            message: Status message to display.
            color: Text color (default: black).
        """
        self.status_label.config(text=message, foreground=color)
        self.root.update_idletasks()
    
    def append_result(self, text: str):
        """Append text to the results area.
        
        Args:
            text: Text to append.
        """
        self.result_text.insert(tk.END, text + "\n")
        self.result_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_results(self):
        """Clear the results text area."""
        self.result_text.delete(1.0, tk.END)
    
    def detect_tap_onsets(self):
        """Handle tap onset detection with file selection."""
        self.clear_results()
        self.update_status("Select WAV file(s) for tap detection...", 'blue')
        
        # Open file dialog for WAV file(s) - now supports multiple selection
        wav_paths = filedialog.askopenfilenames(
            title="Select WAV file(s) for tap detection",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if not wav_paths:
            self.update_status("Selection cancelled", 'orange')
            return
        
        # Convert to list
        wav_paths = list(wav_paths)
        
        # Process files starting with the first one
        self._process_tap_files(wav_paths, 0)
    
    def _process_tap_files(self, wav_paths, current_index):
        """Process tap onset detection for a specific file in the list."""
        if current_index >= len(wav_paths):
            self.update_status("All files processed!", 'green')
            return
        
        wav_path = wav_paths[current_index]
        
        try:
            self.clear_results()
            self.update_status("Processing...", 'blue')
            self.append_result(f"Processing file {current_index + 1} of {len(wav_paths)}: {os.path.basename(wav_path)}")
            self.append_result("=" * 60)
            
            # Initial detection with default parameters using Fujii method
            hp_cutoff = 300.0
            threshold_ratio = 0.1
            min_distance_ms = 100.0
            
            onset_times, peak_times = onset_hilbert.detect_tap_onsets_and_peaks(
                wav_path,
                hp_cutoff_hz=hp_cutoff,
                threshold_ratio=threshold_ratio,
                min_distance_ms=min_distance_ms
            )
            
            # Display results
            self.append_result(f"\nDetected {len(onset_times)} tap onsets using Fujii method:")
            for i, (ot, pt) in enumerate(zip(onset_times, peak_times), 1):
                self.append_result(f"  {i}. onset={ot:.3f}s, peak={pt:.3f}s")
            
            # Plot results with interactive controls using Fujii method
            self.append_result("\nGenerating interactive plot...")
            self.append_result("Use the slider to adjust HPF frequency and click 'Re-detect' to update.")
            self.append_result("Cmd+Shift+Click to delete onset/peak markers.")
            self.append_result("Re-detection uses Fujii method (10% threshold, backward search, linear interpolation).")
            
            # Callback for next file navigation
            def on_next_file():
                if current_index + 1 < len(wav_paths):
                    # Process next file in the list
                    self._process_tap_files(wav_paths, current_index + 1)
                else:
                    # Show file dialog for new file selection
                    self.detect_tap_onsets()
            
            onset_hilbert.plot_waveform_and_envelope_interactive(
                wav_path,
                initial_hp_cutoff_hz=hp_cutoff,
                is_click=False,
                threshold_ratio=threshold_ratio,
                min_distance_ms=min_distance_ms,
                title=f"Tap Onset Detection (Fujii Method) - {os.path.basename(wav_path)} ({current_index + 1}/{len(wav_paths)})",
                on_next_callback=on_next_file,
                enable_marker_deletion=True,
                enable_export=True
            )
            
            self.update_status("Detection complete!", 'green')
            self.append_result("\n" + "=" * 60)
            self.append_result("Plot window opened. Close it to continue.")
            
        except Exception as e:
            error_msg = f"Error during tap detection: {str(e)}"
            self.append_result(f"\nERROR: {error_msg}")
            self.update_status("Error occurred", 'red')
            messagebox.showerror("Error", error_msg)
    
    def detect_voice_segments(self):
        """Handle voice segment detection with feature point extraction.
        
        This method detects voice segments (e.g., "ta" syllables) from audio
        and extracts feature points for each segment:
        - t_start: consonant onset (Fujii 10% method)
        - t_peak: burst maximum point
        - a_start: vowel onset (periodicity-based voicing detection)
        - a_peak: first stable periodic peak
        - a_end: segment end
        """
        self.clear_results()
        self.update_status("Select WAV file for voice segment detection...", 'blue')
        
        # Open file dialog for WAV file
        wav_path = filedialog.askopenfilename(
            title="Select WAV file for voice segment detection",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if not wav_path:
            self.update_status("Selection cancelled", 'orange')
            return
        
        try:
            self.update_status("Processing...", 'blue')
            self.append_result(f"Processing file: {os.path.basename(wav_path)}")
            self.append_result("=" * 60)
            self.append_result("Detecting voice segments based on amplitude threshold...")
            self.root.update_idletasks()
            
            # Detect voice segments with feature points
            y, sr, features_list = detect_voice_segments_with_features(
                wav_path,
                amplitude_threshold_ratio=AMPLITUDE_THRESHOLD_RATIO,
                min_segment_duration_sec=MIN_SEGMENT_DURATION_SEC,
                min_silence_duration_sec=MIN_SILENCE_DURATION_SEC,
            )
            
            # Display results
            self.append_result(f"\nDetected {len(features_list)} voice segments:")
            self.append_result("")
            self.append_result("Feature points for each segment:")
            self.append_result("  - t_start: consonant onset (Fujii 10% method)")
            self.append_result("  - t_peak: burst maximum point")
            self.append_result("  - a_start: vowel onset (periodicity-based)")
            self.append_result("  - a_peak: first stable periodic peak")
            self.append_result("  - a_end: segment end")
            self.append_result("")
            
            for i, features in enumerate(features_list, 1):
                self.append_result(f"Segment {i}:")
                self.append_result(f"  t_start: {features['t_start']:.3f}s")
                self.append_result(f"  t_peak:  {features['t_peak']:.3f}s")
                self.append_result(f"  a_start: {features['a_start']:.3f}s")
                self.append_result(f"  a_peak:  {features['a_peak']:.3f}s")
                self.append_result(f"  a_end:   {features['a_end']:.3f}s")
                self.append_result("")
            
            # Plot results with interactive controls
            self.append_result("Generating interactive plot...")
            self.append_result("Feature point marker colors:")
            self.append_result("  Green (solid):      t_start (consonant onset)")
            self.append_result("  Red (dashed):       t_peak (burst maximum)")
            self.append_result("  Blue (dotted):      a_start (vowel onset)")
            self.append_result("  Cyan (dash-dot):    a_peak (periodic peak)")
            self.append_result("  Magenta (solid):    a_end (segment end)")
            self.append_result("")
            self.append_result("Use the slider to adjust threshold and click 'Re-detect' to update.")
            
            plot_voice_segments_interactive(
                wav_path, y, sr, features_list,
                initial_threshold_ratio=AMPLITUDE_THRESHOLD_RATIO,
                title=f"Voice Segment Detection - {os.path.basename(wav_path)}"
            )
            
            self.update_status("Detection complete!", 'green')
            self.append_result("\n" + "=" * 60)
            self.append_result("Plot window opened. Close it to continue.")
            
        except Exception as e:
            error_msg = f"Error during voice segment detection: {str(e)}"
            self.append_result(f"\nERROR: {error_msg}")
            self.update_status("Error occurred", 'red')
            messagebox.showerror("Error", error_msg)
    



def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = OnsetDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
