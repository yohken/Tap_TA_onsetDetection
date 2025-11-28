"""
GUI application for onset detection with file selection and result plotting.

This module provides a graphical user interface for:
1. Tap onset detection from WAV files
2. Voice segment detection (e.g., "ta" syllables) with feature point extraction

Features:
- File selection dialogs for audio files
- Voice segment detection based on amplitude/RMS threshold
- Feature point extraction for each voice segment:
  - t-start: segment start (consonant onset)
  - t-peak: burst maximum point
  - a-start: transition from t to a (vowel onset)
  - a-stable: vowel stabilization point
  - end: segment end
- Automatic visualization of detection results with color-coded markers
- User-friendly interface using tkinter

Target Python version: 3.10+
Dependencies: tkinter (standard library), onset_detection module,
              librosa, numpy, scipy, matplotlib
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
import os
import json
import numpy as np
import scipy.signal
import pandas as pd
from datetime import datetime

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

# Threshold ratio for t-peak detection (fraction of local max)
T_PEAK_THRESHOLD_RATIO = 0.9  # 90% of local max

# Threshold for a-start detection (ratio of RMS increase from t-peak)
A_START_RMS_INCREASE_RATIO = 0.3  # 30% RMS increase indicates vowel onset

# Threshold for a-stable detection (RMS variance stability)
A_STABLE_VARIANCE_THRESHOLD = 0.1  # 10% variance indicates stability

# Fraction of segment to search for t-peak (consonant burst)
# We search only the first portion of the segment since consonant should be early
CONSONANT_SEARCH_FRACTION = 0.33  # Search first 1/3 of segment

# Small time offset (in seconds) to ensure temporal ordering of feature points
# Used when calculated values would otherwise violate ordering constraints
FEATURE_POINT_OFFSET_SEC = 0.01  # 10ms offset

# ==============================================================================


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
    t_peak_threshold_ratio: float = T_PEAK_THRESHOLD_RATIO,
    a_start_rms_increase_ratio: float = A_START_RMS_INCREASE_RATIO,
    a_stable_variance_threshold: float = A_STABLE_VARIANCE_THRESHOLD,
) -> dict[str, float]:
    """
    Extract feature points for a single voice segment (e.g., "ta" syllable).
    
    Feature points:
    - t_start: Start of segment (consonant onset)
    - t_peak: Burst maximum point (consonant peak)
    - a_start: Transition from consonant to vowel (vowel onset)
    - a_stable: Point where vowel becomes stable
    - end: End of segment
    
    Temporal ordering constraints:
    - t_start <= t_peak (t_peak is always within T segment)
    - t_start <= a_start (a_start is valid if after T segment start)
    - a_start <= a_stable <= end
    
    Note: a_start can be before t_peak as long as it's after t_start.
    This allows detection of A onset within the T segment boundary.
    
    Args:
        y: Full audio signal.
        sr: Sampling rate in Hz.
        segment_start: Start sample index of the segment.
        segment_end: End sample index of the segment.
        t_peak_threshold_ratio: Threshold for t-peak detection (default: 0.9).
        a_start_rms_increase_ratio: RMS increase ratio for a-start detection (default: 0.3).
        a_stable_variance_threshold: Variance threshold for a-stable detection (default: 0.1).
    
    Returns:
        Dictionary with feature point times in seconds:
        {'t_start', 't_peak', 'a_start', 'a_stable', 'end'}
    """
    # Extract segment
    segment = y[segment_start:segment_end]
    if len(segment) == 0:
        t_start_sec = segment_start / sr
        return {
            't_start': t_start_sec,
            't_peak': t_start_sec,
            'a_start': t_start_sec,
            'a_stable': t_start_sec,
            'end': t_start_sec,
        }
    
    # Compute RMS envelope for the segment
    frame_length = max(int(RMS_FRAME_LENGTH_MS * sr / 1000), 2)
    hop_length = max(int(RMS_HOP_LENGTH_MS * sr / 1000), 1)
    
    env = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
    
    if len(env) == 0:
        t_start_sec = segment_start / sr
        return {
            't_start': t_start_sec,
            't_peak': t_start_sec,
            'a_start': t_start_sec,
            'a_stable': t_start_sec,
            'end': segment_end / sr,
        }
    
    # Feature point 1: t_start (segment start)
    t_start_sec = segment_start / sr
    
    # Feature point 5: end (segment end)
    end_sec = segment_end / sr
    
    # Feature point 2: t_peak (burst maximum)
    # Find the first significant peak in the envelope (consonant burst)
    # Look for peaks in the first portion of the segment (where consonant should be)
    search_end = max(int(len(env) * CONSONANT_SEARCH_FRACTION), 1)
    first_part = env[:search_end]
    
    if len(first_part) > 0:
        t_peak_frame = np.argmax(first_part)
        t_peak_sample = segment_start + t_peak_frame * hop_length
        t_peak_sec = t_peak_sample / sr
    else:
        t_peak_frame = 0
        t_peak_sec = t_start_sec
    
    # Feature point 3: a_start (vowel onset)
    # Look for the point after t_peak where RMS starts increasing significantly
    # This indicates the transition from consonant to vowel
    
    # Start searching after t_peak
    search_start = t_peak_frame + 1
    a_start_frame = search_start
    
    if search_start < len(env):
        # Find the minimum after t_peak (dip between consonant and vowel)
        remaining = env[search_start:]
        if len(remaining) > 0:
            # Find local minimum
            min_idx = np.argmin(remaining[:len(remaining)//2]) if len(remaining) > 2 else 0
            
            # a_start is after the minimum where RMS starts rising
            a_start_frame = search_start + min_idx
            
            # Find where RMS starts increasing significantly
            for i in range(a_start_frame, len(env) - 1):
                if env[i + 1] > env[i] * (1 + a_start_rms_increase_ratio * 0.1):
                    a_start_frame = i
                    break
    
    a_start_sample = segment_start + a_start_frame * hop_length
    a_start_sec = a_start_sample / sr
    
    # Ensure a_start is after t_start (T segment start)
    # A start is valid as long as it's within the T segment (i.e., after t_start_sec)
    # Even if a_start is before t_peak, it's accepted if it's after t_start
    if a_start_sec <= t_start_sec:
        a_start_sec = t_start_sec + FEATURE_POINT_OFFSET_SEC  # Add small offset
    
    # Feature point 4: a_stable (vowel stabilization)
    # Look for the point where RMS variance becomes low (stable vowel)
    
    # Start from a_start and look for stability
    stability_window = max(int(0.02 * sr / hop_length), 3)  # 20ms window
    a_stable_frame = a_start_frame
    
    if a_start_frame + stability_window < len(env):
        for i in range(a_start_frame, len(env) - stability_window):
            window = env[i:i + stability_window]
            if len(window) > 0:
                mean_val = np.mean(window)
                if mean_val > 0:
                    variance = np.var(window) / (mean_val ** 2)
                    if variance < a_stable_variance_threshold:
                        a_stable_frame = i
                        break
        else:
            # If no stable point found, use middle of remaining segment
            a_stable_frame = (a_start_frame + len(env)) // 2
    
    a_stable_sample = segment_start + a_stable_frame * hop_length
    a_stable_sec = a_stable_sample / sr
    
    # Ensure a_stable is after a_start
    if a_stable_sec <= a_start_sec:
        a_stable_sec = a_start_sec + FEATURE_POINT_OFFSET_SEC  # Add small offset
    
    # Ensure a_stable is before end
    if a_stable_sec >= end_sec:
        a_stable_sec = (a_start_sec + end_sec) / 2
    
    return {
        't_start': t_start_sec,
        't_peak': t_peak_sec,
        'a_start': a_start_sec,
        'a_stable': a_stable_sec,
        'end': end_sec,
    }


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
        Each feature dict contains: {'t_start', 't_peak', 'a_start', 'a_stable', 'end'}
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
    
    Feature points are color-coded:
    - t_start (segment start): Green solid line
    - t_peak (burst peak): Red dashed line
    - a_start (vowel onset): Blue dotted line
    - a_stable (vowel stable): Cyan dash-dot line
    - end (segment end): Magenta solid line
    
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
    
    # Create figure
    fig = plt.figure(figsize=(14, 8))
    
    # Create subplots
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=2, sharex=ax1)
    
    # Plot waveform
    time_axis = np.arange(len(y)) / sr
    ax1.plot(time_axis, y, alpha=0.5, linewidth=0.5, color='blue', label='Waveform')
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
        't_start': {'color': 'green', 'linestyle': '-', 'linewidth': 2, 'label': 't-start (segment start)'},
        't_peak': {'color': 'red', 'linestyle': '--', 'linewidth': 2, 'label': 't-peak (burst max)'},
        'a_start': {'color': 'blue', 'linestyle': ':', 'linewidth': 2, 'label': 'a-start (vowel onset)'},
        'a_stable': {'color': 'cyan', 'linestyle': '-.', 'linewidth': 2, 'label': 'a-stable (vowel stable)'},
        'end': {'color': 'magenta', 'linestyle': '-', 'linewidth': 2, 'label': 'end (segment end)'},
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
    
    # Add sliders and buttons
    plt.subplots_adjust(bottom=0.22)
    
    # Threshold slider
    ax_slider = plt.axes([0.15, 0.14, 0.50, 0.03])
    slider_threshold = Slider(
        ax_slider,
        'Threshold (%)',
        1.0,
        30.0,
        valinit=initial_threshold_ratio * 100,
        valstep=0.5
    )
    
    # Button positions (two rows)
    button_width = 0.12
    button_height = 0.04
    button_y_row1 = 0.08  # First row
    button_y_row2 = 0.02  # Second row
    
    # Re-detect button
    ax_button_redetect = plt.axes([0.15, button_y_row1, button_width, button_height])
    button_redetect = Button(ax_button_redetect, 'Re-detect')
    
    # CSV export button
    ax_button_csv = plt.axes([0.29, button_y_row1, button_width, button_height])
    button_csv = Button(ax_button_csv, 'CSV書き出し')
    
    # Image export button
    ax_button_image = plt.axes([0.43, button_y_row1, button_width, button_height])
    button_image = Button(ax_button_image, '画像書き出し')
    
    # JSON data export button (for reproducibility)
    ax_button_json = plt.axes([0.57, button_y_row1, button_width + 0.04, button_height])
    button_json = Button(ax_button_json, 'データ書き出し（再現用）')
    
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
    
    def on_export_csv(event):
        """Handle CSV export button click."""
        # Create a hidden tkinter root window for file dialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Get default filename
        default_name = os.path.splitext(os.path.basename(wav_path))[0] + '_voice_segments.csv'
        
        # Show save file dialog
        file_path = filedialog.asksaveasfilename(
            title="CSV書き出し先を選択",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        root.destroy()
        
        if file_path:
            try:
                # Build DataFrame from features
                rows = []
                for i, features in enumerate(state['features_list']):
                    row = {
                        'segment_index': i,
                        't_start_sec': features['t_start'],
                        't_peak_sec': features['t_peak'],
                        'a_start_sec': features['a_start'],
                        'a_stable_sec': features['a_stable'],
                        'end_sec': features['end'],
                    }
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_csv(file_path, index=False)
                print(f"CSV exported to: {file_path}")
            except Exception as e:
                messagebox.showerror("エラー", f"CSV書き出しに失敗しました: {str(e)}")
    
    button_csv.on_clicked(on_export_csv)
    
    def on_export_image(event):
        """Handle image export button click."""
        # Create a hidden tkinter root window for file dialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Get default filename
        default_name = os.path.splitext(os.path.basename(wav_path))[0] + '_plot.png'
        
        # Show save file dialog
        file_path = filedialog.asksaveasfilename(
            title="画像書き出し先を選択",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("All files", "*.*")]
        )
        
        root.destroy()
        
        if file_path:
            try:
                fig.savefig(file_path, dpi=150, bbox_inches='tight')
                print(f"Image exported to: {file_path}")
            except Exception as e:
                messagebox.showerror("エラー", f"画像書き出しに失敗しました: {str(e)}")
    
    button_image.on_clicked(on_export_image)
    
    def on_export_json(event):
        """Handle JSON data export button click (for reproducibility)."""
        # Create a hidden tkinter root window for file dialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Get default filename
        default_name = os.path.splitext(os.path.basename(wav_path))[0] + '_data.json'
        
        # Show save file dialog
        file_path = filedialog.asksaveasfilename(
            title="データ書き出し先を選択（再現用）",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        root.destroy()
        
        if file_path:
            try:
                # Build data dictionary for reproducibility
                export_data = {
                    'metadata': {
                        'source_file': os.path.basename(wav_path),
                        'source_path': wav_path,
                        'sample_rate': sr,
                        'audio_duration_sec': len(y) / sr,
                        'export_timestamp': datetime.now().isoformat(),
                    },
                    'parameters': {
                        'amplitude_threshold_ratio': state['threshold_ratio'],
                        'min_segment_duration_sec': MIN_SEGMENT_DURATION_SEC,
                        'min_silence_duration_sec': MIN_SILENCE_DURATION_SEC,
                        'rms_frame_length_ms': RMS_FRAME_LENGTH_MS,
                        'rms_hop_length_ms': RMS_HOP_LENGTH_MS,
                    },
                    'segments': state['features_list'],
                    'segment_count': len(state['features_list']),
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                print(f"JSON data exported to: {file_path}")
            except Exception as e:
                messagebox.showerror("エラー", f"データ書き出しに失敗しました: {str(e)}")
    
    button_json.on_clicked(on_export_json)
    
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
        - t_start: segment start (consonant onset)
        - t_peak: burst maximum point
        - a_start: transition from consonant to vowel
        - a_stable: vowel stabilization point
        - end: segment end
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
            self.append_result("  - t_start: segment start (consonant onset)")
            self.append_result("  - t_peak: burst maximum point")
            self.append_result("  - a_start: vowel onset (transition from t to a)")
            self.append_result("  - a_stable: vowel stabilization point")
            self.append_result("  - end: segment end")
            self.append_result("")
            
            for i, features in enumerate(features_list, 1):
                self.append_result(f"Segment {i}:")
                self.append_result(f"  t_start:  {features['t_start']:.3f}s")
                self.append_result(f"  t_peak:   {features['t_peak']:.3f}s")
                self.append_result(f"  a_start:  {features['a_start']:.3f}s")
                self.append_result(f"  a_stable: {features['a_stable']:.3f}s")
                self.append_result(f"  end:      {features['end']:.3f}s")
                self.append_result("")
            
            # Plot results with interactive controls
            self.append_result("Generating interactive plot...")
            self.append_result("Feature point marker colors:")
            self.append_result("  Green (solid):      t_start (segment start)")
            self.append_result("  Red (dashed):       t_peak (burst maximum)")
            self.append_result("  Blue (dotted):      a_start (vowel onset)")
            self.append_result("  Cyan (dash-dot):    a_stable (vowel stable)")
            self.append_result("  Magenta (solid):    end (segment end)")
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
