"""
Unified onset + peak detection module for metronome clicks and finger taps.

This module implements a Fujii-style onset detection method using:
- High-pass filter (configurable cutoff)
- Hilbert envelope
- Per-event local peak detection
- Onset defined as when envelope first exceeds 10% of local peak amplitude

Target Python version: 3.10+
Dependencies: numpy, scipy, soundfile, matplotlib, pandas, tkinter
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd


def highpass_filter(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float | None,
    order: int = 4,
) -> np.ndarray:
    """
    Optionally apply a zero-phase high-pass filter to y.

    Args:
        y: mono audio signal, shape (N,).
        sr: sampling rate [Hz].
        cutoff_hz: high-pass cutoff in Hz.
                   If None or <= 0, return y unchanged.
        order: Butterworth filter order.

    Returns:
        y_filt: filtered signal, same shape as y.
    """
    if cutoff_hz is None or cutoff_hz <= 0:
        return y
    
    # Design Butterworth high-pass filter
    sos = butter(order, cutoff_hz, btype='hp', fs=sr, output='sos')
    
    # Apply zero-phase filtering
    y_filt = sosfiltfilt(sos, y)
    
    return y_filt


def hilbert_envelope(
    y: np.ndarray,
    sr: int,
    smooth_ms: float | None = 0.5,
) -> np.ndarray:
    """
    Compute the Hilbert envelope of y.

    Steps:
        1) analytic = hilbert(y)
        2) env = np.abs(analytic)
        3) optionally smooth env with a short moving-average window.

    Args:
        y: mono audio signal (after any desired filtering).
        sr: sampling rate [Hz].
        smooth_ms: length of moving-average in milliseconds.
                   If None or <= 0, no smoothing.

    Returns:
        env: 1D envelope array, same length as y.
    """
    # Compute Hilbert transform and envelope
    analytic = hilbert(y)
    env = np.abs(analytic)
    
    # Optionally smooth with moving average
    if smooth_ms is not None and smooth_ms > 0:
        win = int(round(smooth_ms * 1e-3 * sr))
        if win >= 2:
            kernel = np.ones(win) / win
            env = np.convolve(env, kernel, mode='same')
    
    return env


def detect_onsets_and_peaks_from_envelope(
    env: np.ndarray,
    sr: int,
    *,
    threshold_ratio: float = 0.1,
    min_distance_ms: float = 100.0,
    global_min_height_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect onsets and peaks from a Hilbert envelope.

    Algorithm:
        1) Find candidate peaks using scipy.signal.find_peaks:
             - height >= global_min_height_ratio * env.max()
             - distance >= min_distance_ms (converted to samples)
        2) For each peak index p:
             - Amax = env[p]
             - th = threshold_ratio * Amax  (10% by default)
             - Search backward from p to find the last index k where:
                   env[k] <= th and env[k+1] > th
               (threshold crossing from below to above).
             - If such k is found, linearly interpolate between k and k+1
               to get a fractional sample index for onset.
             - If not found, fall back to p (or the earliest sample).
        3) Convert onset and peak sample indices to time [s]:
             onset_times = onset_indices / sr
             peak_times  = peak_indices  / sr

    Args:
        env: Hilbert envelope, 1D array of length N.
        sr: sampling rate [Hz].
        threshold_ratio: local onset threshold as a fraction of the local peak
                         amplitude (default 0.1 = 10%).
        min_distance_ms: minimum distance between peaks in milliseconds, used
                         to avoid detecting multiple peaks for the same event.
        global_min_height_ratio:
             global height threshold relative to env.max() to ignore very
             small peaks (e.g., noise).

    Returns:
        onset_times: 1D array of onset times [s] for each detected event.
        peak_times:  1D array of corresponding peak times [s].
    """
    if len(env) == 0:
        return np.array([]), np.array([])
    
    # Convert min_distance_ms to samples
    min_distance_samples = int(round(min_distance_ms * 1e-3 * sr))
    
    # Find peaks in envelope
    max_env = np.max(env)
    if max_env == 0:
        return np.array([]), np.array([])
    
    peaks, _ = find_peaks(
        env,
        height=global_min_height_ratio * max_env,
        distance=min_distance_samples
    )
    
    if len(peaks) == 0:
        return np.array([]), np.array([])
    
    # For each peak, find the onset point
    onset_indices = []
    peak_indices = []
    
    for p in peaks:
        Amax = env[p]
        th = threshold_ratio * Amax
        
        # Search backward from peak to find threshold crossing
        k = p - 1
        while k > 0 and env[k] > th:
            k -= 1
        
        # Interpolate onset position
        if k <= 0:
            # No crossing found, use peak position
            onset_idx = float(p)
        else:
            # Found crossing between k and k+1
            e0 = env[k]
            e1 = env[k + 1]
            
            if e1 <= e0:
                # Edge case: no rise, use k+1
                onset_idx = float(k + 1)
            else:
                # Linear interpolation
                alpha = (th - e0) / (e1 - e0)
                onset_idx = k + alpha
        
        onset_indices.append(onset_idx)
        peak_indices.append(float(p))
    
    # Convert to numpy arrays and times
    onset_indices = np.array(onset_indices)
    peak_indices = np.array(peak_indices)
    
    onset_times = onset_indices / sr
    peak_times = peak_indices / sr
    
    return onset_times, peak_times


def detect_click_onsets_and_peaks(
    wav_path: str,
    *,
    hp_cutoff_hz: float | None = 1000.0,
    threshold_ratio: float = 0.1,
    min_distance_ms: float = 100.0,
    smooth_ms: float | None = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect onsets and peaks for a recorded metronome click track.

    Steps:
        1) Load WAV (mono) with soundfile.read.
        2) Optionally apply highpass_filter with hp_cutoff_hz.
        3) Compute Hilbert envelope with hilbert_envelope().
        4) Run detect_onsets_and_peaks_from_envelope().

    Args:
        wav_path: path to a mono WAV file (click track recording).
        hp_cutoff_hz: high-pass cutoff in Hz (e.g., 1000 Hz by default).
        threshold_ratio, min_distance_ms, smooth_ms:
            passed through to the core functions.

    Returns:
        onset_times: 1D array of onset times [s].
        peak_times:  1D array of peak times [s].
    """
    # Load WAV file
    data, sr = sf.read(wav_path)
    
    # Convert to mono if needed
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    
    # Apply high-pass filter
    y_filt = highpass_filter(data, sr, hp_cutoff_hz)
    
    # Compute Hilbert envelope
    env = hilbert_envelope(y_filt, sr, smooth_ms=smooth_ms)
    
    # Detect onsets and peaks
    onset_times, peak_times = detect_onsets_and_peaks_from_envelope(
        env, sr,
        threshold_ratio=threshold_ratio,
        min_distance_ms=min_distance_ms
    )
    
    return onset_times, peak_times


def detect_tap_onsets_and_peaks(
    wav_path: str,
    *,
    hp_cutoff_hz: float | None = 300.0,
    threshold_ratio: float = 0.1,
    min_distance_ms: float = 100.0,
    smooth_ms: float | None = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect onsets and peaks for finger taps, using the same Hilbert + 10% method
    as for clicks, but with a possibly different HPF cutoff.

    Args:
        wav_path: path to a mono WAV file containing finger taps.
        hp_cutoff_hz: high-pass cutoff in Hz (e.g., 300 Hz by default).
        threshold_ratio, min_distance_ms, smooth_ms:
            passed through to the core functions.

    Returns:
        onset_times: 1D array of onset times [s].
        peak_times:  1D array of peak times [s].
    """
    # Load WAV file
    data, sr = sf.read(wav_path)
    
    # Convert to mono if needed
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    
    # Apply high-pass filter
    y_filt = highpass_filter(data, sr, hp_cutoff_hz)
    
    # Compute Hilbert envelope
    env = hilbert_envelope(y_filt, sr, smooth_ms=smooth_ms)
    
    # Detect onsets and peaks
    onset_times, peak_times = detect_onsets_and_peaks_from_envelope(
        env, sr,
        threshold_ratio=threshold_ratio,
        min_distance_ms=min_distance_ms
    )
    
    return onset_times, peak_times


def save_onsets_and_peaks_csv(
    out_path: str,
    onset_times: np.ndarray,
    peak_times: np.ndarray,
    *,
    label: str | None = None,
) -> None:
    """
    Save detected onset and peak times to a CSV file.

    Columns:
        index: integer index (0, 1, 2, ...)
        onset_sec: onset times [s]
        peak_sec:  peak times [s]
        label: optional string label repeated for each row

    Args:
        out_path: path to CSV file.
        onset_times: 1D array of onset times [s].
        peak_times:  1D array of peak times [s].
        label: optional label (e.g., "click" or "tap").
    """
    # Ensure arrays have same length
    if len(onset_times) != len(peak_times):
        raise ValueError(
            f"onset_times and peak_times must have the same length "
            f"(got {len(onset_times)} and {len(peak_times)})"
        )
    
    # Build DataFrame
    data_dict = {
        'index': np.arange(len(onset_times)),
        'onset_sec': onset_times,
        'peak_sec': peak_times,
    }
    
    if label is not None:
        data_dict['label'] = [label] * len(onset_times)
    
    df = pd.DataFrame(data_dict)
    
    # Save to CSV
    df.to_csv(out_path, index=False)


def plot_waveform_and_envelope(
    y: np.ndarray,
    sr: int,
    env: np.ndarray,
    onset_times: np.ndarray,
    peak_times: np.ndarray,
    *,
    hp_cutoff_hz: float | None = None,
    title: str = "",
) -> None:
    """
    Plot:
        - the time-domain waveform
        - the Hilbert envelope
        - vertical lines at detected onsets and peaks

    Legend MUST clearly distinguish:
        - "Waveform"
        - "Hilbert envelope (HPF = XXX Hz)"  or "Hilbert envelope (no HPF)"
        - "Onsets"
        - "Peaks"

    Args:
        y: mono audio signal (after any filtering), shape (N,).
        sr: sampling rate [Hz].
        env: Hilbert envelope, same length as y.
        onset_times: 1D array of onset times [s].
        peak_times:  1D array of peak times [s].
        hp_cutoff_hz: HPF cutoff used (for labeling). If None or <= 0,
                      show "no HPF" in the legend.
        title: optional title string.
    """
    # Create time axis
    t = np.arange(len(y)) / sr
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot waveform
    plt.plot(t, y, label="Waveform", alpha=0.7, linewidth=0.5)
    
    # Plot envelope with appropriate label
    if hp_cutoff_hz is not None and hp_cutoff_hz > 0:
        env_label = f"Hilbert envelope (HPF = {hp_cutoff_hz:.0f} Hz)"
    else:
        env_label = "Hilbert envelope (no HPF)"
    plt.plot(t, env, label=env_label, linewidth=1.5)
    
    # Plot onset markers
    for i, ot in enumerate(onset_times):
        plt.axvline(ot, color='g', linestyle='--', alpha=0.7, 
                   label='Onsets' if i == 0 else None)
    
    # Plot peak markers
    for i, pt in enumerate(peak_times):
        plt.axvline(pt, color='r', linestyle=':', alpha=0.7, 
                   label='Peaks' if i == 0 else None)
    
    # Labels and legend
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude / Envelope')
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_waveform_and_envelope_interactive(
    wav_path: str,
    *,
    initial_hp_cutoff_hz: float | None = 300.0,
    is_click: bool = False,
    threshold_ratio: float = 0.1,
    min_distance_ms: float = 100.0,
    smooth_ms: float | None = 0.5,
    title: str = "",
) -> None:
    """
    Interactive plotting with HPF slider and re-detect button using Fujii method.
    
    This function provides a GUI with:
    - Interactive slider to adjust HPF cutoff frequency (100-2000 Hz)
    - Re-detect button to recompute onsets with new HPF frequency
    - X-axis zoom using mouse wheel
    
    The re-detection ALWAYS uses the Fujii method:
    1. Apply highpass_filter with new cutoff
    2. Compute hilbert_envelope
    3. Run detect_onsets_and_peaks_from_envelope (10% threshold, backward search, linear interpolation)
    
    Args:
        wav_path: path to a mono WAV file (click or tap).
        initial_hp_cutoff_hz: starting HPF cutoff in Hz
                              (e.g., 1000 Hz for clicks, 300 Hz for taps).
        is_click: if True, uses different default cutoff.
        threshold_ratio: onset threshold as fraction of local peak (default 0.1 = 10%).
        min_distance_ms: minimum distance between peaks in milliseconds.
        smooth_ms: envelope smoothing window in milliseconds.
        title: optional plot title.
    """
    from matplotlib.widgets import Slider, Button
    
    # Set initial cutoff based on is_click if not specified
    if initial_hp_cutoff_hz is None:
        initial_hp_cutoff_hz = 1000.0 if is_click else 300.0
    
    hp_cutoff_hz = initial_hp_cutoff_hz
    
    # Load WAV once
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    
    # Initial detection
    y_filt = highpass_filter(data, sr, hp_cutoff_hz)
    env = hilbert_envelope(y_filt, sr, smooth_ms=smooth_ms)
    onset_times, peak_times = detect_onsets_and_peaks_from_envelope(
        env, sr,
        threshold_ratio=threshold_ratio,
        min_distance_ms=min_distance_ms
    )
    
    # Create figure with extra space at bottom for widgets
    fig = plt.figure(figsize=(12, 8))
    
    # Create subplots for waveform and envelope
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=2, sharex=ax1)
    
    # Initial plot - waveform
    time_axis = np.arange(len(y_filt)) / sr
    waveform_line, = ax1.plot(time_axis, y_filt, alpha=0.5, linewidth=0.5, color='blue', label='Waveform')
    ax1.set_ylabel('Amplitude')
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title(f'Fujii Method Detection - {wav_path}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Store onset and peak lines for updating
    onset_lines_ax1 = []
    peak_lines_ax1 = []
    for i, ot in enumerate(onset_times):
        line = ax1.axvline(x=ot, color='g', linestyle='--', alpha=0.7, linewidth=1.5,
                          label='Onsets' if i == 0 else None)
        onset_lines_ax1.append(line)
    for i, pt in enumerate(peak_times):
        line = ax1.axvline(x=pt, color='r', linestyle=':', alpha=0.7, linewidth=1.5,
                          label='Peaks' if i == 0 else None)
        peak_lines_ax1.append(line)
    
    # Update legend after adding markers
    if len(onset_times) > 0 or len(peak_times) > 0:
        ax1.legend(loc='upper right')
    
    # Plot envelope
    env_label = f'Hilbert envelope (HPF = {hp_cutoff_hz:.0f} Hz)' if hp_cutoff_hz and hp_cutoff_hz > 0 else 'Hilbert envelope (no HPF)'
    envelope_line, = ax2.plot(time_axis, env, label=env_label, linewidth=1.5, color='orange')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Envelope')
    ax2.grid(True, alpha=0.3)
    
    onset_lines_ax2 = []
    peak_lines_ax2 = []
    for i, ot in enumerate(onset_times):
        line = ax2.axvline(x=ot, color='g', linestyle='--', alpha=0.7, linewidth=1.5)
        onset_lines_ax2.append(line)
    for i, pt in enumerate(peak_times):
        line = ax2.axvline(x=pt, color='r', linestyle=':', alpha=0.7, linewidth=1.5)
        peak_lines_ax2.append(line)
    
    legend = ax2.legend(loc='upper right')
    
    # Store current onset count as text on ax1
    onset_count_text = ax1.text(0.02, 0.98, f'Onsets detected: {len(onset_times)}',
                                transform=ax1.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add slider for HPF frequency
    plt.subplots_adjust(bottom=0.15)
    ax_slider = plt.axes([0.15, 0.05, 0.55, 0.03])
    slider = Slider(
        ax_slider, 
        'HPF Cutoff (Hz)',
        100.0,  # min value
        2000.0,  # max value
        valinit=hp_cutoff_hz,
        valstep=50.0
    )
    
    # Add re-detect button
    ax_button = plt.axes([0.72, 0.05, 0.12, 0.04])
    button = Button(ax_button, 'Re-detect')
    
    def update_plot(new_hp_cutoff: float):
        """Update the plot with new detection results using Fujii method."""
        # Re-apply HPF with new cutoff
        y_filt_new = highpass_filter(data, sr, new_hp_cutoff)
        
        # Recompute Hilbert envelope
        env_new = hilbert_envelope(y_filt_new, sr, smooth_ms=smooth_ms)
        
        # Re-detect onsets and peaks using Fujii method
        onset_times_new, peak_times_new = detect_onsets_and_peaks_from_envelope(
            env_new, sr,
            threshold_ratio=threshold_ratio,
            min_distance_ms=min_distance_ms
        )
        
        # Update waveform
        waveform_line.set_ydata(y_filt_new)
        
        # Update envelope plot
        envelope_line.set_ydata(env_new)
        
        # Update label
        env_label_new = f'Hilbert envelope (HPF = {new_hp_cutoff:.0f} Hz)' if new_hp_cutoff and new_hp_cutoff > 0 else 'Hilbert envelope (no HPF)'
        legend.texts[0].set_text(env_label_new)
        
        # Remove old onset and peak lines
        for line in onset_lines_ax1:
            line.remove()
        onset_lines_ax1.clear()
        for line in peak_lines_ax1:
            line.remove()
        peak_lines_ax1.clear()
        
        for line in onset_lines_ax2:
            line.remove()
        onset_lines_ax2.clear()
        for line in peak_lines_ax2:
            line.remove()
        peak_lines_ax2.clear()
        
        # Add new onset and peak lines
        for i, ot in enumerate(onset_times_new):
            line1 = ax1.axvline(x=ot, color='g', linestyle='--', alpha=0.7, linewidth=1.5)
            onset_lines_ax1.append(line1)
            line2 = ax2.axvline(x=ot, color='g', linestyle='--', alpha=0.7, linewidth=1.5)
            onset_lines_ax2.append(line2)
        
        for i, pt in enumerate(peak_times_new):
            line1 = ax1.axvline(x=pt, color='r', linestyle=':', alpha=0.7, linewidth=1.5)
            peak_lines_ax1.append(line1)
            line2 = ax2.axvline(x=pt, color='r', linestyle=':', alpha=0.7, linewidth=1.5)
            peak_lines_ax2.append(line2)
        
        # Update onset count text
        onset_count_text.set_text(f'Onsets detected: {len(onset_times_new)}')
        
        # Recompute y-axis limits
        ax1.relim()
        ax1.autoscale_view(scalex=False, scaley=True)
        ax2.relim()
        ax2.autoscale_view(scalex=False, scaley=True)
        
        fig.canvas.draw_idle()
    
    def on_button_click(event):
        """Handle re-detect button click."""
        new_hp_cutoff = slider.val
        update_plot(new_hp_cutoff)
    
    button.on_clicked(on_button_click)
    
    # Add interactive X-axis zoom functionality
    def on_scroll(event):
        """Handle mouse wheel scroll for X-axis zoom or Y-axis zoom with modifier key."""
        if event.inaxes is None or event.inaxes == ax_slider or event.inaxes == ax_button:
            return
        
        ax = event.inaxes
        
        # Check for Ctrl/Cmd key for Y-axis zoom
        if event.key in ('control', 'ctrl', 'cmd', 'super'):
            # Y-axis zoom
            cur_ylim = ax.get_ylim()
            ydata = event.ydata
            
            if ydata is None:
                return
            
            zoom_factor = 1.2
            if event.button == 'up':
                scale_factor = 1 / zoom_factor
            elif event.button == 'down':
                scale_factor = zoom_factor
            else:
                return
            
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
            new_ylim = [ydata - new_height * (1 - rely), ydata + new_height * rely]
            ax.set_ylim(new_ylim)
            fig.canvas.draw_idle()
        else:
            # X-axis zoom (default behavior)
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
    
    # Add pan functionality
    pan_data = {'pressed': False, 'x0': None, 'y0': None, 'xlim0': None, 'ylim0': None, 'ax': None}
    
    def on_press(event):
        """Handle mouse button press for panning."""
        if event.inaxes is None or event.inaxes == ax_slider or event.inaxes == ax_button:
            return
        if event.button != 1:  # Only left mouse button
            return
        
        pan_data['pressed'] = True
        pan_data['x0'] = event.xdata
        pan_data['y0'] = event.ydata
        pan_data['ax'] = event.inaxes
        pan_data['xlim0'] = event.inaxes.get_xlim()
        pan_data['ylim0'] = event.inaxes.get_ylim()
    
    def on_release(event):
        """Handle mouse button release."""
        pan_data['pressed'] = False
    
    def on_motion(event):
        """Handle mouse motion for panning."""
        if not pan_data['pressed']:
            return
        if event.inaxes != pan_data['ax']:
            return
        if event.xdata is None or event.ydata is None:
            return
        
        dx = event.xdata - pan_data['x0']
        dy = event.ydata - pan_data['y0']
        
        xlim = pan_data['xlim0']
        ylim = pan_data['ylim0']
        
        pan_data['ax'].set_xlim([xlim[0] - dx, xlim[1] - dx])
        pan_data['ax'].set_ylim([ylim[0] - dy, ylim[1] - dy])
        fig.canvas.draw_idle()
    
    def on_double_click(event):
        """Handle double-click to auto-scale Y-axis to 85% of max value."""
        if event.inaxes is None or event.inaxes == ax_slider or event.inaxes == ax_button:
            return
        if event.dblclick:
            ax = event.inaxes
            
            # Get the current x-axis limits
            xlim = ax.get_xlim()
            
            # Find the visible data in the current x-range
            if ax == ax1:
                # For waveform
                mask = (time_axis >= xlim[0]) & (time_axis <= xlim[1])
                visible_data = y_filt[mask]
            elif ax == ax2:
                # For envelope
                mask = (time_axis >= xlim[0]) & (time_axis <= xlim[1])
                visible_data = env[mask]
            else:
                return
            
            if len(visible_data) == 0:
                return
            
            # Find max absolute value in visible range
            max_val = np.max(np.abs(visible_data))
            
            if max_val == 0:
                return
            
            # Set Y-axis so max value is at 85% of the display
            target_max = max_val / 0.85
            ax.set_ylim([-target_max, target_max])
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_press_event', on_double_click)
    plt.show()


def interactive_hpf_tuning(
    wav_path: str,
    *,
    initial_hp_cutoff_hz: float | None = 300.0,
    is_click: bool = False,
    threshold_ratio: float = 0.1,
    min_distance_ms: float = 100.0,
    smooth_ms: float | None = 0.5,
) -> None:
    """
    Simple REPL-style helper for manually tuning the HPF cutoff.

    Loop behavior:
        1) Load the WAV.
        2) Apply highpass_filter with the current hp_cutoff_hz.
        3) Compute Hilbert envelope.
        4) Run detect_onsets_and_peaks_from_envelope.
        5) Plot waveform + envelope + onsets/peaks using plot_waveform_and_envelope,
           with legend indicating the current HPF cutoff.
        6) Ask the user via input():
               "New HPF cutoff in Hz (empty to keep / quit): "
           - If the user presses ENTER with no input, exit the loop.
           - If the user enters a number, parse it as float, set hp_cutoff_hz
             to that value, and repeat from step 2.

    Args:
        wav_path: path to a mono WAV file (click or tap).
        initial_hp_cutoff_hz: starting HPF cutoff in Hz
                              (e.g., 1000 Hz for clicks, 300 Hz for taps).
        is_click: if True, you may choose a different default cutoff internally.
        threshold_ratio, min_distance_ms, smooth_ms:
             passed through to the core detection.
    """
    # Set initial cutoff based on is_click if not specified
    if initial_hp_cutoff_hz is None:
        initial_hp_cutoff_hz = 1000.0 if is_click else 300.0
    
    hp_cutoff_hz = initial_hp_cutoff_hz
    
    # Load WAV once
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    
    while True:
        # Apply HPF
        y_filt = highpass_filter(data, sr, hp_cutoff_hz)
        
        # Compute envelope
        env = hilbert_envelope(y_filt, sr, smooth_ms=smooth_ms)
        
        # Detect onsets and peaks
        onset_times, peak_times = detect_onsets_and_peaks_from_envelope(
            env, sr,
            threshold_ratio=threshold_ratio,
            min_distance_ms=min_distance_ms
        )
        
        # Plot
        plot_waveform_and_envelope(
            y_filt, sr, env, onset_times, peak_times,
            hp_cutoff_hz=hp_cutoff_hz,
            title=f"Interactive HPF Tuning - {wav_path}"
        )
        
        # Ask for new cutoff
        user_input = input("New HPF cutoff in Hz (empty to keep / quit): ").strip()
        if not user_input:
            print("Exiting interactive tuning.")
            break
        
        try:
            hp_cutoff_hz = float(user_input)
            print(f"Setting HPF cutoff to {hp_cutoff_hz} Hz...")
        except ValueError:
            print("Invalid input. Please enter a number or press ENTER to quit.")


def run_click_detection_with_dialog(
    *,
    default_hp_cutoff_hz: float | None = 1000.0,
    threshold_ratio: float = 0.1,
    min_distance_ms: float = 100.0,
    smooth_ms: float | None = 0.5,
) -> None:
    """
    GUI helper for metronome clicks:

        1) Open a file-open dialog to select a WAV file.
        2) Run detect_click_onsets_and_peaks on the selected file.
        3) Open a 'save as' dialog to choose a CSV path (folder + file name).
        4) Save onset/peak times to CSV with save_onsets_and_peaks_csv(label="click").

    Behavior:
        - If the user cancels the WAV selection, return without error.
        - If the user cancels the CSV save dialog, return without error.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("Warning: tkinter not available. GUI dialogs cannot be used.")
        return
    
    # Create and hide root window
    root = tk.Tk()
    root.withdraw()
    
    # File-open dialog for WAV
    wav_path = filedialog.askopenfilename(
        title="Select WAV file (click track)",
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
    )
    
    if not wav_path:
        print("No WAV file selected. Exiting.")
        return
    
    # Run detection
    print(f"Detecting click onsets and peaks from: {wav_path}")
    onset_times, peak_times = detect_click_onsets_and_peaks(
        wav_path,
        hp_cutoff_hz=default_hp_cutoff_hz,
        threshold_ratio=threshold_ratio,
        min_distance_ms=min_distance_ms,
        smooth_ms=smooth_ms,
    )
    print(f"Detected {len(onset_times)} clicks.")
    
    # Save-as dialog for CSV
    csv_path = filedialog.asksaveasfilename(
        title="Save CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if not csv_path:
        print("No CSV file selected. Exiting without saving.")
        return
    
    # Save to CSV
    save_onsets_and_peaks_csv(csv_path, onset_times, peak_times, label="click")
    print(f"Saved onset/peak times to: {csv_path}")


def run_tap_detection_with_dialog(
    *,
    default_hp_cutoff_hz: float | None = 300.0,
    threshold_ratio: float = 0.1,
    min_distance_ms: float = 100.0,
    smooth_ms: float | None = 0.5,
) -> None:
    """
    GUI helper for finger taps:

        1) Open a file-open dialog to select a WAV file.
        2) Run detect_tap_onsets_and_peaks on the selected file.
        3) Open a 'save as' dialog to choose a CSV path (folder + file name).
        4) Save onset/peak times to CSV with save_onsets_and_peaks_csv(label="tap").

    Behavior is analogous to run_click_detection_with_dialog.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("Warning: tkinter not available. GUI dialogs cannot be used.")
        return
    
    # Create and hide root window
    root = tk.Tk()
    root.withdraw()
    
    # File-open dialog for WAV
    wav_path = filedialog.askopenfilename(
        title="Select WAV file (finger taps)",
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
    )
    
    if not wav_path:
        print("No WAV file selected. Exiting.")
        return
    
    # Run detection
    print(f"Detecting tap onsets and peaks from: {wav_path}")
    onset_times, peak_times = detect_tap_onsets_and_peaks(
        wav_path,
        hp_cutoff_hz=default_hp_cutoff_hz,
        threshold_ratio=threshold_ratio,
        min_distance_ms=min_distance_ms,
        smooth_ms=smooth_ms,
    )
    print(f"Detected {len(onset_times)} taps.")
    
    # Save-as dialog for CSV
    csv_path = filedialog.asksaveasfilename(
        title="Save CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if not csv_path:
        print("No CSV file selected. Exiting without saving.")
        return
    
    # Save to CSV
    save_onsets_and_peaks_csv(csv_path, onset_times, peak_times, label="tap")
    print(f"Saved onset/peak times to: {csv_path}")


if __name__ == "__main__":
    """
    Demonstration of the unified onset + peak detection module.
    
    Note: This demo requires test audio files.
    Replace the paths below with actual test files to run.
    """
    import os
    
    print("=" * 70)
    print("Unified Onset + Peak Detection Module Demo")
    print("Using Fujii-style Hilbert envelope method")
    print("=" * 70)
    
    # Demo 1: Click detection
    print("\n1. Click Track Detection")
    print("-" * 70)
    click_wav_path = "test_click.wav"
    if os.path.exists(click_wav_path):
        try:
            onset_times, peak_times = detect_click_onsets_and_peaks(
                click_wav_path,
                hp_cutoff_hz=1000.0
            )
            print(f"Detected {len(onset_times)} clicks")
            print(f"Onset times [s]: {onset_times}")
            print(f"Peak times [s]:  {peak_times}")
        except Exception as e:
            print(f"Error processing click file: {e}")
    else:
        print(f"Test file '{click_wav_path}' not found. Skipping demo.")
    
    # Demo 2: Tap detection
    print("\n2. Tap Detection")
    print("-" * 70)
    tap_wav_path = "test_tap.wav"
    if os.path.exists(tap_wav_path):
        try:
            onset_times, peak_times = detect_tap_onsets_and_peaks(
                tap_wav_path,
                hp_cutoff_hz=300.0
            )
            print(f"Detected {len(onset_times)} taps")
            print(f"Onset times [s]: {onset_times}")
            print(f"Peak times [s]:  {peak_times}")
        except Exception as e:
            print(f"Error processing tap file: {e}")
    else:
        print(f"Test file '{tap_wav_path}' not found. Skipping demo.")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
