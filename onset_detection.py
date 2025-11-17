"""
Onset detection module for three types of audio signals:
1. Metronome click tracks (theoretical grid positions)
2. Finger tap recordings
3. Sung Japanese syllable "ta" (/t/ burst detection)

This module provides clean, well-documented Python code for onset detection
using standard DSP techniques (no machine learning).

IMPORTANT NOTE FOR RE-DETECTION:
- The interactive plotting function plot_envelope_with_onsets_interactive in this module
  uses RMS envelope + derivative-based detection.
- It does NOT implement the Fujii method (10% threshold, backward search, linear interpolation).
- For re-detection that complies with the Fujii method, use onset_hilbert module instead:
    * onset_hilbert.plot_waveform_and_envelope_interactive() for interactive re-detection
    * onset_hilbert.detect_tap_onsets_and_peaks() or detect_click_onsets_and_peaks() for detection

This module remains useful for:
- /t/ burst detection with TextGrid guidance (detect_t_burst_onsets_from_mfa)
- Metronome onset generation (get_click_onsets_from_bpm)
- Alternative detection methods for comparison purposes

Target Python version: 3.10+
Dependencies: numpy, scipy, librosa, textgrid
"""

from __future__ import annotations

import numpy as np
import scipy.signal
import librosa
from textgrid import TextGrid
from typing import Optional, Callable
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def compute_rms_envelope(
    y: np.ndarray,
    sr: int,
    band: tuple[float | None, float | None] | None = None,
    frame_length_ms: float = 5.0,
    hop_length_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a short-time RMS envelope of signal y.

    Args:
        y: audio signal, mono, float32 or float64.
        sr: sampling rate in Hz.
        band: optional (low_freq, high_freq) band-pass in Hz.
              If None, use full-band. If high_freq is None, treat it as 'sr/2'.
        frame_length_ms: analysis window length in milliseconds.
        hop_length_ms: hop size in milliseconds.

    Returns:
        env: RMS envelope as a 1D numpy array (frames,).
        times: time stamps (in seconds) for each frame.
    """
    # Apply band-pass filter if specified
    y_filtered = y.copy()
    if band is not None:
        low_freq, high_freq = band
        # Set high_freq to Nyquist if None
        if high_freq is None:
            high_freq = sr / 2.0
        
        # Design Butterworth filter
        nyquist = sr / 2.0
        
        # Determine filter type
        if low_freq is not None and low_freq > 0:
            if high_freq < nyquist:
                # Band-pass filter
                sos = scipy.signal.butter(
                    4, [low_freq, high_freq], btype='band', fs=sr, output='sos'
                )
            else:
                # High-pass filter
                sos = scipy.signal.butter(
                    4, low_freq, btype='high', fs=sr, output='sos'
                )
        else:
            # Low-pass filter
            sos = scipy.signal.butter(
                4, high_freq, btype='low', fs=sr, output='sos'
            )
        
        # Apply filter
        y_filtered = scipy.signal.sosfilt(sos, y_filtered)
    
    # Convert milliseconds to samples
    frame_length = int(frame_length_ms * sr / 1000)
    hop_length = int(hop_length_ms * sr / 1000)
    
    # Compute RMS envelope using librosa
    env = librosa.feature.rms(
        y=y_filtered, 
        frame_length=frame_length, 
        hop_length=hop_length
    )[0]
    
    # Compute time stamps for each frame
    n_frames = len(env)
    times = librosa.frames_to_time(
        np.arange(n_frames), 
        sr=sr, 
        hop_length=hop_length
    )
    
    return env, times


def compute_hilbert_envelope(
    y: np.ndarray,
    sr: int,
    band: tuple[float | None, float | None] | None = None,
) -> np.ndarray:
    """
    Compute the Hilbert envelope of signal y.
    
    The Hilbert envelope is computed as E(t) = sqrt(x(t)^2 + x_hat(t)^2),
    where x(t) is the filtered signal and x_hat(t) is its Hilbert transform.
    
    Args:
        y: audio signal, mono, float32 or float64.
        sr: sampling rate in Hz.
        band: optional (low_freq, high_freq) band-pass in Hz.
              If None, use full-band. If high_freq is None, treat it as 'sr/2'.
    
    Returns:
        env: Hilbert envelope as a 1D numpy array (same length as y).
    """
    # Apply band-pass filter if specified
    y_filtered = y.copy()
    if band is not None:
        low_freq, high_freq = band
        # Set high_freq to Nyquist if None
        if high_freq is None:
            high_freq = sr / 2.0
        
        # Design Butterworth filter
        nyquist = sr / 2.0
        
        # Determine filter type
        if low_freq is not None and low_freq > 0:
            if high_freq < nyquist:
                # Band-pass filter
                sos = scipy.signal.butter(
                    4, [low_freq, high_freq], btype='band', fs=sr, output='sos'
                )
            else:
                # High-pass filter
                sos = scipy.signal.butter(
                    4, low_freq, btype='high', fs=sr, output='sos'
                )
        else:
            # Low-pass filter
            sos = scipy.signal.butter(
                4, high_freq, btype='low', fs=sr, output='sos'
            )
        
        # Apply filter
        y_filtered = scipy.signal.sosfilt(sos, y_filtered)
    
    # Compute Hilbert transform
    analytic_signal = scipy.signal.hilbert(y_filtered)
    
    # Compute envelope as magnitude of analytic signal
    # E(t) = sqrt(x(t)^2 + x_hat(t)^2) = |analytic_signal|
    env = np.abs(analytic_signal)
    
    return env


def detect_onsets_from_envelope(
    env: np.ndarray,
    times: np.ndarray,
    diff_threshold_std: float = 2.0,
    min_interval_ms: float = 20.0,
) -> np.ndarray:
    """
    Detect onset times from an amplitude envelope using the derivative.

    Algorithm:
        1) Compute first-order difference of env: diff = np.diff(env).
        2) Keep only positive values: diff_pos = np.maximum(diff, 0).
        3) Estimate a threshold: mean + diff_threshold_std * std.
        4) Find peaks in diff_pos that exceed the threshold.
        5) Enforce a minimum time interval between peaks.

    Args:
        env: 1D envelope array.
        times: 1D time array (same length as env).
        diff_threshold_std: number of standard deviations above the mean for the threshold.
        min_interval_ms: minimum allowed interval between onsets in milliseconds.

    Returns:
        onset_times: 1D numpy array of onset times (seconds).
    """
    if len(env) == 0:
        return np.array([])
    
    # Compute first-order difference
    diff = np.diff(env)
    
    # Keep only positive values
    diff_pos = np.maximum(diff, 0)
    
    # Compute threshold
    if len(diff_pos) == 0 or np.std(diff_pos) == 0:
        return np.array([])
    
    threshold = np.mean(diff_pos) + diff_threshold_std * np.std(diff_pos)
    
    # Convert minimum interval to samples
    if len(times) > 1:
        time_step = times[1] - times[0]  # Time between consecutive frames
        min_distance = int(min_interval_ms / 1000.0 / time_step)
    else:
        min_distance = 1
    
    # Find peaks that exceed threshold
    peaks, _ = scipy.signal.find_peaks(
        diff_pos, 
        height=threshold, 
        distance=min_distance
    )
    
    # Convert peak indices to times
    # Note: diff reduces length by 1, so peaks correspond to times[peaks+1]
    # But we want the onset time, which is at the beginning of the rise
    onset_times = times[peaks] if len(peaks) > 0 else np.array([])
    
    return onset_times


def get_click_onsets_from_bpm(
    bpm: float,
    n_clicks: int,
    *,
    subdivision: int = 1,
    beat_offset: float = 0.0,
    time_offset_sec: float = 0.0,
) -> np.ndarray:
    """
    Compute theoretical onset times for a click track.

    Args:
        bpm: tempo in beats per minute.
        n_clicks: number of clicks to generate.
        subdivision: 1 for quarter notes, 2 for eighth notes, 4 for sixteenth notes, etc.
        beat_offset: starting beat index (0 means the first click at time_offset_sec).
        time_offset_sec: constant offset in seconds (e.g., if the exported audio has leading silence).

    Returns:
        onset_times: array of onset times in seconds.
    """
    # Seconds per beat
    seconds_per_beat = 60.0 / bpm
    
    # Seconds per click (accounting for subdivision)
    seconds_per_click = seconds_per_beat / subdivision
    
    # Generate onset times
    click_indices = np.arange(n_clicks) + beat_offset
    onset_times = click_indices * seconds_per_click + time_offset_sec
    
    return onset_times


def detect_metronome_onsets_from_audio(
    wav_path: str,
    *,
    target_sr: int = 48000,
    threshold_ratio: float = 0.1,
    min_interval_ms: float = 50.0,
    prominence_ratio: float = 0.3,
) -> np.ndarray:
    """
    Detect metronome onsets from a mono WAV file using Hilbert envelope.
    
    Strategy:
        1) Load audio and resample to target_sr (48,000 Hz by default).
        2) Compute Hilbert envelope: E(t) = sqrt(x(t)^2 + x_hat(t)^2).
        3) Define onset as when envelope exceeds threshold_ratio (10%) of 
           maximum amplitude for each sound burst.
    
    Args:
        wav_path: path to WAV file.
        target_sr: target sampling rate in Hz (default: 48000).
        threshold_ratio: onset threshold as fraction of max amplitude (default: 0.1 = 10%).
        min_interval_ms: minimum interval between onsets in milliseconds (default: 50.0).
        prominence_ratio: minimum prominence for peak detection as fraction of max (default: 0.3).
    
    Returns:
        onset_times: 1D array of onset times in seconds.
    """
    # Load audio and resample to target sampling rate
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    
    # Compute Hilbert envelope (no filtering for metronome)
    env = compute_hilbert_envelope(y, sr, band=None)
    
    # Find peaks in the envelope to identify sound bursts
    # Use distance and prominence parameters to find significant bursts
    min_distance = int(min_interval_ms * sr / 1000.0)
    max_env = np.max(env)
    prominence = prominence_ratio * max_env
    
    peaks, _ = scipy.signal.find_peaks(env, distance=min_distance, prominence=prominence)
    
    # For each peak, find the onset point where envelope exceeds 10% of peak amplitude
    onset_times = []
    for peak_idx in peaks:
        peak_amplitude = env[peak_idx]
        threshold = threshold_ratio * peak_amplitude
        
        # Search backward from peak to find where envelope first exceeds threshold
        onset_idx = peak_idx
        for i in range(peak_idx - 1, max(0, peak_idx - min_distance), -1):
            if env[i] >= threshold:
                onset_idx = i
            else:
                break
        
        # Convert sample index to time
        onset_time = onset_idx / sr
        onset_times.append(onset_time)
    
    return np.array(onset_times)


def detect_tap_onsets_from_audio_hilbert(
    wav_path: str,
    *,
    target_sr: int = 48000,
    hp_cutoff: float = 500.0,
    threshold_ratio: float = 0.1,
    lookback_points: int = 74,
    min_interval_ms: float = 50.0,
    prominence_ratio: float = 0.3,
) -> np.ndarray:
    """
    Detect tap onsets from a mono WAV file using Hilbert envelope with lookback criterion.
    
    Strategy:
        1) Load audio and resample to target_sr (48,000 Hz by default).
        2) Apply high-pass filter (HPF) as preprocessing.
        3) Compute Hilbert envelope: E(t) = sqrt(x(t)^2 + x_hat(t)^2).
        4) Define onset as when amplitude exceeds threshold_ratio (10%) of 
           maximum amplitude of sound burst, BUT only if it was below this 
           threshold for lookback_points (~74 points = ~2ms at 48kHz) immediately before.
    
    Args:
        wav_path: path to WAV file.
        target_sr: target sampling rate in Hz (default: 48000).
        hp_cutoff: high-pass filter cutoff frequency in Hz (default: 500.0).
        threshold_ratio: onset threshold as fraction of max amplitude (default: 0.1 = 10%).
        lookback_points: number of points to check before onset (default: 74 = ~2ms at 48kHz).
        min_interval_ms: minimum interval between onsets in milliseconds (default: 50.0).
        prominence_ratio: minimum prominence for peak detection as fraction of max (default: 0.3).
    
    Returns:
        onset_times: 1D array of onset times in seconds.
    """
    # Load audio and resample to target sampling rate
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    
    # Compute Hilbert envelope with high-pass filtering
    env = compute_hilbert_envelope(y, sr, band=(hp_cutoff, None))
    
    # Find peaks in the envelope to identify potential sound bursts
    min_distance = int(min_interval_ms * sr / 1000.0)
    max_env = np.max(env)
    prominence = prominence_ratio * max_env
    
    peaks, _ = scipy.signal.find_peaks(env, distance=min_distance, prominence=prominence)
    
    # For each peak, find the onset point with lookback criterion
    onset_times = []
    for peak_idx in peaks:
        peak_amplitude = env[peak_idx]
        threshold = threshold_ratio * peak_amplitude
        
        # Search backward from peak to find where envelope first exceeds threshold
        onset_idx = None
        for i in range(peak_idx - 1, max(0, peak_idx - min_distance), -1):
            if env[i] >= threshold:
                # Check if envelope was below threshold for lookback_points before this point
                lookback_start = max(0, i - lookback_points)
                lookback_region = env[lookback_start:i]
                
                # Verify that all points in lookback region are below threshold
                if len(lookback_region) > 0 and np.all(lookback_region < threshold):
                    onset_idx = i
                    break
            else:
                # If we go below threshold, stop searching
                break
        
        # If no valid onset found with lookback criterion, use the first point above threshold
        if onset_idx is None:
            for i in range(peak_idx - 1, max(0, peak_idx - min_distance), -1):
                if env[i] >= threshold:
                    onset_idx = i
                else:
                    break
        
        # Convert sample index to time
        if onset_idx is not None:
            onset_time = onset_idx / sr
            onset_times.append(onset_time)
    
    return np.array(onset_times)


def detect_tap_onsets_from_audio(
    wav_path: str,
    *,
    hp_cutoff: float = 500.0,
    frame_length_ms: float = 5.0,
    hop_length_ms: float = 1.0,
    diff_threshold_std: float = 2.0,
    min_interval_ms: float = 50.0,
) -> np.ndarray:
    """
    Detect tap onsets from a mono WAV file.

    Steps:
        1) Load audio with librosa.load(wav_path, sr=None, mono=True).
        2) Apply a high-pass filter at hp_cutoff Hz (Butterworth, order 4).
        3) Compute a full-band RMS envelope using compute_rms_envelope.
        4) Detect onsets using detect_onsets_from_envelope.

    Returns:
        onset_times: 1D array of onset times in seconds.
    """
    # Load audio
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    
    # Apply high-pass filter by using band parameter with (hp_cutoff, None)
    env, times = compute_rms_envelope(
        y, sr,
        band=(hp_cutoff, None),
        frame_length_ms=frame_length_ms,
        hop_length_ms=hop_length_ms
    )
    
    # Detect onsets
    onset_times = detect_onsets_from_envelope(
        env, times,
        diff_threshold_std=diff_threshold_std,
        min_interval_ms=min_interval_ms
    )
    
    return onset_times


def detect_t_burst_onsets_from_mfa(
    wav_path: str,
    tg_path: str,
    *,
    tier_name: str = "phones",
    phone_label: str = "t",
    high_freq_min: float = 2000.0,
    frame_length_ms: float = 5.0,
    hop_length_ms: float = 1.0,
    diff_threshold_std: float = 2.0,
) -> np.ndarray:
    """
    Detect /t/ burst onsets from a WAV file and its MFA TextGrid.

    Onset definition:
        The onset is defined as the burst of /t/, i.e., the point where
        high-frequency energy rises sharply after a low-energy closure.

    Steps:
        1) Load audio with librosa.load(wav_path, sr=None, mono=True).
        2) Load the TextGrid (tg_path) and find the tier with name tier_name.
        3) For each interval whose .mark == phone_label ("t"):
            a) take the audio segment between interval.minTime and interval.maxTime.
            b) compute a high-frequency RMS envelope:
               - use compute_rms_envelope with band=(high_freq_min, None).
            c) run detect_onsets_from_envelope on this local envelope,
               but inside this interval choose ONLY the earliest onset.
            d) convert the local onset time back to absolute time in seconds.
        4) Collect all burst onset times into a numpy array and return it.

    Returns:
        onset_times: 1D numpy array of burst onset times (seconds).
    """
    # Load audio
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    
    # Load TextGrid
    tg = TextGrid.fromFile(tg_path)
    
    # Find the specified tier
    tier = None
    for t in tg.tiers:
        if t.name == tier_name:
            tier = t
            break
    
    if tier is None:
        raise ValueError(f"Tier '{tier_name}' not found in TextGrid")
    
    # Collect burst onset times
    burst_onsets = []
    
    # Process each interval with the target phone label
    for interval in tier:
        if interval.mark == phone_label:
            min_time = interval.minTime
            max_time = interval.maxTime
            
            # Extract audio segment
            start_sample = int(min_time * sr)
            end_sample = int(max_time * sr)
            
            # Ensure we don't go out of bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(y), end_sample)
            
            if end_sample <= start_sample:
                # Empty interval, use minTime as fallback
                burst_onsets.append(min_time)
                continue
            
            segment = y[start_sample:end_sample]
            
            # Compute high-frequency RMS envelope
            env, times = compute_rms_envelope(
                segment, sr,
                band=(high_freq_min, None),
                frame_length_ms=frame_length_ms,
                hop_length_ms=hop_length_ms
            )
            
            # Detect onsets in this segment
            local_onsets = detect_onsets_from_envelope(
                env, times,
                diff_threshold_std=diff_threshold_std,
                min_interval_ms=10.0  # Use smaller interval for fine-grained detection
            )
            
            # Take the earliest onset, or fall back to minTime
            if len(local_onsets) > 0:
                # Convert local time to absolute time
                absolute_onset = min_time + local_onsets[0]
                burst_onsets.append(absolute_onset)
            else:
                # Fallback to start of interval
                burst_onsets.append(min_time)
    
    return np.array(burst_onsets)


def plot_envelope_with_onsets(
    y: np.ndarray,
    sr: int,
    env: np.ndarray,
    times: np.ndarray,
    onset_times: np.ndarray,
    title: str = "",
    envelope_type: str = "RMS Envelope",
) -> None:
    """
    Plot waveform, envelope, and detected onsets using matplotlib.
    
    Supports interactive X-axis zoom using mouse wheel (or trackpad pinch).
    
    Args:
        y: audio signal.
        sr: sampling rate.
        env: envelope (RMS, Hilbert, or other).
        times: time stamps for envelope frames.
        onset_times: detected onset times in seconds.
        title: optional plot title.
        envelope_type: label for the envelope type (default: "RMS Envelope").
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Plot waveform
    time_axis = np.arange(len(y)) / sr
    ax1.plot(time_axis, y, alpha=0.5, linewidth=0.5)
    ax1.set_ylabel('Amplitude')
    ax1.set_title(title if title else 'Audio Waveform and Detected Onsets')
    ax1.grid(True, alpha=0.3)
    
    # Mark onsets on waveform
    for onset_t in onset_times:
        ax1.axvline(x=onset_t, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Plot envelope
    ax2.plot(times, env, label=envelope_type, linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Envelope')
    ax2.grid(True, alpha=0.3)
    
    # Mark onsets on envelope
    for onset_t in onset_times:
        ax2.axvline(x=onset_t, color='r', linestyle='--', alpha=0.7, linewidth=1.5, label='Onset' if onset_t == onset_times[0] else '')
    
    ax2.legend()
    
    # Add interactive X-axis zoom functionality
    def on_scroll(event):
        """Handle mouse wheel scroll for X-axis zoom."""
        if event.inaxes is None:
            return
        
        # Get the current axis
        ax = event.inaxes
        
        # Get current X-axis limits
        cur_xlim = ax.get_xlim()
        xdata = event.xdata  # Mouse X position in data coordinates
        
        # Zoom factor: scroll up (event.step > 0) zooms in, scroll down zooms out
        zoom_factor = 1.2
        if event.button == 'up':
            scale_factor = 1 / zoom_factor
        elif event.button == 'down':
            scale_factor = zoom_factor
        else:
            return
        
        # Calculate new X-axis limits centered on mouse position
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        
        new_xlim = [xdata - new_width * (1 - relx), xdata + new_width * relx]
        
        # Apply new X-axis limits to the scrolled axis
        ax.set_xlim(new_xlim)
        
        # Redraw the canvas
        fig.canvas.draw_idle()
    
    # Connect the scroll event to both subplots
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    plt.tight_layout()
    plt.show()


def plot_envelope_with_onsets_interactive(
    wav_path: str,
    y: np.ndarray,
    sr: int,
    initial_hp_cutoff: float = 500.0,
    diff_threshold_std: float = 2.0,
    min_interval_ms: float = 50.0,
    title: str = "",
    detection_type: str = "tap",
) -> None:
    """
    Plot waveform, envelope, and detected onsets with interactive HPF frequency control.
    
    Features:
    - Interactive slider to adjust HPF cutoff frequency
    - Re-detect button to recompute onsets with new frequency
    - X-axis zoom using mouse wheel (inherited from plot_envelope_with_onsets)
    
    Args:
        wav_path: Path to the WAV file (for re-detection).
        y: audio signal.
        sr: sampling rate.
        initial_hp_cutoff: Initial high-pass filter cutoff frequency in Hz.
        diff_threshold_std: Threshold for onset detection (mean + k*std).
        min_interval_ms: Minimum interval between onsets in milliseconds.
        title: optional plot title.
        detection_type: Type of detection ("tap" or "t_burst").
    """
    # Initial detection
    if detection_type == "tap":
        env, times = compute_rms_envelope(
            y, sr, 
            band=(initial_hp_cutoff, None)
        )
        onset_times = detect_onsets_from_envelope(
            env, times,
            diff_threshold_std=diff_threshold_std,
            min_interval_ms=min_interval_ms
        )
    else:  # t_burst
        env, times = compute_rms_envelope(
            y, sr,
            band=(initial_hp_cutoff, None)
        )
        onset_times = detect_onsets_from_envelope(
            env, times,
            diff_threshold_std=diff_threshold_std,
            min_interval_ms=min_interval_ms
        )
    
    # Create figure with extra space at bottom for widgets
    fig = plt.figure(figsize=(12, 8))
    
    # Create subplots for waveform and envelope
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=2, sharex=ax1)
    
    # Initial plot
    time_axis = np.arange(len(y)) / sr
    waveform_line, = ax1.plot(time_axis, y, alpha=0.5, linewidth=0.5, color='blue')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(title if title else 'Audio Waveform and Detected Onsets')
    ax1.grid(True, alpha=0.3)
    
    # Store onset lines for updating
    onset_lines_ax1 = []
    for onset_t in onset_times:
        line = ax1.axvline(x=onset_t, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
        onset_lines_ax1.append(line)
    
    # Plot envelope
    envelope_line, = ax2.plot(times, env, label=f'RMS Envelope (HP {initial_hp_cutoff:.0f} Hz)', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Envelope')
    ax2.grid(True, alpha=0.3)
    
    onset_lines_ax2 = []
    for onset_t in onset_times:
        line = ax2.axvline(x=onset_t, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
        onset_lines_ax2.append(line)
    
    legend = ax2.legend()
    
    # Store current onset count as text on ax1
    onset_count_text = ax1.text(0.02, 0.98, f'Onsets detected: {len(onset_times)}',
                                transform=ax1.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add slider for HPF frequency
    plt.subplots_adjust(bottom=0.20)
    ax_slider = plt.axes([0.15, 0.10, 0.65, 0.03])
    slider = Slider(
        ax_slider, 
        'HPF Cutoff (Hz)',
        100.0,  # min value
        2000.0,  # max value
        valinit=initial_hp_cutoff,
        valstep=50.0
    )
    
    # Add re-detect button
    ax_button = plt.axes([0.82, 0.05, 0.12, 0.04])
    button = Button(ax_button, 'Re-detect')
    
    # State to track current parameters
    state = {'hp_cutoff': initial_hp_cutoff}
    
    def update_plot(new_hp_cutoff: float):
        """Update the plot with new detection results."""
        # Recompute envelope with new HPF cutoff
        env_new, times_new = compute_rms_envelope(
            y, sr, 
            band=(new_hp_cutoff, None)
        )
        
        # Detect onsets
        onset_times_new = detect_onsets_from_envelope(
            env_new, times_new,
            diff_threshold_std=diff_threshold_std,
            min_interval_ms=min_interval_ms
        )
        
        # Update envelope plot
        envelope_line.set_ydata(env_new)
        envelope_line.set_xdata(times_new)
        
        # Update label
        legend.texts[0].set_text(f'RMS Envelope (HP {new_hp_cutoff:.0f} Hz)')
        
        # Remove old onset lines
        for line in onset_lines_ax1:
            line.remove()
        onset_lines_ax1.clear()
        
        for line in onset_lines_ax2:
            line.remove()
        onset_lines_ax2.clear()
        
        # Add new onset lines
        for onset_t in onset_times_new:
            line1 = ax1.axvline(x=onset_t, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
            onset_lines_ax1.append(line1)
            line2 = ax2.axvline(x=onset_t, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
            onset_lines_ax2.append(line2)
        
        # Update onset count text
        onset_count_text.set_text(f'Onsets detected: {len(onset_times_new)}')
        
        # Recompute y-axis limits for envelope
        ax2.relim()
        ax2.autoscale_view(scalex=False, scaley=True)
        
        fig.canvas.draw_idle()
    
    def on_button_click(event):
        """Handle re-detect button click."""
        new_hp_cutoff = slider.val
        state['hp_cutoff'] = new_hp_cutoff
        update_plot(new_hp_cutoff)
    
    button.on_clicked(on_button_click)
    
    # Add interactive X-axis zoom functionality
    def on_scroll(event):
        """Handle mouse wheel scroll for X-axis zoom."""
        if event.inaxes is None or event.inaxes == ax_slider or event.inaxes == ax_button:
            return
        
        # Get the current axis
        ax = event.inaxes
        
        # Get current X-axis limits
        cur_xlim = ax.get_xlim()
        xdata = event.xdata  # Mouse X position in data coordinates
        
        if xdata is None:
            return
        
        # Zoom factor: scroll up (event.step > 0) zooms in, scroll down zooms out
        zoom_factor = 1.2
        if event.button == 'up':
            scale_factor = 1 / zoom_factor
        elif event.button == 'down':
            scale_factor = zoom_factor
        else:
            return
        
        # Calculate new X-axis limits centered on mouse position
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        
        new_xlim = [xdata - new_width * (1 - relx), xdata + new_width * relx]
        
        # Apply new X-axis limits to the scrolled axis
        ax.set_xlim(new_xlim)
        
        # Redraw the canvas
        fig.canvas.draw_idle()
    
    # Connect the scroll event
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    plt.show()


if __name__ == "__main__":
    """
    Demonstration of onset detection functions.
    
    Note: This demo requires test audio files and TextGrid files.
    Replace the paths below with actual test files to run.
    """
    import os
    
    print("=" * 60)
    print("Onset Detection Module Demo")
    print("=" * 60)
    
    # Demo 1: Click track onset generation
    print("\n1. Click Track Onset Generation (Theoretical)")
    print("-" * 60)
    bpm = 120
    n_clicks = 8
    click_onsets = get_click_onsets_from_bpm(bpm, n_clicks, subdivision=1)
    print(f"BPM: {bpm}, Number of clicks: {n_clicks}")
    print(f"Generated onset times (seconds): {click_onsets}")
    
    # Demo 2: Tap onset detection
    print("\n2. Tap Onset Detection from Audio")
    print("-" * 60)
    tap_wav_path = "test_tap.wav"  # Replace with actual test file
    if os.path.exists(tap_wav_path):
        try:
            tap_onsets = detect_tap_onsets_from_audio(
                tap_wav_path,
                hp_cutoff=500.0,
                diff_threshold_std=2.0,
                min_interval_ms=50.0
            )
            print(f"Detected {len(tap_onsets)} tap onsets:")
            print(tap_onsets)
            
            # Optionally plot
            # y, sr = librosa.load(tap_wav_path, sr=None, mono=True)
            # env, times = compute_rms_envelope(y, sr, band=(500.0, None))
            # plot_envelope_with_onsets(y, sr, env, times, tap_onsets, "Tap Onsets")
        except Exception as e:
            print(f"Error processing tap file: {e}")
    else:
        print(f"Test file '{tap_wav_path}' not found. Skipping demo.")
    
    # Demo 3: /t/ burst onset detection
    print("\n3. /t/ Burst Onset Detection from MFA TextGrid")
    print("-" * 60)
    ta_wav_path = "test_ta.wav"  # Replace with actual test file
    ta_tg_path = "test_ta.TextGrid"  # Replace with actual test file
    if os.path.exists(ta_wav_path) and os.path.exists(ta_tg_path):
        try:
            t_burst_onsets = detect_t_burst_onsets_from_mfa(
                ta_wav_path,
                ta_tg_path,
                tier_name="phones",
                phone_label="t",
                high_freq_min=2000.0,
                diff_threshold_std=2.0
            )
            print(f"Detected {len(t_burst_onsets)} /t/ burst onsets:")
            print(t_burst_onsets)
            
            # Optionally plot
            # y, sr = librosa.load(ta_wav_path, sr=None, mono=True)
            # env, times = compute_rms_envelope(y, sr, band=(2000.0, None))
            # plot_envelope_with_onsets(y, sr, env, times, t_burst_onsets, "/t/ Burst Onsets")
        except Exception as e:
            print(f"Error processing /t/ burst file: {e}")
    else:
        print(f"Test files '{ta_wav_path}' or '{ta_tg_path}' not found. Skipping demo.")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
