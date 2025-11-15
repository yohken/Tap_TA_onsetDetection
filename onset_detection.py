"""
Onset detection module for three types of audio signals:
1. Metronome click tracks (theoretical grid positions)
2. Finger tap recordings
3. Sung Japanese syllable "ta" (/t/ burst detection)

This module provides clean, well-documented Python code for onset detection
using standard DSP techniques (no machine learning).

Target Python version: 3.10+
Dependencies: numpy, scipy, librosa, textgrid
"""

from __future__ import annotations

import numpy as np
import scipy.signal
import librosa
from textgrid import TextGrid
from typing import Optional
import matplotlib.pyplot as plt


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
) -> None:
    """
    Plot waveform, envelope, and detected onsets using matplotlib.
    
    Args:
        y: audio signal.
        sr: sampling rate.
        env: RMS envelope.
        times: time stamps for envelope frames.
        onset_times: detected onset times in seconds.
        title: optional plot title.
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
    ax2.plot(times, env, label='RMS Envelope', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('RMS Energy')
    ax2.grid(True, alpha=0.3)
    
    # Mark onsets on envelope
    for onset_t in onset_times:
        ax2.axvline(x=onset_t, color='r', linestyle='--', alpha=0.7, linewidth=1.5, label='Onset' if onset_t == onset_times[0] else '')
    
    ax2.legend()
    
    plt.tight_layout()
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
