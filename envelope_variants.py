"""
Modular envelope variant functions for onset detection comparison.

This module provides reusable functions for computing different envelope
variants (Hilbert with/without smoothing, RMS-based) and detecting events
using Fujii-style detection with configurable parameters.

Functions support comparing:
- Different HPF cutoffs
- Smoothing window sizes
- Global peak filtering thresholds
- Minimum inter-event distances
- Lookback quiet-period criteria
- Rise time and slope metrics
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
import librosa
from typing import Optional


def compute_hilbert_envelope_variant(
    y: np.ndarray,
    sr: int,
    hpf_cutoff: float | None = None,
    smooth_ms: float | None = None,
) -> np.ndarray:
    """
    Compute Hilbert envelope variant with optional HPF and smoothing.

    This is the core Fujii-style envelope method, with parameterized
    preprocessing options for comparison studies.

    Args:
        y: mono audio signal, shape (N,).
        sr: sampling rate [Hz].
        hpf_cutoff: high-pass filter cutoff in Hz. If None or 0, no HPF applied.
        smooth_ms: smoothing window in milliseconds. If None or ≤0, no smoothing.

    Returns:
        env: Hilbert envelope, same length as y.
    """
    # Handle empty signal
    if len(y) == 0:
        return np.array([])

    # Apply optional high-pass filter
    if hpf_cutoff is not None and hpf_cutoff > 0:
        sos = butter(4, hpf_cutoff, btype='hp', fs=sr, output='sos')
        y_filt = sosfiltfilt(sos, y)
    else:
        y_filt = y

    # Compute Hilbert transform and envelope
    analytic = hilbert(y_filt)
    env = np.abs(analytic)

    # Optional smoothing with moving average
    if smooth_ms is not None and smooth_ms > 0:
        win = int(round(smooth_ms * 1e-3 * sr))
        if win >= 2:
            kernel = np.ones(win) / win
            env = np.convolve(env, kernel, mode='same')

    return env


def compute_rms_envelope_variant(
    y: np.ndarray,
    sr: int,
    hpf_cutoff: float | None = None,
    frame_ms: float = 5.0,
    hop_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute RMS envelope variant with optional HPF.

    Uses librosa's RMS computation with configurable frame and hop sizes.

    Args:
        y: mono audio signal, shape (N,).
        sr: sampling rate [Hz].
        hpf_cutoff: high-pass filter cutoff in Hz. If None or 0, no HPF applied.
        frame_ms: frame length in milliseconds.
        hop_ms: hop length in milliseconds.

    Returns:
        env: RMS envelope, shape (n_frames,).
        times: time stamps for each frame in seconds, shape (n_frames,).
    """
    # Apply optional high-pass filter
    if hpf_cutoff is not None and hpf_cutoff > 0:
        sos = butter(4, hpf_cutoff, btype='hp', fs=sr, output='sos')
        y_filt = sosfiltfilt(sos, y)
    else:
        y_filt = y

    # Convert milliseconds to samples
    frame_length = int(frame_ms * sr / 1000)
    hop_length = int(hop_ms * sr / 1000)

    # Compute RMS envelope
    env = librosa.feature.rms(
        y=y_filt,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # Compute time stamps
    n_frames = len(env)
    times = librosa.frames_to_time(
        np.arange(n_frames),
        sr=sr,
        hop_length=hop_length
    )

    return env, times


def detect_events_fujii(
    env: np.ndarray,
    sr: int,
    threshold_ratio: float = 0.1,
    global_min_height_ratio: float = 0.0,
    min_distance_ms: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect events from envelope using Fujii-style method.

    Core algorithm:
    1. Find peaks in envelope with global height and distance constraints
    2. For each peak, search backward to find onset (10% threshold crossing)
    3. Return onset and peak times

    Args:
        env: envelope signal, 1D array.
        sr: sampling rate [Hz].
        threshold_ratio: local onset threshold as fraction of peak amplitude
                         (default 0.1 = 10%, following Fujii).
        global_min_height_ratio: global height threshold relative to env.max()
                                 to filter minor peaks (0 = no filtering).
        min_distance_ms: minimum distance between peaks in milliseconds.

    Returns:
        onset_times: 1D array of onset times [s] for each detected event.
        peak_times: 1D array of corresponding peak times [s].
    """
    if len(env) == 0:
        return np.array([]), np.array([])

    # Convert min_distance_ms to samples
    min_distance_samples = int(round(min_distance_ms * 1e-3 * sr))

    # Find peaks in envelope
    max_env = np.max(env)
    if max_env == 0:
        return np.array([]), np.array([])

    # Use global_min_height_ratio for peak detection
    height_threshold = global_min_height_ratio * max_env
    peaks, _ = find_peaks(
        env,
        height=height_threshold,
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
            onset_idx = float(p)
        else:
            # Found crossing between k and k+1
            e0 = env[k]
            e1 = env[k + 1]

            if e1 <= e0:
                onset_idx = float(k + 1)
            else:
                # Linear interpolation
                alpha = (th - e0) / (e1 - e0)
                onset_idx = k + alpha

        onset_indices.append(onset_idx)
        peak_indices.append(float(p))

    # Convert to arrays and times
    onset_indices = np.array(onset_indices)
    peak_indices = np.array(peak_indices)

    onset_times = onset_indices / sr
    peak_times = peak_indices / sr

    return onset_times, peak_times


def apply_lookback_quiet_period(
    env: np.ndarray,
    onset_idx: float,
    peak_idx: int,
    lookback_points: int,
    threshold: float,
) -> float | None:
    """
    Apply lookback quiet-period criterion to validate/adjust onset.

    Checks if envelope was below threshold for lookback_points samples
    before the onset. This enforces a "quiet baseline" requirement.

    Args:
        env: envelope signal, 1D array.
        onset_idx: proposed onset index (can be fractional).
        peak_idx: corresponding peak index (integer).
        lookback_points: number of samples to check before onset (e.g., 74 ≈ 2ms @ 48kHz).
        threshold: amplitude threshold for quiet-period check.

    Returns:
        Adjusted onset_idx if valid, or None if quiet-period criterion fails.
        If lookback_points == 0, returns onset_idx unchanged.
    """
    if lookback_points == 0:
        return onset_idx

    # Convert onset_idx to integer for indexing
    onset_int = int(round(onset_idx))

    # Check lookback region
    lookback_start = max(0, onset_int - lookback_points)
    if lookback_start >= onset_int:
        # Not enough samples to check
        return None

    lookback_region = env[lookback_start:onset_int]

    # Verify all points in lookback region are below threshold
    if len(lookback_region) > 0 and np.all(lookback_region < threshold):
        return onset_idx
    else:
        return None


def calc_rise_time(
    env: np.ndarray,
    onset_idx: float,
    peak_idx: int,
    sr: int,
    low_ratio: float = 0.1,
    high_ratio: float = 0.9,
) -> tuple[float, float]:
    """
    Calculate rise time from low_ratio to high_ratio of peak amplitude.

    Standard rise time is measured from 10% to 90% of peak amplitude.

    Args:
        env: envelope signal, 1D array.
        onset_idx: onset index (can be fractional).
        peak_idx: peak index (integer).
        sr: sampling rate [Hz].
        low_ratio: lower amplitude ratio (default 0.1 = 10%).
        high_ratio: upper amplitude ratio (default 0.9 = 90%).

    Returns:
        rise_samples: rise time in samples (fractional).
        rise_ms: rise time in milliseconds.
    """
    if peak_idx <= 0 or peak_idx >= len(env):
        return 0.0, 0.0

    peak_amp = env[peak_idx]
    if peak_amp == 0:
        return 0.0, 0.0

    low_thresh = low_ratio * peak_amp
    high_thresh = high_ratio * peak_amp

    # Find index where envelope crosses low_thresh (going up)
    low_idx = int(round(onset_idx))
    while low_idx < peak_idx and env[low_idx] < low_thresh:
        low_idx += 1

    # Find index where envelope crosses high_thresh (going up)
    high_idx = low_idx
    while high_idx < peak_idx and env[high_idx] < high_thresh:
        high_idx += 1

    # Compute rise time
    rise_samples = float(high_idx - low_idx)
    rise_ms = (rise_samples / sr) * 1000.0

    return rise_samples, rise_ms


def calc_slope(
    env: np.ndarray,
    onset_idx: float,
    peak_idx: int,
    sr: int,
) -> float:
    """
    Calculate normalized amplitude slope from onset to peak.

    Slope = (peak_amp - onset_amp) / (time_diff) / peak_amp

    This gives a normalized measure of how quickly amplitude rises.

    Args:
        env: envelope signal, 1D array.
        onset_idx: onset index (can be fractional).
        peak_idx: peak index (integer).
        sr: sampling rate [Hz].

    Returns:
        slope: normalized slope in 1/seconds (amplitude/sec / amplitude).
    """
    if peak_idx <= 0 or peak_idx >= len(env):
        return 0.0

    onset_int = int(round(onset_idx))
    if onset_int >= peak_idx:
        return 0.0

    onset_amp = env[onset_int]
    peak_amp = env[peak_idx]

    if peak_amp == 0:
        return 0.0

    # Time difference in seconds
    time_diff = (peak_idx - onset_idx) / sr

    if time_diff == 0:
        return 0.0

    # Normalized slope
    slope = (peak_amp - onset_amp) / time_diff / peak_amp

    return slope


def build_variant_id(
    envelope_type: str,
    hpf_cutoff: float | None,
    smooth_ms: float | None,
    global_min_height_ratio: float,
    min_distance_ms: float,
    lookback_points: int,
    frame_ms: float | None = None,
    hop_ms: float | None = None,
) -> str:
    """
    Construct a variant ID string summarizing all parameters.

    Format: "type|hpf=X|smooth=Y|gmin=Z|mindist=Q|lookback=L[|frame=F|hop=H]"

    Args:
        envelope_type: "hilbert" or "rms".
        hpf_cutoff: HPF cutoff in Hz (None or 0 = no HPF).
        smooth_ms: smoothing window in ms (None or 0 = no smoothing, Hilbert only).
        global_min_height_ratio: global min height ratio.
        min_distance_ms: minimum distance in ms.
        lookback_points: lookback points (0 = disabled).
        frame_ms: RMS frame length in ms (RMS only).
        hop_ms: RMS hop length in ms (RMS only).

    Returns:
        variant_id: string identifier.
    """
    parts = [envelope_type]

    # HPF
    if hpf_cutoff is None or hpf_cutoff == 0:
        parts.append("hpf=none")
    else:
        parts.append(f"hpf={hpf_cutoff:.0f}")

    # Smoothing (Hilbert only)
    if envelope_type == "hilbert":
        if smooth_ms is None or smooth_ms <= 0:
            parts.append("smooth=none")
        else:
            parts.append(f"smooth={smooth_ms:.1f}ms")

    # RMS parameters
    if envelope_type == "rms":
        if frame_ms is not None:
            parts.append(f"frame={frame_ms:.1f}ms")
        if hop_ms is not None:
            parts.append(f"hop={hop_ms:.1f}ms")

    # Detection parameters
    parts.append(f"gmin={global_min_height_ratio:.2f}")
    parts.append(f"mindist={min_distance_ms:.0f}ms")
    parts.append(f"lookback={lookback_points}")

    return "|".join(parts)


if __name__ == "__main__":
    """
    Quick demonstration of envelope variant functions.
    """
    print("=" * 70)
    print("Envelope Variants Module Demo")
    print("=" * 70)

    # Generate synthetic test signal
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Create signal with two transients
    signal = np.zeros_like(t)
    for tap_time in [0.3, 0.7]:
        tap_idx = int(tap_time * sr)
        tap_dur = int(0.03 * sr)
        decay = np.exp(-np.linspace(0, 6, tap_dur))
        noise = np.random.randn(tap_dur)
        if tap_idx + tap_dur < len(signal):
            signal[tap_idx:tap_idx + tap_dur] += noise * decay * 0.5

    # Add background noise
    signal += np.random.randn(len(signal)) * 0.01

    print("\n1. Hilbert Envelope Variant (unsmoothed, 300 Hz HPF)")
    print("-" * 70)
    env_hilbert = compute_hilbert_envelope_variant(signal, sr, hpf_cutoff=300.0, smooth_ms=None)
    onset_times, peak_times = detect_events_fujii(env_hilbert, sr, threshold_ratio=0.1, min_distance_ms=100.0)
    print(f"Detected {len(onset_times)} events")
    print(f"Onset times: {onset_times}")

    print("\n2. Hilbert Envelope Variant (smoothed 0.5ms, 300 Hz HPF)")
    print("-" * 70)
    env_hilbert_smooth = compute_hilbert_envelope_variant(signal, sr, hpf_cutoff=300.0, smooth_ms=0.5)
    onset_times_smooth, peak_times_smooth = detect_events_fujii(env_hilbert_smooth, sr, threshold_ratio=0.1, min_distance_ms=100.0)
    print(f"Detected {len(onset_times_smooth)} events")
    print(f"Onset times: {onset_times_smooth}")

    print("\n3. RMS Envelope Variant (5ms frame, 1ms hop, 300 Hz HPF)")
    print("-" * 70)
    env_rms, times_rms = compute_rms_envelope_variant(signal, sr, hpf_cutoff=300.0, frame_ms=5.0, hop_ms=1.0)
    print(f"RMS envelope computed: {len(env_rms)} frames")

    print("\n4. Variant ID Construction")
    print("-" * 70)
    variant_id = build_variant_id("hilbert", 300.0, None, 0.2, 100.0, 0)
    print(f"Example variant ID: {variant_id}")

    print("\n5. Rise Time Calculation")
    print("-" * 70)
    if len(onset_times) > 0 and len(peak_times) > 0:
        onset_idx = onset_times[0] * sr
        peak_idx = int(peak_times[0] * sr)
        rise_samples, rise_ms = calc_rise_time(env_hilbert, onset_idx, peak_idx, sr)
        print(f"First event rise time (10-90%): {rise_ms:.2f} ms")

        slope = calc_slope(env_hilbert, onset_idx, peak_idx, sr)
        print(f"First event normalized slope: {slope:.2f} 1/s")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
