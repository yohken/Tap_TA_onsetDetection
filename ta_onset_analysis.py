"""
TA onset analysis module for detecting feature points in "ta" syllable segments.

This module implements scientifically-grounded methods for detecting:
1. T_start (consonant onset /t/) - Fujii 10% method with backward search + interpolation
2. T_peak (burst max of /t/) - High-frequency envelope maximum
3. A_start (vowel onset /a/) - Periodicity-based voicing detection
4. A_peak (first stable periodic peak after A_start)
5. A_end (end of vowel /a/)

Target Python version: 3.10+
Dependencies: numpy, scipy, librosa
"""

from __future__ import annotations

import numpy as np
import scipy.signal
from typing import Optional


def _bandpass_filter(
    y: np.ndarray,
    sr: int,
    low_freq: float,
    high_freq: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase bandpass Butterworth filter.
    
    Args:
        y: Input signal.
        sr: Sampling rate in Hz.
        low_freq: Lower cutoff frequency in Hz.
        high_freq: Upper cutoff frequency in Hz.
        order: Filter order.
    
    Returns:
        Filtered signal.
    """
    nyquist = sr / 2.0
    low = max(low_freq / nyquist, 0.001)
    high = min(high_freq / nyquist, 0.999)
    
    if low >= high:
        return y
    
    sos = scipy.signal.butter(order, [low, high], btype='band', output='sos')
    return scipy.signal.sosfiltfilt(sos, y)


def _highpass_filter(
    y: np.ndarray,
    sr: int,
    cutoff_freq: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase highpass Butterworth filter.
    
    Args:
        y: Input signal.
        sr: Sampling rate in Hz.
        cutoff_freq: Cutoff frequency in Hz.
        order: Filter order.
    
    Returns:
        Filtered signal.
    """
    nyquist = sr / 2.0
    normalized_cutoff = cutoff_freq / nyquist
    
    if normalized_cutoff >= 1.0:
        return y
    
    sos = scipy.signal.butter(order, normalized_cutoff, btype='high', output='sos')
    return scipy.signal.sosfiltfilt(sos, y)


def _lowpass_filter(
    y: np.ndarray,
    sr: int,
    cutoff_freq: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase lowpass Butterworth filter.
    
    Args:
        y: Input signal.
        sr: Sampling rate in Hz.
        cutoff_freq: Cutoff frequency in Hz.
        order: Filter order.
    
    Returns:
        Filtered signal.
    """
    nyquist = sr / 2.0
    normalized_cutoff = cutoff_freq / nyquist
    
    if normalized_cutoff >= 1.0:
        return y
    
    sos = scipy.signal.butter(order, normalized_cutoff, btype='low', output='sos')
    return scipy.signal.sosfiltfilt(sos, y)


def _compute_hilbert_envelope(y: np.ndarray, smooth_ms: float = 0.5, sr: int = 44100) -> np.ndarray:
    """
    Compute Hilbert envelope with optional smoothing.
    
    Args:
        y: Input signal.
        smooth_ms: Smoothing window in milliseconds. If <= 0, no smoothing.
        sr: Sampling rate in Hz.
    
    Returns:
        Hilbert envelope.
    """
    if len(y) == 0:
        return np.array([])
    
    analytic = scipy.signal.hilbert(y)
    env = np.abs(analytic)
    
    # Optional smoothing
    if smooth_ms > 0:
        win_samples = max(int(smooth_ms * sr / 1000), 1)
        if win_samples >= 2 and win_samples < len(env):
            kernel = np.ones(win_samples) / win_samples
            env = np.convolve(env, kernel, mode='same')
    
    return env


def _compute_adaptive_threshold(
    data: np.ndarray,
    method: str = 'median_mad',
    percentile: float = 75.0,
) -> float:
    """
    Compute adaptive threshold using median/MAD or percentile-based method.
    
    Args:
        data: Input data array.
        method: 'median_mad' or 'percentile'.
        percentile: Percentile value if method is 'percentile'.
    
    Returns:
        Adaptive threshold value.
    """
    if len(data) == 0:
        return 0.0
    
    if method == 'median_mad':
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        # MAD-based threshold: median + 3*MAD (robust to outliers)
        return median + 3 * mad
    else:  # percentile
        return np.percentile(data, percentile)


def detect_t_start_fujii(
    segment: np.ndarray,
    sr: int,
    *,
    hpf_cutoff: float = 300.0,
    smooth_ms: float = 0.5,
    threshold_ratio: float = 0.1,
) -> tuple[float, int, np.ndarray]:
    """
    Detect T_start (consonant onset /t/) using the Fujii 10% method.
    
    Algorithm:
    1. Compute high-frequency envelope (HPF → absolute value → smoothing OR Hilbert envelope)
    2. Let E_peak = max envelope value within the segment
    3. Threshold = 0.1 * E_peak (10%)
    4. Let t_peak_frame be the frame where E_peak occurs
    5. Search backward from t_peak_frame until envelope goes below threshold
    6. Perform linear interpolation between last sample above 10% and first sample below 10%
    7. T_start = interpolated time (in seconds)
    
    Args:
        segment: Audio segment (mono, float).
        sr: Sampling rate in Hz.
        hpf_cutoff: High-pass filter cutoff in Hz (default: 300 Hz).
        smooth_ms: Envelope smoothing window in milliseconds.
        threshold_ratio: Threshold as fraction of peak (default: 0.1 = 10%).
    
    Returns:
        Tuple of (t_start_sec, peak_idx, envelope):
        - t_start_sec: T_start time in seconds (relative to segment start)
        - peak_idx: Index of envelope peak
        - envelope: The computed high-frequency envelope
    """
    if len(segment) == 0:
        return 0.0, 0, np.array([])
    
    # Apply high-pass filter
    y_hpf = _highpass_filter(segment, sr, hpf_cutoff)
    
    # Compute Hilbert envelope with smoothing
    env = _compute_hilbert_envelope(y_hpf, smooth_ms=smooth_ms, sr=sr)
    
    if len(env) == 0:
        return 0.0, 0, env
    
    # Find peak (E_peak)
    peak_idx = int(np.argmax(env))
    e_peak = env[peak_idx]
    
    if e_peak == 0:
        return 0.0, peak_idx, env
    
    # Compute threshold (10% of peak)
    threshold = threshold_ratio * e_peak
    
    # Search backward from peak to find threshold crossing
    onset_idx = peak_idx
    for k in range(peak_idx - 1, -1, -1):
        if env[k] <= threshold:
            # Found threshold crossing between k and k+1
            # Perform linear interpolation
            e0 = env[k]      # Below threshold
            e1 = env[k + 1]  # Above threshold
            
            if e1 > e0:
                # Linear interpolation: find exact crossing point
                alpha = (threshold - e0) / (e1 - e0)
                onset_idx = k + alpha
            else:
                onset_idx = k + 1
            break
        onset_idx = k
    
    # Convert to time in seconds
    t_start_sec = float(onset_idx) / sr
    
    return t_start_sec, peak_idx, env


def detect_t_peak(
    segment: np.ndarray,
    sr: int,
    *,
    hpf_cutoff: float = 300.0,
    smooth_ms: float = 0.5,
    search_window_ms: float = 50.0,
) -> tuple[float, np.ndarray]:
    """
    Detect T_peak (burst maximum of /t/).
    
    Uses high-frequency envelope to find the maximum energy point.
    By default, searches within the first 30-50 ms of the segment where
    the consonant burst is expected to occur.
    
    Args:
        segment: Audio segment (mono, float).
        sr: Sampling rate in Hz.
        hpf_cutoff: High-pass filter cutoff in Hz.
        smooth_ms: Envelope smoothing window in milliseconds.
        search_window_ms: Time window to search for peak (from segment start).
    
    Returns:
        Tuple of (t_peak_sec, envelope):
        - t_peak_sec: T_peak time in seconds (relative to segment start)
        - envelope: The computed high-frequency envelope
    """
    if len(segment) == 0:
        return 0.0, np.array([])
    
    # Apply high-pass filter
    y_hpf = _highpass_filter(segment, sr, hpf_cutoff)
    
    # Compute Hilbert envelope
    env = _compute_hilbert_envelope(y_hpf, smooth_ms=smooth_ms, sr=sr)
    
    if len(env) == 0:
        return 0.0, env
    
    # Determine search window
    search_samples = int(search_window_ms * sr / 1000)
    search_end = min(search_samples, len(env))
    
    if search_end > 0:
        # Find peak within search window
        peak_idx = int(np.argmax(env[:search_end]))
    else:
        # Fallback: search entire segment
        peak_idx = int(np.argmax(env))
    
    t_peak_sec = peak_idx / sr
    
    return t_peak_sec, env


def detect_a_start_periodic(
    segment: np.ndarray,
    sr: int,
    *,
    f0_min: float = 50.0,
    f0_max: float = 500.0,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 2.0,
    min_voiced_frames: int = 3,
    search_start_sec: float = 0.0,
) -> tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect A_start (vowel onset /a/) using periodicity-based voicing detection.
    
    Algorithm:
    1. Pre-process with bandpass filter focusing on F0 band (50-500 Hz)
    2. Use short-time analysis with frame_length ~20-30ms and hop_length ~1-2ms
    3. For each frame, compute RMS and periodicity (using YIN or autocorrelation)
    4. Compute adaptive thresholds using median/MAD/percentiles
    5. A frame is "voiced" if RMS > energy_threshold AND periodicity > periodicity_threshold
    6. A_start = start time of first frame that becomes voiced and remains voiced for ≥3-5 frames
    
    Args:
        segment: Audio segment (mono, float).
        sr: Sampling rate in Hz.
        f0_min: Minimum F0 frequency for voicing detection (Hz).
        f0_max: Maximum F0 frequency for voicing detection (Hz).
        frame_length_ms: Frame length in milliseconds.
        hop_length_ms: Hop length in milliseconds.
        min_voiced_frames: Minimum consecutive voiced frames for A_start.
        search_start_sec: Start searching from this time (seconds from segment start).
    
    Returns:
        Tuple of (a_start_sec, f0_array, voiced_flags):
        - a_start_sec: A_start time in seconds (relative to segment start), or None
        - f0_array: Array of F0 values per frame (NaN for unvoiced)
        - voiced_flags: Boolean array of voiced/unvoiced per frame
    """
    if len(segment) == 0:
        return 0.0, None, None
    
    # Convert to samples
    frame_length = int(frame_length_ms * sr / 1000)
    hop_length = int(hop_length_ms * sr / 1000)
    search_start_sample = int(search_start_sec * sr)
    
    # Ensure minimum values
    frame_length = max(frame_length, int(2 * sr / f0_min))  # At least 2 periods of min F0
    hop_length = max(hop_length, 1)
    
    # Pre-process: bandpass filter for F0 band
    y_bp = _bandpass_filter(segment, sr, f0_min, f0_max)
    
    # Compute frame-wise RMS and periodicity
    n_frames = max(1, (len(y_bp) - frame_length) // hop_length + 1)
    
    rms_values = np.zeros(n_frames)
    periodicity_values = np.zeros(n_frames)
    f0_values = np.zeros(n_frames)
    f0_values[:] = np.nan
    
    # Compute per-frame features
    for i in range(n_frames):
        start = i * hop_length
        end = min(start + frame_length, len(y_bp))
        frame = y_bp[start:end]
        
        if len(frame) < frame_length // 2:
            continue
        
        # RMS
        rms_values[i] = np.sqrt(np.mean(frame ** 2))
        
        # Periodicity using normalized autocorrelation
        # Find autocorrelation peak in F0 range
        min_lag = max(int(sr / f0_max), 1)
        max_lag = min(int(sr / f0_min), len(frame) - 1)
        
        if max_lag > min_lag and len(frame) > max_lag:
            # Compute autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(frame) - 1:]  # Take positive lags only
            
            # Normalize
            if autocorr[0] > 0:
                autocorr_norm = autocorr / autocorr[0]
            else:
                autocorr_norm = autocorr
            
            # Find peak in F0 range
            if max_lag < len(autocorr_norm):
                search_region = autocorr_norm[min_lag:max_lag + 1]
                if len(search_region) > 0:
                    peak_idx = np.argmax(search_region)
                    periodicity_values[i] = search_region[peak_idx]
                    
                    # Estimate F0 from peak lag
                    lag = min_lag + peak_idx
                    if lag > 0:
                        f0_values[i] = sr / lag
    
    # Compute adaptive thresholds
    valid_rms = rms_values[rms_values > 0]
    if len(valid_rms) > 0:
        rms_threshold = _compute_adaptive_threshold(valid_rms, method='median_mad')
    else:
        rms_threshold = 0.0
    
    valid_periodicity = periodicity_values[periodicity_values > 0]
    if len(valid_periodicity) > 0:
        # Use median as threshold for periodicity (should be > 0.5 for voiced)
        periodicity_threshold = max(0.3, np.median(valid_periodicity) * 0.5)
    else:
        periodicity_threshold = 0.3
    
    # Determine voiced frames
    voiced_flags = (rms_values > rms_threshold) & (periodicity_values > periodicity_threshold)
    
    # Find first sustained voiced region
    search_start_frame = max(0, int(search_start_sample / hop_length))
    a_start_frame = None
    
    for i in range(search_start_frame, len(voiced_flags) - min_voiced_frames + 1):
        # Check if this frame and next min_voiced_frames-1 frames are all voiced
        if np.all(voiced_flags[i:i + min_voiced_frames]):
            a_start_frame = i
            break
    
    if a_start_frame is None:
        # Fallback: find first voiced frame even if not sustained
        for i in range(search_start_frame, len(voiced_flags)):
            if voiced_flags[i]:
                a_start_frame = i
                break
    
    if a_start_frame is None:
        # Last fallback: use middle of segment
        a_start_sec = len(segment) / sr / 2
    else:
        a_start_sec = a_start_frame * hop_length / sr
    
    return a_start_sec, f0_values, voiced_flags


def detect_a_peak_periodic(
    segment: np.ndarray,
    sr: int,
    a_start_sec: float,
    *,
    f0_min: float = 50.0,
    f0_max: float = 500.0,
    search_window_ms: float = 30.0,
) -> float:
    """
    Detect A_peak (first stable periodic peak after A_start).
    
    After A_start is found, refine within the low-frequency filtered waveform
    to identify the first cycle of periodic vibration (zero-crossing → local max
    or local maxima matching estimated F0 period).
    
    Args:
        segment: Audio segment (mono, float).
        sr: Sampling rate in Hz.
        a_start_sec: A_start time in seconds (relative to segment start).
        f0_min: Minimum F0 frequency (Hz).
        f0_max: Maximum F0 frequency (Hz).
        search_window_ms: Time window to search for first peak after A_start.
    
    Returns:
        A_peak time in seconds (relative to segment start).
        Typically 5-15 ms after A_start.
    """
    if len(segment) == 0:
        return a_start_sec
    
    # Convert to samples
    a_start_sample = int(a_start_sec * sr)
    search_window_samples = int(search_window_ms * sr / 1000)
    
    # Ensure we're within bounds
    a_start_sample = max(0, min(a_start_sample, len(segment) - 1))
    search_end = min(a_start_sample + search_window_samples, len(segment))
    
    if search_end <= a_start_sample:
        return a_start_sec
    
    # Low-pass filter to focus on F0 band
    y_lp = _lowpass_filter(segment, sr, f0_max)
    
    # Extract search region
    search_region = y_lp[a_start_sample:search_end]
    
    if len(search_region) < 3:
        return a_start_sec
    
    # Find local maxima in the search region
    # Use scipy.signal.find_peaks for robust peak detection
    peaks, properties = scipy.signal.find_peaks(search_region)
    
    if len(peaks) == 0:
        # No peaks found - try finding the global max
        max_idx = np.argmax(search_region)
        a_peak_sample = a_start_sample + max_idx
    else:
        # Use the first peak as A_peak
        a_peak_sample = a_start_sample + peaks[0]
    
    # Convert to seconds
    a_peak_sec = a_peak_sample / sr
    
    # Ensure A_peak is after A_start (with small tolerance)
    if a_peak_sec < a_start_sec:
        # Typical offset is 5-15ms after A_start
        a_peak_sec = a_start_sec + 0.010  # 10ms default offset
    
    return a_peak_sec


def extract_ta_feature_points(
    y: np.ndarray,
    sr: int,
    segment_start: int,
    segment_end: int,
    *,
    hpf_cutoff: float = 300.0,
    smooth_ms: float = 0.5,
    threshold_ratio: float = 0.1,
    f0_min: float = 50.0,
    f0_max: float = 500.0,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 2.0,
    min_voiced_frames: int = 3,
    t_peak_search_ms: float = 50.0,
    a_peak_search_ms: float = 30.0,
) -> dict[str, float]:
    """
    Extract all 5 feature points for a single "ta" segment.
    
    Feature points:
    1. t_start: Consonant onset /t/ (Fujii 10% method)
    2. t_peak: Burst maximum of /t/
    3. a_start: Vowel onset /a/ (periodicity-based voicing detection)
    4. a_peak: First stable periodic peak after a_start
    5. a_end: End of vowel /a/ (segment end)
    
    Args:
        y: Full audio signal.
        sr: Sampling rate in Hz.
        segment_start: Start sample index of the segment.
        segment_end: End sample index of the segment.
        hpf_cutoff: High-pass filter cutoff for T detection (Hz).
        smooth_ms: Envelope smoothing window (ms).
        threshold_ratio: Fujii threshold ratio (default 0.1 = 10%).
        f0_min: Minimum F0 for voicing detection (Hz).
        f0_max: Maximum F0 for voicing detection (Hz).
        frame_length_ms: Frame length for voicing analysis (ms).
        hop_length_ms: Hop length for voicing analysis (ms).
        min_voiced_frames: Minimum consecutive voiced frames for A_start.
        t_peak_search_ms: Search window for T_peak (ms).
        a_peak_search_ms: Search window for A_peak after A_start (ms).
    
    Returns:
        Dictionary with feature point times in seconds:
        {'t_start': float, 't_peak': float, 'a_start': float, 'a_peak': float, 'a_end': float}
    """
    # Extract segment
    segment_start = max(0, segment_start)
    segment_end = min(len(y), segment_end)
    
    if segment_start >= segment_end:
        base_time = segment_start / sr
        return {
            't_start': base_time,
            't_peak': base_time,
            'a_start': base_time,
            'a_peak': base_time,
            'a_end': base_time,
        }
    
    segment = y[segment_start:segment_end]
    segment_duration = len(segment) / sr
    
    # 1. Detect T_start using Fujii 10% method
    t_start_rel, peak_idx, hf_env = detect_t_start_fujii(
        segment, sr,
        hpf_cutoff=hpf_cutoff,
        smooth_ms=smooth_ms,
        threshold_ratio=threshold_ratio,
    )
    t_start_sec = segment_start / sr + t_start_rel
    
    # 2. Detect T_peak
    t_peak_rel, _ = detect_t_peak(
        segment, sr,
        hpf_cutoff=hpf_cutoff,
        smooth_ms=smooth_ms,
        search_window_ms=t_peak_search_ms,
    )
    t_peak_sec = segment_start / sr + t_peak_rel
    
    # Ensure t_peak is after or at t_start
    if t_peak_sec < t_start_sec:
        t_peak_sec = t_start_sec + 0.001  # 1ms offset
    
    # 3. Detect A_start using periodicity-based voicing detection
    # Start searching after T_peak to find vowel onset
    search_start = max(0.0, t_peak_rel)
    a_start_rel, f0_values, voiced_flags = detect_a_start_periodic(
        segment, sr,
        f0_min=f0_min,
        f0_max=f0_max,
        frame_length_ms=frame_length_ms,
        hop_length_ms=hop_length_ms,
        min_voiced_frames=min_voiced_frames,
        search_start_sec=search_start,
    )
    a_start_sec = segment_start / sr + a_start_rel
    
    # Ensure a_start is after t_start
    if a_start_sec <= t_start_sec:
        a_start_sec = t_start_sec + 0.005  # 5ms offset
    
    # 4. Detect A_peak (first stable periodic peak after A_start)
    a_peak_rel = detect_a_peak_periodic(
        segment, sr,
        a_start_rel,
        f0_min=f0_min,
        f0_max=f0_max,
        search_window_ms=a_peak_search_ms,
    )
    a_peak_sec = segment_start / sr + a_peak_rel
    
    # Ensure a_peak is after a_start
    if a_peak_sec <= a_start_sec:
        a_peak_sec = a_start_sec + 0.010  # 10ms typical offset
    
    # 5. A_end = segment end
    a_end_sec = segment_end / sr
    
    # Ensure a_peak is before a_end
    if a_peak_sec >= a_end_sec:
        a_peak_sec = (a_start_sec + a_end_sec) / 2
    
    return {
        't_start': t_start_sec,
        't_peak': t_peak_sec,
        'a_start': a_start_sec,
        'a_peak': a_peak_sec,
        'a_end': a_end_sec,
    }
