#!/usr/bin/env python3
"""
Envelope comparison script for onset detection parameter sweeps.

This script compares different envelope configurations side-by-side:
- Hilbert (smoothed/unsmoothed, varying smoothing windows)
- RMS-based envelopes
- Different HPF cutoffs
- Different global peak filters
- Different minimum inter-event distances
- Lookback quiet-period criteria

Provides reproducible quantitative metrics for each variant.
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib import cm
import envelope_variants as ev


def get_git_commit_hash() -> str:
    """
    Get current git commit hash for reproducibility.

    Returns:
        commit_hash: Short commit hash or "unknown" if git unavailable.
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def parse_comma_list(s: str) -> List[float]:
    """
    Parse comma-separated numeric list.

    Args:
        s: string like "0,300,500,1000"

    Returns:
        List of floats
    """
    return [float(x.strip()) for x in s.split(',')]


def load_audio(wav_path: str, target_sr: int | None = None) -> Tuple[np.ndarray, int]:
    """
    Load audio file, optionally resampling.

    Args:
        wav_path: path to WAV file.
        target_sr: target sampling rate. If None, keep native.

    Returns:
        y: mono audio signal.
        sr: sampling rate.
    """
    data, sr = sf.read(wav_path)

    # Convert to mono if needed
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if requested
    if target_sr is not None and target_sr != sr:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return data, sr


def generate_variants(
    y: np.ndarray,
    sr: int,
    hpf_cutoffs: List[float],
    smooth_ms_list: List[float],
    global_min_height_ratios: List[float],
    min_distance_ms_list: List[float],
    lookback_points_list: List[int],
    rms_frame_ms: float,
    rms_hop_ms: float,
    threshold_ratio: float,
) -> List[dict]:
    """
    Generate all envelope variants and detect onsets.

    Uses caching: filters once per HPF cutoff, then reuses for multiple variants.

    Args:
        y: audio signal.
        sr: sampling rate.
        hpf_cutoffs: list of HPF cutoffs to test.
        smooth_ms_list: list of smoothing windows for Hilbert.
        global_min_height_ratios: list of global height ratios.
        min_distance_ms_list: list of min distance values.
        lookback_points_list: list of lookback points.
        rms_frame_ms: RMS frame length.
        rms_hop_ms: RMS hop length.
        threshold_ratio: onset threshold ratio (Fujii 10%).

    Returns:
        List of variant dicts with keys:
            variant_id, envelope_type, env, times, onset_times, peak_times, params
    """
    variants = []
    hpf_cache = {}  # Cache filtered signals

    total_variants = 0
    # Count total variants
    for hpf in hpf_cutoffs:
        # Hilbert variants
        total_variants += len(smooth_ms_list) * len(global_min_height_ratios) * len(min_distance_ms_list) * len(lookback_points_list)
        # RMS variants
        total_variants += len(global_min_height_ratios) * len(min_distance_ms_list) * len(lookback_points_list)

    print(f"\nGenerating {total_variants} envelope variants...")
    variant_count = 0

    for hpf_cutoff in hpf_cutoffs:
        # Cache filtered signal for this HPF cutoff
        if hpf_cutoff not in hpf_cache:
            if hpf_cutoff is None or hpf_cutoff == 0:
                hpf_cache[hpf_cutoff] = y
            else:
                from scipy.signal import butter, sosfiltfilt
                sos = butter(4, hpf_cutoff, btype='hp', fs=sr, output='sos')
                hpf_cache[hpf_cutoff] = sosfiltfilt(sos, y)

        y_filt = hpf_cache[hpf_cutoff]

        # Hilbert variants
        for smooth_ms in smooth_ms_list:
            # Compute envelope
            env = ev.compute_hilbert_envelope_variant(
                y_filt, sr, hpf_cutoff=None, smooth_ms=smooth_ms
            )
            times = np.arange(len(env)) / sr

            # Detect with all parameter combinations
            for gmin in global_min_height_ratios:
                for mindist in min_distance_ms_list:
                    for lookback in lookback_points_list:
                        onset_times, peak_times = ev.detect_events_fujii(
                            env, sr,
                            threshold_ratio=threshold_ratio,
                            global_min_height_ratio=gmin,
                            min_distance_ms=mindist
                        )

                        # Apply lookback filter if requested
                        if lookback > 0 and len(onset_times) > 0:
                            filtered_onsets = []
                            filtered_peaks = []
                            for ot, pt in zip(onset_times, peak_times):
                                onset_idx = ot * sr
                                peak_idx = int(pt * sr)
                                th = threshold_ratio * env[peak_idx]
                                adjusted = ev.apply_lookback_quiet_period(
                                    env, onset_idx, peak_idx, lookback, th
                                )
                                if adjusted is not None:
                                    filtered_onsets.append(ot)
                                    filtered_peaks.append(pt)
                            onset_times = np.array(filtered_onsets)
                            peak_times = np.array(filtered_peaks)

                        variant_id = ev.build_variant_id(
                            "hilbert", hpf_cutoff, smooth_ms, gmin, mindist, lookback
                        )

                        variants.append({
                            'variant_id': variant_id,
                            'envelope_type': 'hilbert',
                            'env': env,
                            'times': times,
                            'onset_times': onset_times,
                            'peak_times': peak_times,
                            'params': {
                                'hpf_cutoff': hpf_cutoff if hpf_cutoff else 0,
                                'smooth_ms': smooth_ms if smooth_ms else 0,
                                'global_min_height_ratio': gmin,
                                'min_distance_ms': mindist,
                                'lookback_points': lookback,
                                'threshold_ratio': threshold_ratio,
                            }
                        })

                        variant_count += 1
                        if variant_count % 10 == 0 or variant_count == total_variants:
                            print(f"  Progress: {variant_count}/{total_variants} variants processed")

        # RMS variants
        env_rms, times_rms = ev.compute_rms_envelope_variant(
            y_filt, sr, hpf_cutoff=None, frame_ms=rms_frame_ms, hop_ms=rms_hop_ms
        )

        # Need to upsample RMS envelope for detection at original sample rate
        # For simplicity, interpolate to match signal length
        env_rms_interp = np.interp(
            np.arange(len(y)) / sr,
            times_rms,
            env_rms
        )
        times_rms_full = np.arange(len(y)) / sr

        for gmin in global_min_height_ratios:
            for mindist in min_distance_ms_list:
                for lookback in lookback_points_list:
                    onset_times, peak_times = ev.detect_events_fujii(
                        env_rms_interp, sr,
                        threshold_ratio=threshold_ratio,
                        global_min_height_ratio=gmin,
                        min_distance_ms=mindist
                    )

                    # Apply lookback filter if requested
                    if lookback > 0 and len(onset_times) > 0:
                        filtered_onsets = []
                        filtered_peaks = []
                        for ot, pt in zip(onset_times, peak_times):
                            onset_idx = ot * sr
                            peak_idx = int(pt * sr)
                            th = threshold_ratio * env_rms_interp[peak_idx]
                            adjusted = ev.apply_lookback_quiet_period(
                                env_rms_interp, onset_idx, peak_idx, lookback, th
                            )
                            if adjusted is not None:
                                filtered_onsets.append(ot)
                                filtered_peaks.append(pt)
                        onset_times = np.array(filtered_onsets)
                        peak_times = np.array(filtered_peaks)

                    variant_id = ev.build_variant_id(
                        "rms", hpf_cutoff, None, gmin, mindist, lookback,
                        frame_ms=rms_frame_ms, hop_ms=rms_hop_ms
                    )

                    variants.append({
                        'variant_id': variant_id,
                        'envelope_type': 'rms',
                        'env': env_rms_interp,
                        'times': times_rms_full,
                        'onset_times': onset_times,
                        'peak_times': peak_times,
                        'params': {
                            'hpf_cutoff': hpf_cutoff if hpf_cutoff else 0,
                            'smooth_ms': 0,
                            'global_min_height_ratio': gmin,
                            'min_distance_ms': mindist,
                            'lookback_points': lookback,
                            'threshold_ratio': threshold_ratio,
                            'rms_frame_ms': rms_frame_ms,
                            'rms_hop_ms': rms_hop_ms,
                        }
                    })

                    variant_count += 1
                    if variant_count % 10 == 0 or variant_count == total_variants:
                        print(f"  Progress: {variant_count}/{total_variants} variants processed")

    print(f"Generated {len(variants)} variants successfully.\n")
    return variants


def compute_variant_metrics(variant: dict, sr: int) -> dict:
    """
    Compute metrics for a single variant.

    Args:
        variant: variant dict with env, onset_times, peak_times, params.
        sr: sampling rate.

    Returns:
        metrics dict with computed values.
    """
    env = variant['env']
    onset_times = variant['onset_times']
    peak_times = variant['peak_times']

    n_events = len(onset_times)

    if n_events == 0:
        return {
            'variant_id': variant['variant_id'],
            'n_events': 0,
            'mean_onset_to_peak_ms': 0.0,
            'rise_time_10_90_ms': 0.0,
            'median_rise_time_ms': 0.0,
            'slope_10_90': 0.0,
            **variant['params']
        }

    # Calculate onset-to-peak times
    onset_to_peak_ms = (peak_times - onset_times) * 1000.0

    # Calculate rise times and slopes for each event
    rise_times = []
    slopes = []

    for ot, pt in zip(onset_times, peak_times):
        onset_idx = ot * sr
        peak_idx = int(pt * sr)

        _, rise_ms = ev.calc_rise_time(env, onset_idx, peak_idx, sr)
        rise_times.append(rise_ms)

        slope = ev.calc_slope(env, onset_idx, peak_idx, sr)
        slopes.append(slope)

    rise_times = np.array(rise_times)
    slopes = np.array(slopes)

    return {
        'variant_id': variant['variant_id'],
        'n_events': n_events,
        'mean_onset_to_peak_ms': float(np.mean(onset_to_peak_ms)),
        'rise_time_10_90_ms': float(np.mean(rise_times)),
        'median_rise_time_ms': float(np.median(rise_times)),
        'slope_10_90': float(np.mean(slopes)),
        **variant['params']
    }


def plot_comparison(
    y: np.ndarray,
    sr: int,
    variants: List[dict],
    out_path: str | None = None,
    max_variants_to_plot: int = 20,
):
    """
    Create multi-panel comparison plot.

    Args:
        y: audio signal.
        sr: sampling rate.
        variants: list of variant dicts.
        out_path: output path for plot. If None, display only.
        max_variants_to_plot: limit number of variants in plot.
    """
    # Limit variants for readability
    if len(variants) > max_variants_to_plot:
        print(f"  Note: Plotting first {max_variants_to_plot} of {len(variants)} variants")
        variants = variants[:max_variants_to_plot]

    # Use colorblind-friendly palette
    colors = plt.cm.tab20(np.linspace(0, 1, len(variants)))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X', 'P', 'h'] * 3

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top panel: waveform
    t = np.arange(len(y)) / sr
    axes[0].plot(t, y, color='gray', alpha=0.5, linewidth=0.5, label='Waveform')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Waveform with Detected Onsets')
    axes[0].grid(True, alpha=0.3)

    # Bottom panel: envelopes
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Envelope Amplitude')
    axes[1].set_title('Envelope Variants Comparison')
    axes[1].grid(True, alpha=0.3)

    # Plot each variant
    for i, variant in enumerate(variants):
        color = colors[i]
        marker = markers[i % len(markers)]
        label = variant['variant_id']

        # Plot envelope (subsample for readability if long)
        times = variant['times']
        env = variant['env']
        if len(times) > 10000:
            step = len(times) // 5000
            axes[1].plot(times[::step], env[::step], color=color, alpha=0.6, linewidth=1, label=label)
        else:
            axes[1].plot(times, env, color=color, alpha=0.6, linewidth=1, label=label)

        # Plot onsets on both panels
        for j, ot in enumerate(variant['onset_times']):
            axes[0].axvline(ot, color=color, linestyle='--', alpha=0.5, linewidth=0.8)
            axes[1].scatter([ot], [env[int(ot * sr)]], color=color, marker=marker, s=40, zorder=10)

    # Legends
    axes[0].legend(loc='upper right', fontsize=8)
    axes[1].legend(loc='upper right', fontsize=6, ncol=2)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"  Saved plot to: {out_path}")
        # Also save SVG
        svg_path = out_path.replace('.png', '.svg')
        plt.savefig(svg_path)
        print(f"  Saved SVG to: {svg_path}")
    else:
        plt.show()

    plt.close()


def save_metrics_csv(metrics_list: List[dict], out_path: str, commit_hash: str):
    """
    Save metrics to CSV with metadata header.

    Args:
        metrics_list: list of metrics dicts.
        out_path: output CSV path.
        commit_hash: git commit hash for reproducibility.
    """
    if len(metrics_list) == 0:
        print("  Warning: No metrics to save.")
        return

    df = pd.DataFrame(metrics_list)

    # Write metadata as comments
    with open(out_path, 'w') as f:
        f.write(f"# Envelope Comparison Metrics\n")
        f.write(f"# Git commit: {commit_hash}\n")
        f.write(f"# Total variants: {len(metrics_list)}\n")
        f.write("#\n")

    # Append DataFrame
    df.to_csv(out_path, mode='a', index=False)
    print(f"  Saved metrics to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare onset envelope variants with parameter sweeps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_envelopes.py --wav audio.wav
  python compare_envelopes.py --wav audio.wav --hpf_cutoffs 0,300,500 --export_plots
  python compare_envelopes.py --wav audio.wav --smooth_ms 0,0.5,1.0 --out_dir results
        """
    )

    parser.add_argument('--wav', required=True, help='Path to WAV file')
    parser.add_argument('--sr', type=int, default=None, help='Target sampling rate (default: keep native)')
    parser.add_argument('--hpf_cutoffs', type=str, default='0,300,500',
                       help='Comma-separated HPF cutoffs in Hz (0=no HPF)')
    parser.add_argument('--smooth_ms', type=str, default='0,0.5',
                       help='Comma-separated smoothing windows in ms (0=no smoothing)')
    parser.add_argument('--global_min_height_ratios', type=str, default='0,0.2',
                       help='Comma-separated global min height ratios')
    parser.add_argument('--min_distance_ms', type=str, default='50,100',
                       help='Comma-separated min distance values in ms')
    parser.add_argument('--lookback_points', type=str, default='0',
                       help='Comma-separated lookback points (0=disabled)')
    parser.add_argument('--rms_frame_ms', type=float, default=5.0,
                       help='RMS frame length in ms')
    parser.add_argument('--rms_hop_ms', type=float, default=1.0,
                       help='RMS hop length in ms')
    parser.add_argument('--threshold_ratio', type=float, default=0.1,
                       help='Onset threshold ratio (Fujii 10%%)')
    parser.add_argument('--export_plots', action='store_true',
                       help='Export plots to files')
    parser.add_argument('--out_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--label', type=str, default='',
                       help='Optional label for this analysis (tap/click/generic)')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.wav):
        print(f"Error: WAV file not found: {args.wav}")
        sys.exit(1)

    # Parse parameter lists
    try:
        hpf_cutoffs = parse_comma_list(args.hpf_cutoffs)
        smooth_ms_list = parse_comma_list(args.smooth_ms)
        global_min_height_ratios = parse_comma_list(args.global_min_height_ratios)
        min_distance_ms_list = parse_comma_list(args.min_distance_ms)
        lookback_points_list = [int(x) for x in args.lookback_points.split(',')]
    except ValueError as e:
        print(f"Error parsing parameters: {e}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Get git commit hash
    commit_hash = get_git_commit_hash()

    print("=" * 80)
    print("ENVELOPE COMPARISON FRAMEWORK")
    print("=" * 80)
    print(f"Input file: {args.wav}")
    print(f"Output directory: {args.out_dir}")
    print(f"Git commit: {commit_hash}")
    if args.label:
        print(f"Label: {args.label}")

    # Load audio
    print("\nLoading audio...")
    y, sr = load_audio(args.wav, args.sr)
    duration = len(y) / sr
    print(f"  Loaded: {len(y)} samples @ {sr} Hz ({duration:.2f} sec)")

    # Edge case: very short audio
    if duration < 0.05:
        print("  Warning: Audio is very short (<50ms). Results may be unreliable.")

    # Edge case: silence
    if np.max(np.abs(y)) < 1e-6:
        print("  Warning: Audio appears to be silence. No onsets will be detected.")

    # Generate variants
    variants = generate_variants(
        y, sr,
        hpf_cutoffs, smooth_ms_list,
        global_min_height_ratios, min_distance_ms_list,
        lookback_points_list,
        args.rms_frame_ms, args.rms_hop_ms,
        args.threshold_ratio
    )

    # Compute metrics
    print("Computing metrics...")
    metrics_list = []
    for variant in variants:
        metrics = compute_variant_metrics(variant, sr)
        metrics_list.append(metrics)

    # Display summary
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    df = pd.DataFrame(metrics_list)
    print(df.to_string(index=False))

    # Save CSV
    base_name = Path(args.wav).stem
    label_suffix = f"_{args.label}" if args.label else ""
    csv_path = os.path.join(args.out_dir, f"{base_name}{label_suffix}_metrics.csv")
    save_metrics_csv(metrics_list, csv_path, commit_hash)

    # Plot
    if args.export_plots:
        print("\nGenerating comparison plot...")
        plot_path = os.path.join(args.out_dir, f"{base_name}{label_suffix}_comparison.png")
        plot_comparison(y, sr, variants, plot_path)
    else:
        print("\nSkipping plot export (use --export_plots to enable)")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
