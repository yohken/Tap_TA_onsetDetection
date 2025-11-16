#!/usr/bin/env python3
"""
Example script demonstrating the unified onset + peak detection module.

This script shows how to use onset_hilbert.py for detecting onsets and peaks
in metronome clicks and finger taps.
"""

import numpy as np
import scipy.io.wavfile as wavfile
import onset_hilbert
import tempfile
import os


def example_1_basic_click_detection():
    """Example 1: Basic click detection with synthetic audio."""
    print("\n" + "="*70)
    print("Example 1: Basic Click Detection")
    print("="*70)
    
    # Create synthetic click track
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.zeros_like(t)
    
    # Add 4 clicks at known times
    click_times = [0.3, 0.8, 1.3, 1.8]
    print(f"Creating synthetic clicks at: {click_times} seconds")
    
    for click_time in click_times:
        click_duration = 0.02
        click_samples = int(click_duration * sr)
        click_start = int(click_time * sr)
        
        decay = np.exp(-np.linspace(0, 8, click_samples))
        click_sound = decay * 0.8
        
        if click_start + click_samples < len(audio):
            audio[click_start:click_start + click_samples] += click_sound
    
    audio += np.random.randn(len(audio)) * 0.005
    audio = audio / np.max(np.abs(audio)) * 0.9
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        wav_path = f.name
        wavfile.write(wav_path, sr, audio_int16)
    
    try:
        # Detect onsets and peaks
        onset_times, peak_times = onset_hilbert.detect_click_onsets_and_peaks(
            wav_path,
            hp_cutoff_hz=1000.0,
            min_distance_ms=100.0
        )
        
        print(f"\nDetected {len(onset_times)} clicks:")
        for i, (ot, pt) in enumerate(zip(onset_times, peak_times)):
            expected = click_times[i] if i < len(click_times) else None
            if expected:
                error_ms = abs(ot - expected) * 1000
                print(f"  Click {i+1}: onset={ot:.4f}s, peak={pt:.4f}s, "
                      f"expected={expected:.3f}s, error={error_ms:.2f}ms")
            else:
                print(f"  Click {i+1}: onset={ot:.4f}s, peak={pt:.4f}s")
    finally:
        os.remove(wav_path)
    
    print("✓ Example 1 complete\n")


def example_2_tap_detection():
    """Example 2: Tap detection with different HPF cutoff."""
    print("\n" + "="*70)
    print("Example 2: Tap Detection")
    print("="*70)
    
    # Create synthetic tap audio with lower frequencies
    sr = 48000
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.zeros_like(t)
    
    # Add 3 taps
    tap_times = [0.3, 0.7, 1.1]
    print(f"Creating synthetic taps at: {tap_times} seconds")
    
    for tap_time in tap_times:
        tap_duration = 0.03
        tap_samples = int(tap_duration * sr)
        tap_start = int(tap_time * sr)
        
        decay = np.exp(-np.linspace(0, 6, tap_samples))
        noise = np.random.randn(tap_samples)
        tap_sound = noise * decay * 0.7
        
        if tap_start + tap_samples < len(audio):
            audio[tap_start:tap_start + tap_samples] += tap_sound
    
    audio += np.random.randn(len(audio)) * 0.01
    audio = audio / np.max(np.abs(audio)) * 0.85
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        wav_path = f.name
        wavfile.write(wav_path, sr, audio_int16)
    
    try:
        # Detect with lower HPF cutoff (300 Hz for taps)
        onset_times, peak_times = onset_hilbert.detect_tap_onsets_and_peaks(
            wav_path,
            hp_cutoff_hz=300.0,
            min_distance_ms=100.0
        )
        
        print(f"\nDetected {len(onset_times)} taps:")
        for i, (ot, pt) in enumerate(zip(onset_times, peak_times)):
            expected = tap_times[i] if i < len(tap_times) else None
            if expected:
                error_ms = abs(ot - expected) * 1000
                print(f"  Tap {i+1}: onset={ot:.4f}s, peak={pt:.4f}s, "
                      f"expected={expected:.3f}s, error={error_ms:.2f}ms")
            else:
                print(f"  Tap {i+1}: onset={ot:.4f}s, peak={pt:.4f}s")
    finally:
        os.remove(wav_path)
    
    print("✓ Example 2 complete\n")


def example_3_csv_export():
    """Example 3: CSV export."""
    print("\n" + "="*70)
    print("Example 3: CSV Export")
    print("="*70)
    
    # Create simple data
    onset_times = np.array([0.1, 0.5, 1.0, 1.5])
    peak_times = np.array([0.12, 0.52, 1.02, 1.52])
    
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        csv_path = f.name
    
    try:
        # Save to CSV
        onset_hilbert.save_onsets_and_peaks_csv(
            csv_path,
            onset_times,
            peak_times,
            label='example'
        )
        
        print(f"Saved results to: {csv_path}")
        
        # Read and display
        import pandas as pd
        df = pd.read_csv(csv_path)
        print("\nCSV contents:")
        print(df.to_string())
    finally:
        os.remove(csv_path)
    
    print("\n✓ Example 3 complete\n")


def example_4_comparing_hpf_cutoffs():
    """Example 4: Comparing different HPF cutoffs."""
    print("\n" + "="*70)
    print("Example 4: Comparing HPF Cutoffs")
    print("="*70)
    
    # Create test audio
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.zeros_like(t)
    
    # Add single click
    click_time = 0.5
    click_samples = int(0.02 * sr)
    click_start = int(click_time * sr)
    decay = np.exp(-np.linspace(0, 8, click_samples))
    audio[click_start:click_start + click_samples] = decay * 0.8
    audio += np.random.randn(len(audio)) * 0.005
    audio = audio / np.max(np.abs(audio)) * 0.9
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        wav_path = f.name
        wavfile.write(wav_path, sr, audio_int16)
    
    try:
        print("Testing different HPF cutoffs on the same audio:")
        
        for cutoff in [None, 100.0, 500.0, 1000.0, 2000.0]:
            onset_times, peak_times = onset_hilbert.detect_click_onsets_and_peaks(
                wav_path,
                hp_cutoff_hz=cutoff,
                min_distance_ms=50.0
            )
            
            cutoff_str = f"{cutoff:.0f} Hz" if cutoff else "no HPF"
            if len(onset_times) > 0:
                error_ms = abs(onset_times[0] - click_time) * 1000
                print(f"  HPF {cutoff_str:>10}: onset={onset_times[0]:.4f}s, "
                      f"error={error_ms:.2f}ms")
            else:
                print(f"  HPF {cutoff_str:>10}: no detection")
    finally:
        os.remove(wav_path)
    
    print("\n✓ Example 4 complete\n")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Unified Onset + Peak Detection Module - Examples")
    print("Using Fujii-style Hilbert envelope method")
    print("="*70)
    
    example_1_basic_click_detection()
    example_2_tap_detection()
    example_3_csv_export()
    example_4_comparing_hpf_cutoffs()
    
    print("\n" + "="*70)
    print("All examples complete!")
    print("="*70)
    print("\nFor more information, see onset_hilbert_README.md")


if __name__ == "__main__":
    main()
