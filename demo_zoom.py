#!/usr/bin/env python3
"""
Manual demo script to test X-axis zoom functionality.

This script creates a test plot and allows you to:
- Use mouse wheel to zoom in/out on the X-axis
- The zoom is centered on the mouse cursor position
- Both subplots should zoom together (sharex=True)

Instructions:
1. Run this script
2. Move your mouse over the plot
3. Scroll up to zoom in
4. Scroll down to zoom out
5. The zoom should only affect the X-axis (time)
"""

import numpy as np
import matplotlib.pyplot as plt
import onset_detection

def main():
    print("=" * 70)
    print("X-Axis Zoom Functionality Demo")
    print("=" * 70)
    print()
    print("Instructions:")
    print("  - Move your mouse over the plot")
    print("  - Scroll UP (or pinch OUT) to ZOOM IN on X-axis")
    print("  - Scroll DOWN (or pinch IN) to ZOOM OUT on X-axis")
    print("  - The zoom is centered on your mouse cursor position")
    print("  - Both plots zoom together (shared X-axis)")
    print()
    print("Creating test data...")
    
    # Create synthetic test data with clear features
    sr = 1000  # Sample rate
    duration = 5.0  # 5 seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a signal with some taps (sudden amplitude increases)
    y = np.zeros_like(t)
    tap_times = [0.5, 1.2, 2.0, 2.8, 3.5, 4.3]
    
    for tap_time in tap_times:
        # Create a damped sine wave for each tap
        tap_start_idx = int(tap_time * sr)
        tap_duration = 0.2
        tap_length = int(tap_duration * sr)
        
        if tap_start_idx + tap_length < len(y):
            tap_t = np.linspace(0, tap_duration, tap_length)
            tap_signal = np.exp(-10 * tap_t) * np.sin(2 * np.pi * 50 * tap_t)
            y[tap_start_idx:tap_start_idx + tap_length] += tap_signal
    
    # Add some noise
    y += np.random.randn(len(t)) * 0.05
    
    # Create envelope (RMS)
    env = np.abs(y)
    times = t
    
    # Onset times (known from our synthetic data)
    onset_times = np.array(tap_times)
    
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {duration} seconds")
    print(f"Number of onsets: {len(onset_times)}")
    print()
    print("Opening plot window...")
    print("Try scrolling with your mouse wheel over the plot!")
    print()
    
    # Create the plot with zoom functionality
    onset_detection.plot_envelope_with_onsets(
        y, sr, env, times, onset_times,
        title="Test Plot - Try Mouse Wheel to Zoom X-axis"
    )
    
    print()
    print("Plot closed.")
    print("=" * 70)

if __name__ == "__main__":
    main()
