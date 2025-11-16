#!/usr/bin/env python3
"""
Integration test for X-axis zoom functionality with real onset detection.

This test creates a synthetic audio file, runs tap onset detection,
and verifies that the resulting plot has the scroll event handler attached.
"""

import os
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import onset_detection


def test_zoom_with_real_detection():
    """Test that zoom functionality works with actual onset detection workflow."""
    print("Integration Test: X-axis Zoom with Real Onset Detection")
    print("=" * 70)
    
    # Create temporary test audio file
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.zeros_like(t)
    
    # Add synthetic taps at known times
    tap_times = [0.5, 1.0, 1.5]
    print(f"Creating synthetic audio with {len(tap_times)} taps at times: {tap_times}")
    
    for tap_time in tap_times:
        tap_duration = 0.05
        tap_samples = int(tap_duration * sr)
        tap_start = int(tap_time * sr)
        
        decay = np.exp(-np.linspace(0, 5, tap_samples))
        noise = np.random.randn(tap_samples)
        tap_sound = noise * decay * 0.5
        
        if tap_start + tap_samples < len(audio):
            audio[tap_start:tap_start + tap_samples] += tap_sound
    
    # Add background noise
    audio += np.random.randn(len(audio)) * 0.01
    audio = audio / np.max(np.abs(audio)) * 0.8
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_wav_path = temp_file.name
    
    try:
        wavfile.write(temp_wav_path, sr, audio_int16)
        print(f"Created temporary test file: {temp_wav_path}")
        
        # Run tap onset detection
        print("Running tap onset detection...")
        detected_onsets = onset_detection.detect_tap_onsets_from_audio(
            temp_wav_path,
            hp_cutoff=500.0,
            diff_threshold_std=2.0,
            min_interval_ms=50.0
        )
        
        print(f"Detected {len(detected_onsets)} onsets: {detected_onsets}")
        
        # Load audio and compute envelope for plotting
        print("Computing envelope...")
        import librosa
        y, sr_loaded = librosa.load(temp_wav_path, sr=None, mono=True)
        env, times = onset_detection.compute_rms_envelope(
            y, sr_loaded,
            band=(500.0, None)
        )
        
        # Close any existing plots
        plt.close('all')
        
        # Mock plt.show() to prevent blocking
        original_show = plt.show
        plt.show = lambda: None
        
        try:
            # Create the plot with zoom functionality
            print("Creating plot with zoom functionality...")
            onset_detection.plot_envelope_with_onsets(
                y, sr_loaded, env, times, detected_onsets,
                title="Integration Test - X-axis Zoom"
            )
            
            # Verify the plot was created
            fig = plt.gcf()
            assert fig is not None, "No figure created"
            
            # Verify scroll event handler is connected
            callbacks = fig.canvas.callbacks.callbacks.get('scroll_event', {})
            assert len(callbacks) > 0, "No scroll event handler found"
            
            print("✓ Plot created successfully with scroll event handler")
            
            # Verify both subplots exist and share X-axis
            axes = fig.get_axes()
            assert len(axes) == 2, f"Expected 2 subplots, got {len(axes)}"
            print(f"✓ Found {len(axes)} subplots")
            
            # Test that the handler modifies X-axis
            ax = axes[0]
            initial_xlim = ax.get_xlim()
            print(f"  Initial X-axis limits: {initial_xlim}")
            
            # Get the scroll event handler directly
            # The callbacks are stored as StrongRef or WeakRef wrappers
            from matplotlib.backend_bases import MouseEvent
            event = MouseEvent('scroll_event', fig.canvas, x=100, y=100, button='up')
            event.inaxes = ax
            event.xdata = (initial_xlim[0] + initial_xlim[1]) / 2
            event.ydata = 0.5
            
            # Process the event through the canvas - this will trigger all callbacks
            fig.canvas.callbacks.process('scroll_event', event)
            
            new_xlim = ax.get_xlim()
            print(f"  After zoom in: {new_xlim}")
            
            # Verify X-axis width decreased (zoom in)
            initial_width = initial_xlim[1] - initial_xlim[0]
            new_width = new_xlim[1] - new_xlim[0]
            assert new_width < initial_width, f"X-axis did not zoom in: {new_width} >= {initial_width}"
            print(f"✓ X-axis zoom working correctly (width: {initial_width:.2f} → {new_width:.2f})")
            
        finally:
            plt.show = original_show
            plt.close('all')
        
        print()
        print("=" * 70)
        print("Integration test PASSED ✓")
        print("=" * 70)
        
    finally:
        # Clean up
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)
            print(f"Cleaned up temporary file: {temp_wav_path}")


if __name__ == "__main__":
    test_zoom_with_real_detection()
