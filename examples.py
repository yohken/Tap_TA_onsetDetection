"""
Example usage of the onset detection module with plotting.

This script demonstrates the three main use cases:
1. Tap onset detection from audio
2. /t/ burst onset detection with TextGrid
3. Click track generation

Run this script after creating test audio files, or modify the paths
to point to your own audio files.
"""

import onset_detection
import librosa
import numpy as np

def example_tap_detection():
    """Example: Detect tap onsets and plot results."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Tap Onset Detection")
    print("="*60)
    
    # Path to your tap audio file
    wav_path = "your_tap_recording.wav"  # Change this path
    
    try:
        # Detect tap onsets
        print(f"\nProcessing: {wav_path}")
        onset_times = onset_detection.detect_tap_onsets_from_audio(
            wav_path,
            hp_cutoff=500.0,        # High-pass filter at 500 Hz
            diff_threshold_std=2.0,  # Threshold sensitivity
            min_interval_ms=50.0     # Minimum 50ms between taps
        )
        
        # Display results
        print(f"Detected {len(onset_times)} tap onsets:")
        for i, t in enumerate(onset_times, 1):
            print(f"  {i}. {t:.3f} seconds")
        
        # Load audio and compute envelope for plotting
        print("\nGenerating plot...")
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        env, times = onset_detection.compute_rms_envelope(
            y, sr, band=(500.0, None)
        )
        
        # Plot the results
        onset_detection.plot_envelope_with_onsets(
            y, sr, env, times, onset_times,
            title="Tap Onset Detection"
        )
        
        print("✓ Plot displayed. Close the plot window to continue.")
        
    except FileNotFoundError:
        print(f"✗ File not found: {wav_path}")
        print("  Please create a tap recording or modify the path.")


def example_t_burst_detection():
    """Example: Detect /t/ burst onsets with MFA TextGrid."""
    print("\n" + "="*60)
    print("EXAMPLE 2: /t/ Burst Onset Detection")
    print("="*60)
    
    # Paths to your speech files
    wav_path = "your_speech.wav"        # Change these paths
    tg_path = "your_speech.TextGrid"
    
    try:
        # Detect /t/ burst onsets
        print(f"\nProcessing: {wav_path}")
        print(f"TextGrid: {tg_path}")
        onset_times = onset_detection.detect_t_burst_onsets_from_mfa(
            wav_path,
            tg_path,
            tier_name="phones",       # Tier containing phone segments
            phone_label="t",          # Look for 't' segments
            high_freq_min=2000.0,     # High frequency threshold
            diff_threshold_std=2.0
        )
        
        # Display results
        print(f"Detected {len(onset_times)} /t/ burst onsets:")
        for i, t in enumerate(onset_times, 1):
            print(f"  {i}. {t:.3f} seconds")
        
        # Load audio and compute envelope for plotting
        print("\nGenerating plot...")
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        env, times = onset_detection.compute_rms_envelope(
            y, sr, band=(2000.0, None)
        )
        
        # Plot the results
        onset_detection.plot_envelope_with_onsets(
            y, sr, env, times, onset_times,
            title="/t/ Burst Onset Detection"
        )
        
        print("✓ Plot displayed. Close the plot window to continue.")
        
    except FileNotFoundError:
        print(f"✗ Files not found")
        print("  Please ensure both WAV and TextGrid files exist.")


def example_click_track():
    """Example: Generate theoretical click track."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Click Track Generation")
    print("="*60)
    
    # Generate click track at 120 BPM
    bpm = 120
    n_clicks = 16
    subdivision = 1  # 1=quarter notes, 2=eighth notes, 4=sixteenth notes
    
    onset_times = onset_detection.get_click_onsets_from_bpm(
        bpm, 
        n_clicks,
        subdivision=subdivision
    )
    
    # Display results
    print(f"\nGenerated {len(onset_times)} clicks at {bpm} BPM:")
    print(f"Subdivision: {subdivision} (1=quarter, 2=eighth, 4=sixteenth)")
    for i, t in enumerate(onset_times, 1):
        print(f"  {i}. {t:.3f} seconds")
    
    # Calculate inter-onset intervals
    if len(onset_times) > 1:
        intervals = np.diff(onset_times)
        print(f"\nInter-onset interval: {intervals[0]:.3f} seconds")
        print(f"Frequency: {1.0/intervals[0]:.2f} Hz")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("ONSET DETECTION EXAMPLES")
    print("="*60)
    print("\nThis script demonstrates the onset detection functionality.")
    print("Modify the file paths in the code to use your own audio files.")
    
    # Example 3 works without any files
    example_click_track()
    
    # These require audio files (commented out by default)
    # Uncomment and modify paths to run:
    # example_tap_detection()
    # example_t_burst_detection()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETE")
    print("="*60)
    print("\nTo use the GUI application, run:")
    print("  python onset_detection_gui.py")
    print("\nOr:")
    print("  python demo_gui.py")
    print()


if __name__ == "__main__":
    main()
