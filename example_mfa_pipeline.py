#!/usr/bin/env python
"""
Example usage of the MFA onset detection pipeline.

This script demonstrates how to use the pipeline to process speech recordings
with /t/ burst onsets, comparing MFA-based and Hilbert-based detection methods.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

# Generate example audio files with synthetic /t/ bursts
def create_example_audio():
    """Create example WAV files with synthetic /t/ burst patterns."""
    sr = 48000
    duration = 3.0
    
    # Create 3 example files
    for i in range(3):
        t = np.linspace(0, duration, int(sr * duration))
        y = np.zeros_like(t)
        
        # Add 4 /t/ bursts at different times
        burst_times = [0.5 + i*0.05, 1.2 + i*0.05, 1.9 + i*0.05, 2.6 + i*0.05]
        
        for burst_t in burst_times:
            idx = int(burst_t * sr)
            # Create a burst with high-frequency content
            burst_len = int(0.03 * sr)  # 30ms burst
            freq = 3000 + np.random.randint(-500, 500)  # Variable high frequency
            burst = np.sin(2 * np.pi * freq * np.linspace(0, 0.03, burst_len))
            # Add envelope
            envelope = np.hanning(burst_len)
            burst = burst * envelope * 0.3
            # Add to signal
            if idx + burst_len < len(y):
                y[idx:idx+burst_len] += burst
        
        # Add some low-level background noise
        y += np.random.randn(len(y)) * 0.01
        
        # Save
        filename = f"example_speech_{i+1}.wav"
        sf.write(filename, y, sr)
        print(f"Created {filename} with bursts at {burst_times}")
    
    print("\n" + "="*70)
    print("Example audio files created successfully!")
    print("="*70)
    print("\nNow you can process them with the pipeline:")
    print("\nBasic usage (Hilbert detection only):")
    print("  python mfa_onset_pipeline.py example_speech_*.wav")
    print("\nWith MFA alignment (requires MFA installed):")
    print("  python mfa_onset_pipeline.py example_speech_*.wav --run-mfa")
    print("\nWith custom parameters:")
    print("  python mfa_onset_pipeline.py example_speech_*.wav \\")
    print("      --hilbert-hp-cutoff 600 \\")
    print("      --hilbert-threshold 0.15 \\")
    print("      -o my_results/ \\")
    print("      -v")
    print("\n" + "="*70)


if __name__ == "__main__":
    create_example_audio()
