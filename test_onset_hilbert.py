"""
Unit tests for onset_hilbert module.

This test suite validates the unified onset + peak detection module
using Fujii-style Hilbert envelope method.
"""

import unittest
import numpy as np
import os
import tempfile
import scipy.io.wavfile as wavfile
import onset_hilbert


class TestHighpassFilter(unittest.TestCase):
    """Test cases for highpass_filter function."""
    
    def test_no_filter_when_cutoff_none(self):
        """Test that no filtering is applied when cutoff is None."""
        sr = 48000
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration))
        y = np.sin(2 * np.pi * 100 * t)
        
        y_filt = onset_hilbert.highpass_filter(y, sr, None)
        
        # Should return unchanged signal
        np.testing.assert_array_equal(y_filt, y)
    
    def test_no_filter_when_cutoff_zero(self):
        """Test that no filtering is applied when cutoff is 0."""
        sr = 48000
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration))
        y = np.sin(2 * np.pi * 100 * t)
        
        y_filt = onset_hilbert.highpass_filter(y, sr, 0.0)
        
        # Should return unchanged signal
        np.testing.assert_array_equal(y_filt, y)
    
    def test_highpass_removes_low_frequencies(self):
        """Test that high-pass filter attenuates low frequencies."""
        sr = 48000
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create signal with low and high frequency components
        y_low = np.sin(2 * np.pi * 50 * t)
        y_high = np.sin(2 * np.pi * 2000 * t)
        y = y_low + y_high
        
        # Apply HPF at 500 Hz
        y_filt = onset_hilbert.highpass_filter(y, sr, 500.0)
        
        # Filtered signal should be non-zero
        self.assertGreater(np.max(np.abs(y_filt)), 0)
        
        # Filtered signal should be different from original
        self.assertFalse(np.allclose(y_filt, y))


class TestHilbertEnvelope(unittest.TestCase):
    """Test cases for hilbert_envelope function."""
    
    def test_envelope_same_length(self):
        """Test that envelope has same length as input signal."""
        sr = 48000
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration))
        y = np.sin(2 * np.pi * 1000 * t)
        
        env = onset_hilbert.hilbert_envelope(y, sr)
        
        self.assertEqual(len(env), len(y))
    
    def test_envelope_non_negative(self):
        """Test that envelope values are non-negative."""
        sr = 48000
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration))
        y = np.sin(2 * np.pi * 1000 * t)
        
        env = onset_hilbert.hilbert_envelope(y, sr)
        
        self.assertTrue(np.all(env >= 0))
    
    def test_envelope_with_smoothing(self):
        """Test envelope computation with smoothing."""
        sr = 48000
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration))
        y = np.sin(2 * np.pi * 1000 * t)
        
        env_no_smooth = onset_hilbert.hilbert_envelope(y, sr, smooth_ms=None)
        env_smooth = onset_hilbert.hilbert_envelope(y, sr, smooth_ms=1.0)
        
        # Both should be valid envelopes
        self.assertEqual(len(env_smooth), len(y))
        self.assertTrue(np.all(env_smooth >= 0))
        
        # Smoothed envelope should be different (usually smoother)
        self.assertFalse(np.allclose(env_no_smooth, env_smooth))


class TestDetectOnsetsAndPeaksFromEnvelope(unittest.TestCase):
    """Test cases for detect_onsets_and_peaks_from_envelope function."""
    
    def test_empty_envelope(self):
        """Test that empty envelope returns empty arrays."""
        env = np.array([])
        sr = 48000
        
        onset_times, peak_times = onset_hilbert.detect_onsets_and_peaks_from_envelope(
            env, sr
        )
        
        self.assertEqual(len(onset_times), 0)
        self.assertEqual(len(peak_times), 0)
    
    def test_zero_envelope(self):
        """Test that zero envelope returns empty arrays."""
        env = np.zeros(1000)
        sr = 48000
        
        onset_times, peak_times = onset_hilbert.detect_onsets_and_peaks_from_envelope(
            env, sr
        )
        
        self.assertEqual(len(onset_times), 0)
        self.assertEqual(len(peak_times), 0)
    
    def test_single_peak_detection(self):
        """Test detection of a single clear peak."""
        sr = 48000
        duration = 1.0
        n_samples = int(sr * duration)
        
        # Create envelope with single peak
        env = np.zeros(n_samples)
        peak_idx = n_samples // 2
        peak_width = 2000
        
        # Create a triangular peak (rising then falling)
        for i in range(peak_width):
            if peak_idx - peak_width + i >= 0:
                env[peak_idx - peak_width + i] = i / peak_width
            if peak_idx + i < n_samples:
                env[peak_idx + i] = 1.0 - i / peak_width
        
        onset_times, peak_times = onset_hilbert.detect_onsets_and_peaks_from_envelope(
            env, sr, min_distance_ms=10.0
        )
        
        # Should detect one peak
        self.assertEqual(len(onset_times), 1)
        self.assertEqual(len(peak_times), 1)
        
        # Peak should be near the middle
        self.assertAlmostEqual(peak_times[0], 0.5, delta=0.01)
        
        # Onset should be before peak
        self.assertLess(onset_times[0], peak_times[0])
    
    def test_multiple_peaks_detection(self):
        """Test detection of multiple peaks."""
        sr = 48000
        duration = 2.0
        n_samples = int(sr * duration)
        
        # Create envelope with 3 peaks
        env = np.zeros(n_samples)
        peak_positions = [0.4, 1.0, 1.6]  # in seconds
        
        for peak_time in peak_positions:
            peak_idx = int(peak_time * sr)
            peak_width = 1000
            
            for i in range(peak_width):
                if 0 <= peak_idx - i < n_samples:
                    env[peak_idx - i] = i / peak_width
                if 0 <= peak_idx + i < n_samples:
                    env[peak_idx + i] = i / peak_width
            if 0 <= peak_idx < n_samples:
                env[peak_idx] = 1.0
        
        onset_times, peak_times = onset_hilbert.detect_onsets_and_peaks_from_envelope(
            env, sr, min_distance_ms=50.0
        )
        
        # Should detect 3 peaks
        self.assertEqual(len(onset_times), 3)
        self.assertEqual(len(peak_times), 3)
        
        # Each onset should be before its corresponding peak
        for ot, pt in zip(onset_times, peak_times):
            self.assertLess(ot, pt)


class TestDetectionWrappers(unittest.TestCase):
    """Test cases for click and tap detection wrappers."""
    
    @classmethod
    def setUpClass(cls):
        """Create synthetic test audio files."""
        sr = 48000
        
        # Create click test file
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        click_audio = np.zeros_like(t)
        
        cls.expected_click_times = [0.3, 0.8, 1.3, 1.8]
        
        for click_time in cls.expected_click_times:
            click_duration = 0.02
            click_samples = int(click_duration * sr)
            click_start = int(click_time * sr)
            
            # Create sharp transient with exponential decay
            decay = np.exp(-np.linspace(0, 8, click_samples))
            click_sound = decay * 0.8
            
            if click_start + click_samples < len(click_audio):
                click_audio[click_start:click_start + click_samples] += click_sound
        
        # Add minimal noise
        click_audio += np.random.randn(len(click_audio)) * 0.005
        click_audio = click_audio / np.max(np.abs(click_audio)) * 0.9
        click_int16 = (click_audio * 32767).astype(np.int16)
        
        cls.temp_click_wav = tempfile.NamedTemporaryFile(suffix='_click.wav', delete=False)
        cls.temp_click_path = cls.temp_click_wav.name
        cls.temp_click_wav.close()
        wavfile.write(cls.temp_click_path, sr, click_int16)
        
        # Create tap test file
        tap_audio = np.zeros_like(t)
        cls.expected_tap_times = [0.4, 0.9, 1.4]
        
        for tap_time in cls.expected_tap_times:
            tap_duration = 0.03
            tap_samples = int(tap_duration * sr)
            tap_start = int(tap_time * sr)
            
            # Create tap sound with high-frequency content and decay
            decay = np.exp(-np.linspace(0, 6, tap_samples))
            noise = np.random.randn(tap_samples)
            tap_sound = noise * decay * 0.7
            
            if tap_start + tap_samples < len(tap_audio):
                tap_audio[tap_start:tap_start + tap_samples] += tap_sound
        
        # Add background noise
        tap_audio += np.random.randn(len(tap_audio)) * 0.01
        tap_audio = tap_audio / np.max(np.abs(tap_audio)) * 0.85
        tap_int16 = (tap_audio * 32767).astype(np.int16)
        
        cls.temp_tap_wav = tempfile.NamedTemporaryFile(suffix='_tap.wav', delete=False)
        cls.temp_tap_path = cls.temp_tap_wav.name
        cls.temp_tap_wav.close()
        wavfile.write(cls.temp_tap_path, sr, tap_int16)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        if os.path.exists(cls.temp_click_path):
            os.unlink(cls.temp_click_path)
        if os.path.exists(cls.temp_tap_path):
            os.unlink(cls.temp_tap_path)
    
    def test_detect_click_onsets_and_peaks(self):
        """Test click detection wrapper."""
        onset_times, peak_times = onset_hilbert.detect_click_onsets_and_peaks(
            self.temp_click_path,
            hp_cutoff_hz=1000.0,
            min_distance_ms=50.0
        )
        
        # Should detect approximately the expected number of clicks
        self.assertEqual(len(onset_times), len(self.expected_click_times))
        
        # Each detected onset should be close to an expected time
        for exp_time in self.expected_click_times:
            diffs = np.abs(onset_times - exp_time)
            min_diff = np.min(diffs)
            # Should be within 50ms
            self.assertLess(min_diff, 0.05, 
                          f"No click onset detected near {exp_time}s (closest: {min_diff}s)")
        
        # Onsets should be before peaks
        for ot, pt in zip(onset_times, peak_times):
            self.assertLess(ot, pt)
    
    def test_detect_tap_onsets_and_peaks(self):
        """Test tap detection wrapper."""
        onset_times, peak_times = onset_hilbert.detect_tap_onsets_and_peaks(
            self.temp_tap_path,
            hp_cutoff_hz=300.0,
            min_distance_ms=50.0
        )
        
        # Should detect approximately the expected number of taps
        self.assertEqual(len(onset_times), len(self.expected_tap_times))
        
        # Each detected onset should be close to an expected time
        for exp_time in self.expected_tap_times:
            diffs = np.abs(onset_times - exp_time)
            min_diff = np.min(diffs)
            # Should be within 50ms
            self.assertLess(min_diff, 0.05, 
                          f"No tap onset detected near {exp_time}s (closest: {min_diff}s)")
        
        # Onsets should be before peaks
        for ot, pt in zip(onset_times, peak_times):
            self.assertLess(ot, pt)


class TestCSVExport(unittest.TestCase):
    """Test cases for CSV export functionality."""
    
    def test_save_onsets_and_peaks_csv(self):
        """Test saving onset and peak times to CSV."""
        onset_times = np.array([0.1, 0.5, 1.0])
        peak_times = np.array([0.15, 0.55, 1.05])
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            csv_path = f.name
        
        try:
            # Save without label
            onset_hilbert.save_onsets_and_peaks_csv(
                csv_path, onset_times, peak_times
            )
            
            # Check file exists and can be read
            self.assertTrue(os.path.exists(csv_path))
            
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            # Check columns
            self.assertIn('index', df.columns)
            self.assertIn('onset_sec', df.columns)
            self.assertIn('peak_sec', df.columns)
            
            # Check data
            self.assertEqual(len(df), 3)
            np.testing.assert_array_almost_equal(df['onset_sec'].values, onset_times)
            np.testing.assert_array_almost_equal(df['peak_sec'].values, peak_times)
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_save_onsets_and_peaks_csv_with_label(self):
        """Test saving onset and peak times to CSV with label."""
        onset_times = np.array([0.1, 0.5])
        peak_times = np.array([0.15, 0.55])
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            csv_path = f.name
        
        try:
            # Save with label
            onset_hilbert.save_onsets_and_peaks_csv(
                csv_path, onset_times, peak_times, label="test"
            )
            
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            # Check label column exists
            self.assertIn('label', df.columns)
            
            # Check all labels are correct
            self.assertTrue(all(df['label'] == 'test'))
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_save_mismatched_lengths_raises_error(self):
        """Test that mismatched onset and peak arrays raise an error."""
        onset_times = np.array([0.1, 0.5])
        peak_times = np.array([0.15, 0.55, 1.05])
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            csv_path = f.name
        
        try:
            with self.assertRaises(ValueError):
                onset_hilbert.save_onsets_and_peaks_csv(
                    csv_path, onset_times, peak_times
                )
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)


class TestModuleStructure(unittest.TestCase):
    """Test cases for module structure and API."""
    
    def test_required_functions_exist(self):
        """Test that all required functions are exported."""
        required_functions = [
            'highpass_filter',
            'hilbert_envelope',
            'detect_onsets_and_peaks_from_envelope',
            'detect_click_onsets_and_peaks',
            'detect_tap_onsets_and_peaks',
            'save_onsets_and_peaks_csv',
            'plot_waveform_and_envelope',
            'interactive_hpf_tuning',
            'run_click_detection_with_dialog',
            'run_tap_detection_with_dialog',
        ]
        
        for func_name in required_functions:
            self.assertTrue(
                hasattr(onset_hilbert, func_name),
                f"Module missing function: {func_name}"
            )
    
    def test_functions_have_docstrings(self):
        """Test that functions have documentation."""
        functions = [
            onset_hilbert.highpass_filter,
            onset_hilbert.hilbert_envelope,
            onset_hilbert.detect_onsets_and_peaks_from_envelope,
            onset_hilbert.detect_click_onsets_and_peaks,
            onset_hilbert.detect_tap_onsets_and_peaks,
            onset_hilbert.save_onsets_and_peaks_csv,
        ]
        
        for func in functions:
            self.assertIsNotNone(func.__doc__)
            self.assertGreater(len(func.__doc__.strip()), 10)


if __name__ == '__main__':
    unittest.main()
