"""
Unit tests for ta_onset_analysis module.

Tests the scientifically-grounded feature point detection methods:
- T_start (Fujii 10% method)
- T_peak (HF envelope maximum)
- A_start (periodicity-based voicing detection)
- A_peak (first stable periodic peak)
- A_end (segment end)
"""

import unittest
import numpy as np
import ta_onset_analysis


class TestTStartFujii(unittest.TestCase):
    """Test cases for detect_t_start_fujii function (Fujii 10% method)."""
    
    def setUp(self):
        """Create synthetic test signals."""
        self.sr = 44100
    
    def test_basic_detection(self):
        """Test basic T_start detection with a clear burst."""
        # Create a signal with a clear burst
        duration = 0.1
        t = np.linspace(0, duration, int(self.sr * duration))
        
        # Burst with exponential decay
        signal = np.exp(-50 * t) * 0.5
        
        t_start, peak_idx, env = ta_onset_analysis.detect_t_start_fujii(
            signal, self.sr
        )
        
        # T_start should be at or near the beginning
        self.assertGreaterEqual(t_start, 0.0)
        self.assertLess(t_start, 0.01)  # Should be within first 10ms
        
        # Envelope should be returned
        self.assertGreater(len(env), 0)
    
    def test_10_percent_threshold(self):
        """Test that 10% threshold is correctly applied."""
        # Create a simple step signal
        duration = 0.1
        samples = int(self.sr * duration)
        
        # Signal: quiet then loud
        signal = np.zeros(samples)
        transition = samples // 2
        signal[transition:] = 1.0
        
        t_start, peak_idx, env = ta_onset_analysis.detect_t_start_fujii(
            signal, self.sr,
            threshold_ratio=0.1
        )
        
        # T_start should be near the transition point
        transition_time = transition / self.sr
        self.assertGreater(t_start, 0)  # Not at very beginning (signal was quiet)
    
    def test_empty_signal(self):
        """Test handling of empty signal."""
        signal = np.array([])
        
        t_start, peak_idx, env = ta_onset_analysis.detect_t_start_fujii(
            signal, self.sr
        )
        
        self.assertEqual(t_start, 0.0)
        self.assertEqual(peak_idx, 0)
        self.assertEqual(len(env), 0)
    
    def test_linear_interpolation(self):
        """Test that linear interpolation produces valid results."""
        # Create a gradual rise from 0 to 1
        duration = 0.05
        samples = int(self.sr * duration)
        signal = np.linspace(0, 1, samples)
        
        t_start, peak_idx, env = ta_onset_analysis.detect_t_start_fujii(
            signal, self.sr,
            threshold_ratio=0.1
        )
        
        # T_start should be >= 0 (valid time)
        self.assertGreaterEqual(t_start, 0.0)
        # T_start should be less than peak time (which is at the end for rising signal)
        self.assertLess(t_start, duration)


class TestTPeak(unittest.TestCase):
    """Test cases for detect_t_peak function."""
    
    def setUp(self):
        """Create synthetic test signals."""
        self.sr = 44100
    
    def test_basic_peak_detection(self):
        """Test basic T_peak detection."""
        # Create a signal with a clear peak
        duration = 0.1
        t = np.linspace(0, duration, int(self.sr * duration))
        
        # Burst at 10ms
        peak_time = 0.01
        signal = np.exp(-((t - peak_time) ** 2) / (2 * 0.002 ** 2))
        
        t_peak, env = ta_onset_analysis.detect_t_peak(signal, self.sr)
        
        # T_peak should be near the expected peak
        self.assertAlmostEqual(t_peak, peak_time, delta=0.005)
    
    def test_peak_within_search_window(self):
        """Test that peak is found within the specified search window."""
        duration = 0.2
        t = np.linspace(0, duration, int(self.sr * duration))
        
        # Two peaks: one at 10ms (small) and one at 100ms (large)
        signal = 0.3 * np.exp(-((t - 0.01) ** 2) / (2 * 0.002 ** 2))
        signal += 1.0 * np.exp(-((t - 0.1) ** 2) / (2 * 0.002 ** 2))
        
        # With 50ms window, should find the first peak
        t_peak, env = ta_onset_analysis.detect_t_peak(
            signal, self.sr, search_window_ms=50.0
        )
        
        self.assertLess(t_peak, 0.05)  # Should be within first 50ms
    
    def test_empty_signal(self):
        """Test handling of empty signal."""
        signal = np.array([])
        
        t_peak, env = ta_onset_analysis.detect_t_peak(signal, self.sr)
        
        self.assertEqual(t_peak, 0.0)
        self.assertEqual(len(env), 0)


class TestAStartPeriodic(unittest.TestCase):
    """Test cases for detect_a_start_periodic function."""
    
    def setUp(self):
        """Create synthetic test signals with voiced regions."""
        self.sr = 44100
        
        # Create a signal with voiced region (periodic vowel)
        duration = 0.3
        t = np.linspace(0, duration, int(self.sr * duration))
        
        self.signal = np.zeros_like(t)
        
        # Noise burst (0-30ms)
        burst_end = int(0.03 * self.sr)
        self.signal[:burst_end] = np.random.randn(burst_end) * 0.3
        
        # Voiced vowel (50ms onwards) - periodic with F0 = 200 Hz
        vowel_start = int(0.05 * self.sr)
        f0 = 200
        for h in range(1, 4):
            self.signal[vowel_start:] += (0.7 / h) * np.sin(2 * np.pi * f0 * h * t[vowel_start:])
        
        self.signal = self.signal / np.max(np.abs(self.signal)) * 0.9
    
    def test_voicing_detection(self):
        """Test that voicing onset is detected."""
        a_start, f0_arr, voiced_flags = ta_onset_analysis.detect_a_start_periodic(
            self.signal, self.sr
        )
        
        # A_start should be detected near 50ms (where voicing starts)
        self.assertIsNotNone(a_start)
        self.assertGreater(a_start, 0.03)  # After the burst
        self.assertLess(a_start, 0.15)  # Before middle of signal
    
    def test_search_start_parameter(self):
        """Test that search_start_sec parameter is respected."""
        # Start searching from 40ms
        a_start, _, _ = ta_onset_analysis.detect_a_start_periodic(
            self.signal, self.sr,
            search_start_sec=0.04
        )
        
        # A_start should be >= 40ms
        self.assertGreaterEqual(a_start, 0.04)
    
    def test_pure_noise(self):
        """Test handling of pure noise (no voicing)."""
        noise = np.random.randn(int(0.2 * self.sr)) * 0.5
        
        a_start, f0_arr, voiced_flags = ta_onset_analysis.detect_a_start_periodic(
            noise, self.sr
        )
        
        # Should return some value (fallback mechanism)
        self.assertIsNotNone(a_start)
    
    def test_empty_signal(self):
        """Test handling of empty signal."""
        signal = np.array([])
        
        a_start, f0_arr, voiced_flags = ta_onset_analysis.detect_a_start_periodic(
            signal, self.sr
        )
        
        self.assertEqual(a_start, 0.0)


class TestAPeakPeriodic(unittest.TestCase):
    """Test cases for detect_a_peak_periodic function."""
    
    def setUp(self):
        """Create synthetic test signals."""
        self.sr = 44100
        
        # Create a periodic vowel signal
        duration = 0.2
        t = np.linspace(0, duration, int(self.sr * duration))
        
        f0 = 200
        self.signal = np.sin(2 * np.pi * f0 * t)
    
    def test_first_peak_detection(self):
        """Test detection of first periodic peak."""
        a_start_sec = 0.01  # Start from 10ms
        
        a_peak = ta_onset_analysis.detect_a_peak_periodic(
            self.signal, self.sr, a_start_sec
        )
        
        # A_peak should be after a_start
        self.assertGreater(a_peak, a_start_sec)
        
        # Should be within search window
        self.assertLess(a_peak, a_start_sec + 0.05)
    
    def test_peak_after_a_start(self):
        """Test that A_peak is always after A_start."""
        a_start_sec = 0.05
        
        a_peak = ta_onset_analysis.detect_a_peak_periodic(
            self.signal, self.sr, a_start_sec
        )
        
        self.assertGreater(a_peak, a_start_sec)
    
    def test_empty_signal(self):
        """Test handling of empty signal."""
        signal = np.array([])
        
        a_peak = ta_onset_analysis.detect_a_peak_periodic(
            signal, self.sr, 0.0
        )
        
        self.assertEqual(a_peak, 0.0)


class TestExtractTaFeaturePoints(unittest.TestCase):
    """Test cases for extract_ta_feature_points function."""
    
    def setUp(self):
        """Create synthetic 'ta' syllable signal."""
        self.sr = 44100
        duration = 0.3
        t = np.linspace(0, duration, int(self.sr * duration))
        
        self.signal = np.zeros_like(t)
        
        # T burst (0-30ms): noise burst with decay
        t_end = int(0.03 * self.sr)
        self.signal[:t_end] = np.random.randn(t_end) * np.exp(-np.linspace(0, 3, t_end)) * 0.5
        
        # A vowel (50ms onwards): periodic with harmonics
        a_start = int(0.05 * self.sr)
        f0 = 200
        for h in range(1, 4):
            self.signal[a_start:] += (0.7 / h) * np.sin(2 * np.pi * f0 * h * t[a_start:])
        
        self.signal = self.signal / np.max(np.abs(self.signal)) * 0.9
    
    def test_all_feature_points_returned(self):
        """Test that all 5 feature points are returned."""
        features = ta_onset_analysis.extract_ta_feature_points(
            self.signal, self.sr, 0, len(self.signal)
        )
        
        required_keys = ['t_start', 't_peak', 'a_start', 'a_peak', 'a_end']
        for key in required_keys:
            self.assertIn(key, features, f"Missing feature point: {key}")
    
    def test_feature_values_are_floats(self):
        """Test that all feature values are floats."""
        features = ta_onset_analysis.extract_ta_feature_points(
            self.signal, self.sr, 0, len(self.signal)
        )
        
        for key, value in features.items():
            self.assertIsInstance(value, float, f"{key} should be a float")
    
    def test_temporal_ordering(self):
        """Test that feature points are in correct temporal order."""
        features = ta_onset_analysis.extract_ta_feature_points(
            self.signal, self.sr, 0, len(self.signal)
        )
        
        self.assertLessEqual(features['t_start'], features['t_peak'],
                           "t_start should be <= t_peak")
        self.assertLessEqual(features['t_start'], features['a_start'],
                           "t_start should be <= a_start")
        self.assertLessEqual(features['a_start'], features['a_peak'],
                           "a_start should be <= a_peak")
        self.assertLessEqual(features['a_peak'], features['a_end'],
                           "a_peak should be <= a_end")
    
    def test_t_start_in_first_portion(self):
        """Test that t_start is in the first portion of the segment."""
        features = ta_onset_analysis.extract_ta_feature_points(
            self.signal, self.sr, 0, len(self.signal)
        )
        
        segment_duration = len(self.signal) / self.sr
        self.assertLess(features['t_start'], segment_duration * 0.5,
                       "t_start should be in first half of segment")
    
    def test_a_end_matches_segment_end(self):
        """Test that a_end matches the segment end."""
        segment_end = len(self.signal)
        expected_a_end = segment_end / self.sr
        
        features = ta_onset_analysis.extract_ta_feature_points(
            self.signal, self.sr, 0, segment_end
        )
        
        self.assertAlmostEqual(features['a_end'], expected_a_end, delta=0.001)
    
    def test_empty_segment(self):
        """Test handling of empty segment."""
        features = ta_onset_analysis.extract_ta_feature_points(
            self.signal, self.sr, 0, 0
        )
        
        # Should return valid dict with all keys
        required_keys = ['t_start', 't_peak', 'a_start', 'a_peak', 'a_end']
        for key in required_keys:
            self.assertIn(key, features)
    
    def test_segment_with_offset(self):
        """Test detection with non-zero segment_start."""
        segment_start = int(0.01 * self.sr)  # Start at 10ms
        segment_end = len(self.signal)
        
        features = ta_onset_analysis.extract_ta_feature_points(
            self.signal, self.sr, segment_start, segment_end
        )
        
        # All feature times should be >= segment_start / sr
        segment_start_sec = segment_start / self.sr
        for key, value in features.items():
            self.assertGreaterEqual(value, segment_start_sec,
                                  f"{key} should be >= segment start time")


class TestHelperFunctions(unittest.TestCase):
    """Test cases for helper functions."""
    
    def setUp(self):
        """Create test signals."""
        self.sr = 44100
    
    def test_bandpass_filter(self):
        """Test bandpass filter."""
        # Create signal with multiple frequency components
        t = np.linspace(0, 0.1, int(0.1 * self.sr))
        signal = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 1000 * t)
        
        # Filter to keep 50-200 Hz
        filtered = ta_onset_analysis._bandpass_filter(signal, self.sr, 50, 200)
        
        # Output should exist and be same length
        self.assertEqual(len(filtered), len(signal))
    
    def test_highpass_filter(self):
        """Test highpass filter."""
        t = np.linspace(0, 0.1, int(0.1 * self.sr))
        signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 1000 * t)
        
        filtered = ta_onset_analysis._highpass_filter(signal, self.sr, 500)
        
        self.assertEqual(len(filtered), len(signal))
    
    def test_lowpass_filter(self):
        """Test lowpass filter."""
        t = np.linspace(0, 0.1, int(0.1 * self.sr))
        signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 1000 * t)
        
        filtered = ta_onset_analysis._lowpass_filter(signal, self.sr, 200)
        
        self.assertEqual(len(filtered), len(signal))
    
    def test_hilbert_envelope(self):
        """Test Hilbert envelope computation."""
        t = np.linspace(0, 0.1, int(0.1 * self.sr))
        signal = np.sin(2 * np.pi * 100 * t)
        
        env = ta_onset_analysis._compute_hilbert_envelope(signal, sr=self.sr)
        
        self.assertEqual(len(env), len(signal))
        # Envelope should be non-negative
        self.assertTrue(np.all(env >= 0))
    
    def test_adaptive_threshold_median_mad(self):
        """Test adaptive threshold computation using median/MAD."""
        data = np.array([1, 2, 3, 4, 5, 100])  # With outlier
        
        threshold = ta_onset_analysis._compute_adaptive_threshold(
            data, method='median_mad'
        )
        
        # Threshold should be robust to outlier
        self.assertGreater(threshold, 0)
        self.assertLess(threshold, 100)  # Should not be dominated by outlier
    
    def test_adaptive_threshold_percentile(self):
        """Test adaptive threshold computation using percentile."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        threshold = ta_onset_analysis._compute_adaptive_threshold(
            data, method='percentile', percentile=75
        )
        
        self.assertAlmostEqual(threshold, 7.75, delta=0.1)


if __name__ == '__main__':
    unittest.main()
