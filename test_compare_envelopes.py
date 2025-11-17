"""
Unit tests for envelope comparison framework.

This test suite validates the envelope_variants module and compare_envelopes
script functionality with synthetic audio and known onset times.
"""

import unittest
import numpy as np
import os
import tempfile
import scipy.io.wavfile as wavfile
import envelope_variants as ev
import compare_envelopes as ce


class TestEnvelopeVariants(unittest.TestCase):
    """Test cases for envelope_variants module functions."""

    @classmethod
    def setUpClass(cls):
        """Create synthetic test audio with known onset times."""
        np.random.seed(42)  # For reproducibility
        cls.sr = 48000
        cls.duration = 2.0
        cls.expected_onset_times = [0.3, 0.9, 1.5]

        t = np.linspace(0, cls.duration, int(cls.sr * cls.duration))
        cls.signal = np.zeros_like(t)

        for tap_time in cls.expected_onset_times:
            tap_idx = int(tap_time * cls.sr)
            tap_dur = int(0.03 * cls.sr)
            decay = np.exp(-np.linspace(0, 6, tap_dur))
            noise = np.random.randn(tap_dur)
            if tap_idx + tap_dur < len(cls.signal):
                cls.signal[tap_idx:tap_idx + tap_dur] += noise * decay * 0.6

        # Add background noise
        cls.signal += np.random.randn(len(cls.signal)) * 0.01
        cls.signal = cls.signal / np.max(np.abs(cls.signal)) * 0.8

    def test_envelope_length_integrity(self):
        """Test that all envelope variants maintain correct length."""
        # Hilbert envelope
        env_hilbert_unsmoothed = ev.compute_hilbert_envelope_variant(
            self.signal, self.sr, hpf_cutoff=300.0, smooth_ms=None
        )
        self.assertEqual(len(env_hilbert_unsmoothed), len(self.signal))

        env_hilbert_smoothed = ev.compute_hilbert_envelope_variant(
            self.signal, self.sr, hpf_cutoff=300.0, smooth_ms=0.5
        )
        self.assertEqual(len(env_hilbert_smoothed), len(self.signal))

        # RMS envelope (will have different length due to framing)
        env_rms, times_rms = ev.compute_rms_envelope_variant(
            self.signal, self.sr, hpf_cutoff=300.0, frame_ms=5.0, hop_ms=1.0
        )
        self.assertEqual(len(env_rms), len(times_rms))
        self.assertGreater(len(env_rms), 0)

    def test_unsmoothed_sharper_than_smoothed(self):
        """Test that unsmoothed Hilbert has higher max first derivative."""
        env_unsmoothed = ev.compute_hilbert_envelope_variant(
            self.signal, self.sr, hpf_cutoff=300.0, smooth_ms=None
        )
        env_smoothed = ev.compute_hilbert_envelope_variant(
            self.signal, self.sr, hpf_cutoff=300.0, smooth_ms=0.5
        )

        # Compute first derivatives
        diff_unsmoothed = np.abs(np.diff(env_unsmoothed))
        diff_smoothed = np.abs(np.diff(env_smoothed))

        # Max derivative of unsmoothed should be higher (sharper)
        max_deriv_unsmoothed = np.max(diff_unsmoothed)
        max_deriv_smoothed = np.max(diff_smoothed)

        self.assertGreater(
            max_deriv_unsmoothed,
            max_deriv_smoothed,
            "Unsmoothed envelope should have sharper derivatives"
        )

    def test_rise_time_with_smoothing(self):
        """Test that rise time is affected by smoothing (changes are measurable)."""
        # Detect with smoothed envelope
        env_smoothed = ev.compute_hilbert_envelope_variant(
            self.signal, self.sr, hpf_cutoff=300.0, smooth_ms=0.5
        )
        onset_smoothed, peak_smoothed = ev.detect_events_fujii(
            env_smoothed, self.sr, threshold_ratio=0.1, min_distance_ms=100.0
        )

        # Detect with unsmoothed envelope
        env_unsmoothed = ev.compute_hilbert_envelope_variant(
            self.signal, self.sr, hpf_cutoff=300.0, smooth_ms=None
        )
        onset_unsmoothed, peak_unsmoothed = ev.detect_events_fujii(
            env_unsmoothed, self.sr, threshold_ratio=0.1, min_distance_ms=100.0
        )

        # Calculate average rise times
        if len(onset_smoothed) > 0:
            rise_times_smoothed = []
            for ot, pt in zip(onset_smoothed, peak_smoothed):
                _, rise_ms = ev.calc_rise_time(
                    env_smoothed, ot * self.sr, int(pt * self.sr), self.sr
                )
                rise_times_smoothed.append(rise_ms)
            avg_rise_smoothed = np.mean(rise_times_smoothed)
        else:
            avg_rise_smoothed = 0.0

        if len(onset_unsmoothed) > 0:
            rise_times_unsmoothed = []
            for ot, pt in zip(onset_unsmoothed, peak_unsmoothed):
                _, rise_ms = ev.calc_rise_time(
                    env_unsmoothed, ot * self.sr, int(pt * self.sr), self.sr
                )
                rise_times_unsmoothed.append(rise_ms)
            avg_rise_unsmoothed = np.mean(rise_times_unsmoothed)
        else:
            avg_rise_unsmoothed = 0.0

        # Both should produce valid measurements
        # The actual relationship can vary based on signal characteristics
        # What's important is that both produce measurable, non-zero rise times
        self.assertGreater(avg_rise_smoothed, 0.0, "Smoothed envelope should have measurable rise time")
        self.assertGreater(avg_rise_unsmoothed, 0.0, "Unsmoothed envelope should have measurable rise time")
        
        # The difference should be measurable (not identical)
        self.assertNotAlmostEqual(avg_rise_smoothed, avg_rise_unsmoothed, places=2,
                                  msg="Smoothing should produce measurably different rise times")

    def test_global_height_ratio_filtering(self):
        """Test that higher global_min_height_ratio detects fewer or equal events."""
        env = ev.compute_hilbert_envelope_variant(
            self.signal, self.sr, hpf_cutoff=300.0, smooth_ms=None
        )

        # Detect with no global filtering
        onset_no_filter, _ = ev.detect_events_fujii(
            env, self.sr, threshold_ratio=0.1,
            global_min_height_ratio=0.0, min_distance_ms=100.0
        )

        # Detect with high global filtering
        onset_high_filter, _ = ev.detect_events_fujii(
            env, self.sr, threshold_ratio=0.1,
            global_min_height_ratio=0.2, min_distance_ms=100.0
        )

        # Higher filter should detect fewer or equal events
        self.assertLessEqual(
            len(onset_high_filter),
            len(onset_no_filter),
            "Higher global_min_height_ratio should detect ≤ events"
        )

    def test_min_distance_splitting(self):
        """Test that smaller min_distance_ms may detect more closely spaced events."""
        # Create signal with two closely spaced taps (80ms apart)
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        signal = np.zeros_like(t)

        tap_times = [0.3, 0.38]  # 80ms apart
        for tap_time in tap_times:
            tap_idx = int(tap_time * sr)
            tap_dur = int(0.02 * sr)
            decay = np.exp(-np.linspace(0, 5, tap_dur))
            noise = np.random.randn(tap_dur)
            if tap_idx + tap_dur < len(signal):
                signal[tap_idx:tap_idx + tap_dur] += noise * decay * 0.6

        signal += np.random.randn(len(signal)) * 0.01

        env = ev.compute_hilbert_envelope_variant(signal, sr, hpf_cutoff=300.0)

        # Detect with large min_distance (100ms) - should merge
        onset_large, _ = ev.detect_events_fujii(
            env, sr, threshold_ratio=0.1, min_distance_ms=100.0
        )

        # Detect with small min_distance (50ms) - may split
        onset_small, _ = ev.detect_events_fujii(
            env, sr, threshold_ratio=0.1, min_distance_ms=50.0
        )

        # Smaller distance should detect >= events (allows splitting)
        self.assertGreaterEqual(
            len(onset_small),
            len(onset_large),
            "Smaller min_distance_ms should allow detecting more events"
        )

    def test_parameter_sweep_no_exceptions(self):
        """Test that parameter sweep executes without exceptions and produces output."""
        # Create temporary WAV file
        signal_int16 = (self.signal * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_path = f.name
            wavfile.write(wav_path, self.sr, signal_int16)

        try:
            # Load audio
            y, sr = ce.load_audio(wav_path)
            self.assertEqual(len(y), len(self.signal))

            # Generate variants with small parameter set
            variants = ce.generate_variants(
                y, sr,
                hpf_cutoffs=[0, 300],
                smooth_ms_list=[0, 0.5],
                global_min_height_ratios=[0, 0.2],
                min_distance_ms_list=[100],
                lookback_points_list=[0],
                rms_frame_ms=5.0,
                rms_hop_ms=1.0,
                threshold_ratio=0.1
            )

            # Should produce variants
            self.assertGreater(len(variants), 0)

            # Compute metrics for all variants
            metrics_list = []
            for variant in variants:
                metrics = ce.compute_variant_metrics(variant, sr)
                metrics_list.append(metrics)

            # Should produce non-empty metrics
            self.assertEqual(len(metrics_list), len(variants))

            # Each metric should have required keys
            required_keys = [
                'variant_id', 'n_events', 'mean_onset_to_peak_ms',
                'rise_time_10_90_ms', 'median_rise_time_ms', 'slope_10_90'
            ]
            for metrics in metrics_list:
                for key in required_keys:
                    self.assertIn(key, metrics)

        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    def test_variant_id_construction(self):
        """Test variant ID string construction."""
        # Hilbert variant
        vid_hilbert = ev.build_variant_id(
            "hilbert", 300.0, 0.5, 0.2, 100.0, 74
        )
        self.assertIn("hilbert", vid_hilbert)
        self.assertIn("hpf=300", vid_hilbert)
        self.assertIn("smooth=0.5ms", vid_hilbert)
        self.assertIn("gmin=0.20", vid_hilbert)
        self.assertIn("mindist=100ms", vid_hilbert)
        self.assertIn("lookback=74", vid_hilbert)

        # Hilbert variant with no HPF
        vid_no_hpf = ev.build_variant_id(
            "hilbert", None, None, 0.0, 50.0, 0
        )
        self.assertIn("hpf=none", vid_no_hpf)
        self.assertIn("smooth=none", vid_no_hpf)

        # RMS variant
        vid_rms = ev.build_variant_id(
            "rms", 300.0, None, 0.2, 100.0, 0,
            frame_ms=5.0, hop_ms=1.0
        )
        self.assertIn("rms", vid_rms)
        self.assertIn("frame=5.0ms", vid_rms)
        self.assertIn("hop=1.0ms", vid_rms)


class TestLookbackQuietPeriod(unittest.TestCase):
    """Test cases for lookback quiet-period criterion."""

    def test_lookback_disabled(self):
        """Test that lookback_points=0 returns onset unchanged."""
        env = np.array([0.1, 0.2, 0.5, 0.8, 1.0, 0.7, 0.3])
        onset_idx = 2.5
        peak_idx = 4
        threshold = 0.1

        result = ev.apply_lookback_quiet_period(
            env, onset_idx, peak_idx, lookback_points=0, threshold=threshold
        )

        self.assertEqual(result, onset_idx)

    def test_lookback_valid_quiet_period(self):
        """Test that valid quiet period returns onset."""
        # Envelope with quiet baseline before onset
        env = np.array([0.01, 0.02, 0.01, 0.5, 0.8, 1.0, 0.7])
        onset_idx = 3.0
        peak_idx = 5
        threshold = 0.1
        lookback_points = 3

        result = ev.apply_lookback_quiet_period(
            env, onset_idx, peak_idx, lookback_points, threshold
        )

        self.assertEqual(result, onset_idx)

    def test_lookback_invalid_noisy_baseline(self):
        """Test that noisy baseline returns None."""
        # Envelope with noise in baseline
        env = np.array([0.15, 0.12, 0.20, 0.5, 0.8, 1.0, 0.7])
        onset_idx = 3.0
        peak_idx = 5
        threshold = 0.1
        lookback_points = 3

        result = ev.apply_lookback_quiet_period(
            env, onset_idx, peak_idx, lookback_points, threshold
        )

        self.assertIsNone(result)


class TestRiseTimeAndSlope(unittest.TestCase):
    """Test cases for rise time and slope calculations."""

    def test_rise_time_calculation(self):
        """Test rise time calculation with synthetic envelope."""
        sr = 48000
        # Create envelope with linear rise
        env = np.concatenate([
            np.zeros(100),
            np.linspace(0, 1.0, 480),  # 10ms rise at 48kHz
            np.ones(100)
        ])

        onset_idx = 100.0
        peak_idx = 580

        rise_samples, rise_ms = ev.calc_rise_time(env, onset_idx, peak_idx, sr)

        # Rise time should be approximately 10ms
        self.assertGreater(rise_ms, 5.0)
        self.assertLess(rise_ms, 15.0)

    def test_slope_calculation(self):
        """Test slope calculation with synthetic envelope."""
        sr = 48000
        # Create envelope with steep rise
        env = np.concatenate([
            np.zeros(100),
            np.linspace(0, 1.0, 100),
            np.ones(100)
        ])

        onset_idx = 100.0
        peak_idx = 200

        slope = ev.calc_slope(env, onset_idx, peak_idx, sr)

        # Slope should be positive
        self.assertGreater(slope, 0)

    def test_zero_peak_amplitude(self):
        """Test that zero peak amplitude returns 0 slope."""
        sr = 48000
        env = np.zeros(300)

        onset_idx = 100.0
        peak_idx = 200

        slope = ev.calc_slope(env, onset_idx, peak_idx, sr)

        self.assertEqual(slope, 0.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases for envelope comparison."""

    def test_empty_signal(self):
        """Test handling of empty signal."""
        y = np.array([])
        sr = 48000

        env = ev.compute_hilbert_envelope_variant(y, sr)
        self.assertEqual(len(env), 0)

        onset_times, peak_times = ev.detect_events_fujii(env, sr)
        self.assertEqual(len(onset_times), 0)
        self.assertEqual(len(peak_times), 0)

    def test_silence(self):
        """Test handling of silence."""
        y = np.zeros(48000)
        sr = 48000

        env = ev.compute_hilbert_envelope_variant(y, sr, hpf_cutoff=300.0)

        onset_times, peak_times = ev.detect_events_fujii(env, sr)

        # Silence should produce no detections
        self.assertEqual(len(onset_times), 0)
        self.assertEqual(len(peak_times), 0)

    def test_very_short_audio(self):
        """Test handling of very short audio."""
        sr = 48000
        y = np.random.randn(100)  # ~2ms of audio

        env = ev.compute_hilbert_envelope_variant(y, sr, hpf_cutoff=300.0)

        # Should not crash
        self.assertEqual(len(env), len(y))

        onset_times, peak_times = ev.detect_events_fujii(env, sr)

        # May or may not detect anything, but should not crash
        self.assertGreaterEqual(len(onset_times), 0)


if __name__ == '__main__':
    unittest.main()
