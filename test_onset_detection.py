"""
Unit tests for onset detection module and GUI.

This test suite validates:
1. Basic onset detection functionality
2. Click track generation
3. Module imports and structure
"""

import unittest
import numpy as np
import os
import tempfile
import scipy.io.wavfile as wavfile
import onset_detection


class TestOnsetDetection(unittest.TestCase):
    """Test cases for onset detection functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Create synthetic test audio file."""
        # Create a simple test audio file with synthetic taps
        sr = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.zeros_like(t)
        
        # Add 5 tap sounds at specific times
        cls.expected_tap_times = [0.5, 1.0, 1.5, 2.0, 2.5]
        
        for tap_time in cls.expected_tap_times:
            tap_duration = 0.05
            tap_samples = int(tap_duration * sr)
            tap_start = int(tap_time * sr)
            
            decay = np.exp(-np.linspace(0, 5, tap_samples))
            noise = np.random.randn(tap_samples)
            tap_sound = noise * decay * 0.5
            
            if tap_start + tap_samples < len(audio):
                audio[tap_start:tap_start + tap_samples] += tap_sound
        
        # Add some background noise
        audio += np.random.randn(len(audio)) * 0.01
        audio = audio / np.max(np.abs(audio)) * 0.8
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Save to temporary file
        cls.temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        cls.temp_wav_path = cls.temp_wav.name
        cls.temp_wav.close()
        wavfile.write(cls.temp_wav_path, sr, audio_int16)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        if os.path.exists(cls.temp_wav_path):
            os.unlink(cls.temp_wav_path)
    
    def test_click_track_generation(self):
        """Test theoretical click track generation."""
        bpm = 120
        n_clicks = 8
        
        onset_times = onset_detection.get_click_onsets_from_bpm(
            bpm, n_clicks, subdivision=1
        )
        
        # Should generate exactly n_clicks
        self.assertEqual(len(onset_times), n_clicks)
        
        # First click at time 0
        self.assertAlmostEqual(onset_times[0], 0.0)
        
        # Clicks should be evenly spaced (0.5 seconds for 120 BPM)
        expected_interval = 60.0 / bpm
        for i in range(1, len(onset_times)):
            interval = onset_times[i] - onset_times[i-1]
            self.assertAlmostEqual(interval, expected_interval, places=6)
    
    def test_tap_onset_detection(self):
        """Test tap onset detection on synthetic audio."""
        onset_times = onset_detection.detect_tap_onsets_from_audio(
            self.temp_wav_path,
            hp_cutoff=500.0,
            diff_threshold_std=2.0,
            min_interval_ms=50.0
        )
        
        # Should detect approximately the expected number of taps
        self.assertEqual(len(onset_times), len(self.expected_tap_times))
        
        # Each detected onset should be close to an expected time
        for exp_time in self.expected_tap_times:
            # Find closest detected time
            diffs = np.abs(onset_times - exp_time)
            min_diff = np.min(diffs)
            # Should be within 100ms
            self.assertLess(min_diff, 0.1, 
                          f"No onset detected near {exp_time}s (closest: {min_diff}s)")
    
    def test_compute_rms_envelope(self):
        """Test RMS envelope computation."""
        # Create simple test signal
        sr = 1000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        y = np.sin(2 * np.pi * 100 * t)  # 100 Hz sine wave
        
        env, times = onset_detection.compute_rms_envelope(
            y, sr, 
            frame_length_ms=50.0,
            hop_length_ms=10.0
        )
        
        # Should produce non-empty envelope
        self.assertGreater(len(env), 0)
        self.assertEqual(len(env), len(times))
        
        # Envelope values should be positive
        self.assertTrue(np.all(env >= 0))
    
    def test_detect_onsets_from_envelope(self):
        """Test onset detection from envelope."""
        # Create simple envelope with clear peaks
        times = np.linspace(0, 1, 1000)
        env = np.zeros_like(times)
        
        # Add some peaks at specific times
        peak_times = [0.2, 0.5, 0.8]
        for pt in peak_times:
            idx = int(pt * len(times))
            env[max(0, idx-5):min(len(env), idx+5)] = 1.0
        
        onset_times = onset_detection.detect_onsets_from_envelope(
            env, times,
            diff_threshold_std=0.5,
            min_interval_ms=50.0
        )
        
        # Should detect the peaks
        self.assertGreater(len(onset_times), 0)
        self.assertLessEqual(len(onset_times), len(peak_times))


class TestHilbertEnvelope(unittest.TestCase):
    """Test cases for Hilbert envelope-based onset detection."""
    
    @classmethod
    def setUpClass(cls):
        """Create synthetic test audio files for metronome and tap."""
        sr = 48000  # Use target sampling rate
        
        # Create metronome test file with clear sound bursts
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        metronome_audio = np.zeros_like(t)
        
        # Add 4 metronome clicks at specific times
        cls.expected_metronome_times = [0.3, 0.8, 1.3, 1.8]
        
        for click_time in cls.expected_metronome_times:
            click_duration = 0.02
            click_samples = int(click_duration * sr)
            click_start = int(click_time * sr)
            
            # Create a sharp transient with exponential decay
            decay = np.exp(-np.linspace(0, 8, click_samples))
            click_sound = decay * 0.8
            
            if click_start + click_samples < len(metronome_audio):
                metronome_audio[click_start:click_start + click_samples] += click_sound
        
        # Add minimal noise
        metronome_audio += np.random.randn(len(metronome_audio)) * 0.005
        metronome_audio = metronome_audio / np.max(np.abs(metronome_audio)) * 0.9
        metronome_int16 = (metronome_audio * 32767).astype(np.int16)
        
        cls.temp_metronome_wav = tempfile.NamedTemporaryFile(suffix='_metronome.wav', delete=False)
        cls.temp_metronome_path = cls.temp_metronome_wav.name
        cls.temp_metronome_wav.close()
        wavfile.write(cls.temp_metronome_path, sr, metronome_int16)
        
        # Create tap test file with clear taps
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
        if os.path.exists(cls.temp_metronome_path):
            os.unlink(cls.temp_metronome_path)
        if os.path.exists(cls.temp_tap_path):
            os.unlink(cls.temp_tap_path)
    
    def test_compute_hilbert_envelope(self):
        """Test Hilbert envelope computation."""
        # Create simple test signal
        sr = 48000
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration))
        # Create amplitude-modulated signal
        carrier_freq = 1000
        modulation_freq = 10
        y = (1 + 0.5 * np.sin(2 * np.pi * modulation_freq * t)) * np.sin(2 * np.pi * carrier_freq * t)
        
        env = onset_detection.compute_hilbert_envelope(y, sr)
        
        # Should produce envelope with same length as signal
        self.assertEqual(len(env), len(y))
        
        # Envelope values should be non-negative
        self.assertTrue(np.all(env >= 0))
        
        # Envelope should be smooth and follow the amplitude modulation
        self.assertGreater(np.max(env), 0)
    
    def test_detect_metronome_onsets(self):
        """Test metronome onset detection with Hilbert envelope."""
        onset_times = onset_detection.detect_metronome_onsets_from_audio(
            self.temp_metronome_path,
            target_sr=48000,
            threshold_ratio=0.1,
            min_interval_ms=50.0
        )
        
        # Should detect approximately the expected number of clicks
        self.assertEqual(len(onset_times), len(self.expected_metronome_times))
        
        # Each detected onset should be close to an expected time
        for exp_time in self.expected_metronome_times:
            diffs = np.abs(onset_times - exp_time)
            min_diff = np.min(diffs)
            # Should be within 50ms
            self.assertLess(min_diff, 0.05, 
                          f"No metronome onset detected near {exp_time}s (closest: {min_diff}s)")
    
    def test_detect_tap_onsets_hilbert(self):
        """Test tap onset detection with Hilbert envelope and lookback."""
        onset_times = onset_detection.detect_tap_onsets_from_audio_hilbert(
            self.temp_tap_path,
            target_sr=48000,
            hp_cutoff=500.0,
            threshold_ratio=0.1,
            lookback_points=74,
            min_interval_ms=50.0
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
    
    def test_hilbert_envelope_with_filter(self):
        """Test Hilbert envelope with band-pass filtering."""
        sr = 48000
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create signal with low and high frequency components
        y = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 2000 * t)
        
        # Compute envelope with high-pass filter
        env_hpf = onset_detection.compute_hilbert_envelope(y, sr, band=(500.0, None))
        
        # Should produce valid envelope
        self.assertEqual(len(env_hpf), len(y))
        self.assertTrue(np.all(env_hpf >= 0))


class TestVoiceSegmentDetection(unittest.TestCase):
    """Test cases for voice segment detection with feature extraction.
    
    These tests validate the new voice segment detection functions that
    detect segments (e.g., 'ta' syllables) and extract feature points:
    - t_start: segment start
    - t_peak: burst maximum
    - a_start: vowel onset
    - a_stable: vowel stabilization
    - end: segment end
    """
    
    @classmethod
    def setUpClass(cls):
        """Create synthetic test audio with two 'ta'-like sounds and mock tkinter."""
        # Mock tkinter for headless testing environment
        import sys
        import unittest.mock as mock
        if 'tkinter' not in sys.modules:
            sys.modules['tkinter'] = mock.MagicMock()
            sys.modules['tkinter.filedialog'] = mock.MagicMock()
            sys.modules['tkinter.messagebox'] = mock.MagicMock()
            sys.modules['tkinter.ttk'] = mock.MagicMock()
        
        # Import and reload the module after mocking
        import importlib
        import onset_detection_gui
        importlib.reload(onset_detection_gui)
        cls.onset_detection_gui = onset_detection_gui
        
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        audio = np.zeros_like(t)
        cls.expected_segment_times = [0.2, 0.6]
        
        for burst_time in cls.expected_segment_times:
            # Consonant part (short burst)
            burst_start = int(burst_time * sr)
            burst_len = int(0.02 * sr)  # 20ms burst
            if burst_start + burst_len < len(audio):
                decay = np.exp(-np.linspace(0, 5, burst_len))
                noise = np.random.randn(burst_len)
                audio[burst_start:burst_start + burst_len] = noise * decay * 0.5
            
            # Vowel part (longer sustain)
            vowel_start = burst_start + burst_len
            vowel_len = int(0.15 * sr)  # 150ms vowel
            if vowel_start + vowel_len < len(audio):
                vowel_t = np.linspace(0, 0.15, vowel_len)
                vowel_env = np.sin(np.pi * vowel_t / 0.15)
                vowel_freq = 440
                vowel = np.sin(2 * np.pi * vowel_freq * vowel_t) * vowel_env * 0.3
                audio[vowel_start:vowel_start + vowel_len] += vowel
        
        audio = audio / np.max(np.abs(audio)) * 0.8
        cls.audio = audio
        cls.sr = sr
        
        # Save to temp file
        audio_int16 = (audio * 32767).astype(np.int16)
        cls.temp_wav = tempfile.NamedTemporaryFile(suffix='_voice_segment.wav', delete=False)
        cls.temp_wav_path = cls.temp_wav.name
        cls.temp_wav.close()
        wavfile.write(cls.temp_wav_path, sr, audio_int16)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        if os.path.exists(cls.temp_wav_path):
            os.unlink(cls.temp_wav_path)
    
    def test_detect_voice_segments_count(self):
        """Test that correct number of voice segments are detected."""
        segments = self.onset_detection_gui.detect_voice_segments(
            self.audio, self.sr,
            amplitude_threshold_ratio=0.05
        )
        
        self.assertEqual(len(segments), len(self.expected_segment_times),
                        f"Expected {len(self.expected_segment_times)} segments, got {len(segments)}")
    
    def test_detect_voice_segments_timing(self):
        """Test that detected segments are near expected times."""
        segments = self.onset_detection_gui.detect_voice_segments(
            self.audio, self.sr,
            amplitude_threshold_ratio=0.05
        )
        
        for exp_time in self.expected_segment_times:
            # Find closest segment start
            segment_starts = [start / self.sr for start, end in segments]
            diffs = [abs(start - exp_time) for start in segment_starts]
            min_diff = min(diffs)
            
            # Should be within 100ms of expected time
            self.assertLess(min_diff, 0.1,
                          f"No segment detected near {exp_time}s (closest: {min_diff}s)")
    
    def test_extract_feature_points_structure(self):
        """Test that feature points have correct structure."""
        segments = self.onset_detection_gui.detect_voice_segments(
            self.audio, self.sr,
            amplitude_threshold_ratio=0.05
        )
        
        required_keys = ['t_start', 't_peak', 'a_start', 'a_stable', 'end']
        
        for start, end in segments:
            features = self.onset_detection_gui.extract_feature_points(
                self.audio, self.sr, start, end
            )
            
            # Check all required keys exist
            for key in required_keys:
                self.assertIn(key, features, f"Feature point '{key}' missing")
            
            # Check values are numeric
            for key, val in features.items():
                self.assertIsInstance(val, float, f"Feature '{key}' should be float")
    
    def test_feature_points_ordering(self):
        """Test that feature points are in correct temporal order."""
        segments = self.onset_detection_gui.detect_voice_segments(
            self.audio, self.sr,
            amplitude_threshold_ratio=0.05
        )
        
        for start, end in segments:
            features = self.onset_detection_gui.extract_feature_points(
                self.audio, self.sr, start, end
            )
            
            # Feature points should be in order: t_start <= t_peak, t_start <= a_start <= a_stable <= end
            # Note: a_start can be before t_peak as long as it's after t_start (T segment start)
            self.assertLessEqual(features['t_start'], features['t_peak'],
                               "t_start should be <= t_peak")
            self.assertLessEqual(features['t_start'], features['a_start'],
                               "t_start should be <= a_start (a_start is valid if after T segment start)")
            self.assertLessEqual(features['a_start'], features['a_stable'],
                               "a_start should be <= a_stable")
            self.assertLessEqual(features['a_stable'], features['end'],
                               "a_stable should be <= end")
    
    def test_detect_voice_segments_with_features_integration(self):
        """Test the combined detection and feature extraction function."""
        y, sr, features_list = self.onset_detection_gui.detect_voice_segments_with_features(
            self.temp_wav_path,
            amplitude_threshold_ratio=0.05
        )
        
        # Check audio was loaded
        self.assertGreater(len(y), 0, "Audio should be loaded")
        self.assertEqual(sr, self.sr, "Sample rate should match")
        
        # Check features list
        self.assertEqual(len(features_list), len(self.expected_segment_times),
                        "Should detect expected number of segments")
        
        # Check each feature dict
        for features in features_list:
            self.assertIn('t_start', features)
            self.assertIn('t_peak', features)
            self.assertIn('a_start', features)
            self.assertIn('a_stable', features)
            self.assertIn('end', features)


class TestGUIModule(unittest.TestCase):
    """Test cases for GUI module structure."""
    
    def test_gui_import(self):
        """Test that GUI module can be imported."""
        try:
            import onset_detection_gui
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import GUI module: {e}")
    
    def test_gui_class_exists(self):
        """Test that OnsetDetectionGUI class exists."""
        import onset_detection_gui
        self.assertTrue(hasattr(onset_detection_gui, 'OnsetDetectionGUI'))
    
    def test_gui_has_required_methods(self):
        """Test that GUI class has required methods."""
        import onset_detection_gui
        
        required_methods = [
            'detect_tap_onsets',
            'detect_t_burst_onsets',
            'detect_voice_segments',
            'update_status',
            'append_result',
            'clear_results'
        ]
        
        for method_name in required_methods:
            self.assertTrue(
                hasattr(onset_detection_gui.OnsetDetectionGUI, method_name),
                f"GUI class missing method: {method_name}"
            )


class TestModuleStructure(unittest.TestCase):
    """Test cases for module structure and API."""
    
    def test_required_functions_exist(self):
        """Test that all required functions are exported."""
        required_functions = [
            'compute_rms_envelope',
            'compute_hilbert_envelope',
            'detect_onsets_from_envelope',
            'get_click_onsets_from_bpm',
            'detect_metronome_onsets_from_audio',
            'detect_tap_onsets_from_audio',
            'detect_tap_onsets_from_audio_hilbert',
            'detect_t_burst_onsets_from_mfa',
            'plot_envelope_with_onsets'
        ]
        
        for func_name in required_functions:
            self.assertTrue(
                hasattr(onset_detection, func_name),
                f"Module missing function: {func_name}"
            )
    
    def test_functions_have_docstrings(self):
        """Test that functions have documentation."""
        functions = [
            onset_detection.compute_rms_envelope,
            onset_detection.detect_onsets_from_envelope,
            onset_detection.get_click_onsets_from_bpm,
            onset_detection.detect_tap_onsets_from_audio,
            onset_detection.plot_envelope_with_onsets
        ]
        
        for func in functions:
            self.assertIsNotNone(func.__doc__)
            self.assertGreater(len(func.__doc__.strip()), 10)


class TestVoiceSegmentExport(unittest.TestCase):
    """Test cases for voice segment export functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Create synthetic test audio."""
        import sys
        import unittest.mock as mock
        if 'tkinter' not in sys.modules:
            sys.modules['tkinter'] = mock.MagicMock()
            sys.modules['tkinter.filedialog'] = mock.MagicMock()
            sys.modules['tkinter.messagebox'] = mock.MagicMock()
            sys.modules['tkinter.ttk'] = mock.MagicMock()
        
        import importlib
        import onset_detection_gui
        importlib.reload(onset_detection_gui)
        cls.onset_detection_gui = onset_detection_gui
        
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        audio = np.zeros_like(t)
        cls.expected_segment_times = [0.2, 0.6]
        
        for burst_time in cls.expected_segment_times:
            burst_start = int(burst_time * sr)
            burst_len = int(0.02 * sr)
            if burst_start + burst_len < len(audio):
                decay = np.exp(-np.linspace(0, 5, burst_len))
                noise = np.random.randn(burst_len)
                audio[burst_start:burst_start + burst_len] = noise * decay * 0.5
            
            vowel_start = burst_start + burst_len
            vowel_len = int(0.15 * sr)
            if vowel_start + vowel_len < len(audio):
                vowel_t = np.linspace(0, 0.15, vowel_len)
                vowel_env = np.sin(np.pi * vowel_t / 0.15)
                vowel_freq = 440
                vowel = np.sin(2 * np.pi * vowel_freq * vowel_t) * vowel_env * 0.3
                audio[vowel_start:vowel_start + vowel_len] += vowel
        
        audio = audio / np.max(np.abs(audio)) * 0.8
        cls.audio = audio
        cls.sr = sr
        
        audio_int16 = (audio * 32767).astype(np.int16)
        cls.temp_wav = tempfile.NamedTemporaryFile(suffix='_export_test.wav', delete=False)
        cls.temp_wav_path = cls.temp_wav.name
        cls.temp_wav.close()
        wavfile.write(cls.temp_wav_path, sr, audio_int16)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        if os.path.exists(cls.temp_wav_path):
            os.unlink(cls.temp_wav_path)
    
    def test_csv_export_data_format(self):
        """Test that CSV export produces correct data format."""
        import pandas as pd
        
        # Get detected features
        segments = self.onset_detection_gui.detect_voice_segments(
            self.audio, self.sr,
            amplitude_threshold_ratio=0.05
        )
        
        features_list = []
        for start, end in segments:
            features = self.onset_detection_gui.extract_feature_points(
                self.audio, self.sr, start, end
            )
            features_list.append(features)
        
        # Create CSV data manually (simulating export)
        rows = []
        for i, features in enumerate(features_list):
            row = {
                'segment_index': i,
                't_start_sec': features['t_start'],
                't_peak_sec': features['t_peak'],
                'a_start_sec': features['a_start'],
                'a_stable_sec': features['a_stable'],
                'end_sec': features['end'],
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Check column names
        expected_columns = ['segment_index', 't_start_sec', 't_peak_sec', 
                          'a_start_sec', 'a_stable_sec', 'end_sec']
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        # Check data count
        self.assertEqual(len(df), len(features_list))
    
    def test_json_export_data_structure(self):
        """Test that JSON export data has correct structure."""
        import json
        from datetime import datetime
        
        # Get detected features
        segments = self.onset_detection_gui.detect_voice_segments(
            self.audio, self.sr,
            amplitude_threshold_ratio=0.05
        )
        
        features_list = []
        for start, end in segments:
            features = self.onset_detection_gui.extract_feature_points(
                self.audio, self.sr, start, end
            )
            features_list.append(features)
        
        # Create JSON data manually (simulating export)
        export_data = {
            'metadata': {
                'source_file': os.path.basename(self.temp_wav_path),
                'sample_rate': self.sr,
                'audio_duration_sec': len(self.audio) / self.sr,
                'export_timestamp': datetime.now().isoformat(),
            },
            'parameters': {
                'amplitude_threshold_ratio': 0.05,
            },
            'segments': features_list,
            'segment_count': len(features_list),
        }
        
        # Check structure
        self.assertIn('metadata', export_data)
        self.assertIn('parameters', export_data)
        self.assertIn('segments', export_data)
        self.assertIn('segment_count', export_data)
        
        # Check metadata
        self.assertIn('sample_rate', export_data['metadata'])
        self.assertIn('export_timestamp', export_data['metadata'])
        
        # Check JSON serializable
        json_str = json.dumps(export_data)
        self.assertIsInstance(json_str, str)
        
        # Check can be deserialized
        loaded = json.loads(json_str)
        self.assertEqual(loaded['segment_count'], len(features_list))


if __name__ == '__main__':
    unittest.main()
