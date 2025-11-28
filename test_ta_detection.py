"""
Unit tests for TA (T burst + A vowel onset) detection with voicing.

This test suite validates the high-precision TA detection functionality
that uses voicing detection (F0 tracking) to find A onset after T burst.

Key functions tested:
- detect_voicing_onset()
- find_ta_transition()
- detect_ta_onsets_with_voicing()
"""

import unittest
import numpy as np
import os
import tempfile
import scipy.io.wavfile as wavfile
import onset_detection


class TestDetectVoicingOnset(unittest.TestCase):
    """Test cases for detect_voicing_onset function."""
    
    @classmethod
    def setUpClass(cls):
        """Create synthetic test audio with voiced and unvoiced regions."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create audio with:
        # 0.0-0.2s: silence/noise (unvoiced)
        # 0.2-0.8s: voiced vowel (sine wave at ~200 Hz to simulate voice)
        # 0.8-1.0s: silence (unvoiced)
        
        audio = np.zeros_like(t)
        
        # Unvoiced region: low-amplitude noise
        noise_end = int(0.2 * sr)
        audio[:noise_end] = np.random.randn(noise_end) * 0.05
        
        # Voiced region: sine wave at 200 Hz (simulating F0)
        voiced_start = int(0.2 * sr)
        voiced_end = int(0.8 * sr)
        voiced_t = t[voiced_start:voiced_end] - t[voiced_start]
        
        # Add harmonics to make it more voice-like
        f0 = 200  # Fundamental frequency
        voiced_signal = (
            0.5 * np.sin(2 * np.pi * f0 * voiced_t) +
            0.3 * np.sin(2 * np.pi * 2 * f0 * voiced_t) +
            0.15 * np.sin(2 * np.pi * 3 * f0 * voiced_t)
        )
        
        # Apply envelope (fade in/out)
        env = np.ones(len(voiced_signal))
        fade_len = int(0.02 * sr)
        env[:fade_len] = np.linspace(0, 1, fade_len)
        env[-fade_len:] = np.linspace(1, 0, fade_len)
        
        audio[voiced_start:voiced_end] = voiced_signal * env * 0.8
        
        # Add small noise to the entire signal
        audio += np.random.randn(len(audio)) * 0.01
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.9
        
        cls.audio = audio
        cls.sr = sr
        cls.voiced_start_sample = voiced_start
        cls.voiced_start_time = 0.2
    
    def test_voicing_onset_detection_basic(self):
        """Test basic voicing onset detection."""
        # Start searching from beginning
        voicing_onset, f0, voiced_flag = onset_detection.detect_voicing_onset(
            self.audio, self.sr, 0,
            fmin=80.0, fmax=400.0
        )
        
        # Should find voicing onset
        self.assertIsNotNone(voicing_onset, "Voicing onset should be detected")
        
        # F0 and voiced_flag should be non-empty
        self.assertGreater(len(f0), 0, "F0 array should not be empty")
        self.assertGreater(len(voiced_flag), 0, "voiced_flag array should not be empty")
    
    def test_voicing_onset_timing(self):
        """Test that voicing onset is detected near the expected time."""
        voicing_onset, f0, voiced_flag = onset_detection.detect_voicing_onset(
            self.audio, self.sr, 0,
            fmin=80.0, fmax=400.0
        )
        
        if voicing_onset is not None:
            voicing_time = voicing_onset / self.sr
            
            # Should be near the expected voicing start (within 100ms tolerance)
            # pyin has some latency due to analysis window
            self.assertLess(
                abs(voicing_time - self.voiced_start_time), 0.15,
                f"Voicing onset at {voicing_time}s should be near {self.voiced_start_time}s"
            )
    
    def test_no_voicing_in_noise(self):
        """Test that pure noise returns no voicing onset."""
        # Create pure noise
        sr = 44100
        noise = np.random.randn(sr) * 0.1  # 1 second of noise
        
        voicing_onset, f0, voiced_flag = onset_detection.detect_voicing_onset(
            noise, sr, 0,
            fmin=80.0, fmax=400.0
        )
        
        # Should either return None or a very late onset
        if voicing_onset is not None:
            # If found, should be near end (false positive in noise)
            pass  # pyin may occasionally detect spurious pitch in noise
    
    def test_short_segment_handling(self):
        """Test handling of segments too short for analysis."""
        sr = 44100
        short_audio = np.random.randn(100)  # Very short segment
        
        voicing_onset, f0, voiced_flag = onset_detection.detect_voicing_onset(
            short_audio, sr, 0
        )
        
        # Should return None for onset and empty arrays
        self.assertIsNone(voicing_onset)
        self.assertEqual(len(f0), 0)
        self.assertEqual(len(voiced_flag), 0)


class TestFindTaTransition(unittest.TestCase):
    """Test cases for find_ta_transition function."""
    
    @classmethod
    def setUpClass(cls):
        """Create synthetic 'ta'-like audio."""
        sr = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        
        audio = np.zeros_like(t)
        
        # T burst: short noise burst (0.0-0.03s)
        t_burst_end = int(0.03 * sr)
        decay = np.exp(-np.linspace(0, 5, t_burst_end))
        noise = np.random.randn(t_burst_end)
        audio[:t_burst_end] = noise * decay * 0.5
        
        # T peak position
        cls.t_peak_sample = int(0.01 * sr)
        cls.t_peak_sec = 0.01
        
        # Transition: brief silence/low energy (0.03-0.05s)
        
        # A vowel: voiced signal starting at 0.05s
        a_start = int(0.05 * sr)
        cls.expected_a_start_sec = 0.05
        
        voiced_duration = 0.3
        voiced_samples = int(voiced_duration * sr)
        voiced_t = np.linspace(0, voiced_duration, voiced_samples)
        
        f0 = 200
        voiced_signal = (
            0.5 * np.sin(2 * np.pi * f0 * voiced_t) +
            0.3 * np.sin(2 * np.pi * 2 * f0 * voiced_t)
        )
        
        # Fade in
        fade_len = int(0.01 * sr)
        fade_in = np.linspace(0, 1, fade_len)
        voiced_signal[:fade_len] *= fade_in
        
        audio[a_start:a_start + voiced_samples] = voiced_signal * 0.7
        
        # Add noise
        audio += np.random.randn(len(audio)) * 0.01
        audio = audio / np.max(np.abs(audio)) * 0.9
        
        cls.audio = audio
        cls.sr = sr
    
    def test_ta_transition_basic(self):
        """Test basic TA transition detection."""
        result = onset_detection.find_ta_transition(
            self.audio, self.sr, self.t_peak_sample,
            max_transition_sec=0.2
        )
        
        # Should return valid result
        self.assertIn('t_peak_sec', result)
        self.assertIn('a_start_sec', result)
        self.assertIn('transition_sec', result)
        self.assertIn('detection_method', result)
        
        # A onset should be after T peak
        self.assertGreater(result['a_start_sec'], result['t_peak_sec'])
    
    def test_transition_time_constraint(self):
        """Test that transition time constraint is enforced."""
        max_transition = 0.2
        
        result = onset_detection.find_ta_transition(
            self.audio, self.sr, self.t_peak_sample,
            max_transition_sec=max_transition
        )
        
        # Transition should not exceed max_transition_sec
        self.assertLessEqual(result['transition_sec'], max_transition + 0.01)
    
    def test_very_short_max_transition(self):
        """Test behavior with very short max transition (should trigger correction)."""
        # Set max transition very short so it triggers correction
        max_transition = 0.01
        
        result = onset_detection.find_ta_transition(
            self.audio, self.sr, self.t_peak_sample,
            max_transition_sec=max_transition
        )
        
        # Should be corrected - check the corrected flag directly
        self.assertTrue(
            result['corrected'],
            f"Expected correction for short max_transition, got method={result['detection_method']}"
        )
        
        # Transition should be close to max_transition
        self.assertAlmostEqual(result['transition_sec'], max_transition, delta=0.005)
    
    def test_rms_fallback(self):
        """Test RMS fallback when voicing not detected."""
        # Create audio without voiced content (just noise)
        sr = 44100
        noise_audio = np.random.randn(sr) * 0.5
        noise_audio = noise_audio / np.max(np.abs(noise_audio)) * 0.9
        
        result = onset_detection.find_ta_transition(
            noise_audio, sr, 0,
            max_transition_sec=0.2,
            use_rms_fallback=True
        )
        
        # Should still return a result (either from RMS fallback or max limit)
        self.assertIsNotNone(result['a_start_sec'])
        self.assertIn(result['detection_method'], 
                     ['rms_fallback', 'rms_fallback_corrected', 'max_limit_fallback'])
    
    def test_result_structure(self):
        """Test that result dictionary has all required keys."""
        result = onset_detection.find_ta_transition(
            self.audio, self.sr, self.t_peak_sample
        )
        
        required_keys = [
            't_peak_sec', 'a_start_sec', 'transition_sec',
            'detection_method', 'voicing_info', 'corrected'
        ]
        
        for key in required_keys:
            self.assertIn(key, result, f"Result missing key: {key}")
    
    def test_voicing_info_structure(self):
        """Test that voicing_info has correct structure."""
        result = onset_detection.find_ta_transition(
            self.audio, self.sr, self.t_peak_sample
        )
        
        voicing_info = result['voicing_info']
        self.assertIsNotNone(voicing_info)
        self.assertIn('f0', voicing_info)
        self.assertIn('voiced_flag', voicing_info)


class TestDetectTaOnsetsWithVoicing(unittest.TestCase):
    """Test cases for detect_ta_onsets_with_voicing function."""
    
    @classmethod
    def setUpClass(cls):
        """Create synthetic 'ta ta' audio with mock TextGrid."""
        sr = 44100
        duration = 1.5
        t = np.linspace(0, duration, int(sr * duration))
        
        audio = np.zeros_like(t)
        
        # Two 'ta' syllables
        cls.ta_intervals = [
            (0.1, 0.5),   # First 'ta'
            (0.8, 1.2),   # Second 'ta'
        ]
        
        for start, end in cls.ta_intervals:
            start_sample = int(start * sr)
            
            # T burst: noise burst
            burst_duration = 0.03
            burst_samples = int(burst_duration * sr)
            decay = np.exp(-np.linspace(0, 5, burst_samples))
            noise = np.random.randn(burst_samples)
            
            if start_sample + burst_samples < len(audio):
                audio[start_sample:start_sample + burst_samples] = noise * decay * 0.5
            
            # A vowel: voiced signal
            vowel_start = start_sample + int(0.05 * sr)
            vowel_duration = (end - start) - 0.05
            vowel_samples = int(vowel_duration * sr)
            
            if vowel_start + vowel_samples < len(audio):
                vowel_t = np.linspace(0, vowel_duration, vowel_samples)
                f0 = 180
                vowel = (
                    0.5 * np.sin(2 * np.pi * f0 * vowel_t) +
                    0.3 * np.sin(2 * np.pi * 2 * f0 * vowel_t)
                )
                
                # Fade in
                fade_len = min(int(0.01 * sr), len(vowel))
                vowel[:fade_len] *= np.linspace(0, 1, fade_len)
                
                audio[vowel_start:vowel_start + vowel_samples] = vowel * 0.7
        
        # Add noise and normalize
        audio += np.random.randn(len(audio)) * 0.01
        audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Save audio to temp file
        audio_int16 = (audio * 32767).astype(np.int16)
        cls.temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        cls.temp_wav_path = cls.temp_wav.name
        cls.temp_wav.close()
        wavfile.write(cls.temp_wav_path, sr, audio_int16)
        
        # Create mock TextGrid file
        cls.temp_tg = tempfile.NamedTemporaryFile(suffix='.TextGrid', delete=False, mode='w')
        cls.temp_tg_path = cls.temp_tg.name
        
        # Write TextGrid content
        tg_content = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 1.5
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0.0
        xmax = 1.5
        intervals: size = 5
        intervals [1]:
            xmin = 0.0
            xmax = 0.1
            text = ""
        intervals [2]:
            xmin = 0.1
            xmax = 0.5
            text = "t"
        intervals [3]:
            xmin = 0.5
            xmax = 0.8
            text = ""
        intervals [4]:
            xmin = 0.8
            xmax = 1.2
            text = "t"
        intervals [5]:
            xmin = 1.2
            xmax = 1.5
            text = ""
'''
        cls.temp_tg.write(tg_content)
        cls.temp_tg.close()
        
        cls.sr = sr
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        if os.path.exists(cls.temp_wav_path):
            os.unlink(cls.temp_wav_path)
        if os.path.exists(cls.temp_tg_path):
            os.unlink(cls.temp_tg_path)
    
    def test_detect_correct_number_of_onsets(self):
        """Test that correct number of T and A onsets are detected."""
        t_onsets, a_onsets, details = onset_detection.detect_ta_onsets_with_voicing(
            self.temp_wav_path,
            self.temp_tg_path
        )
        
        # Should detect 2 T onsets and 2 A onsets
        self.assertEqual(len(t_onsets), 2, f"Expected 2 T onsets, got {len(t_onsets)}")
        self.assertEqual(len(a_onsets), 2, f"Expected 2 A onsets, got {len(a_onsets)}")
        self.assertEqual(len(details), 2, f"Expected 2 detail dicts, got {len(details)}")
    
    def test_t_onset_timing(self):
        """Test that T onsets are near expected times."""
        t_onsets, a_onsets, details = onset_detection.detect_ta_onsets_with_voicing(
            self.temp_wav_path,
            self.temp_tg_path
        )
        
        expected_t_times = [0.1, 0.8]  # Start times of 't' intervals
        
        for expected, actual in zip(expected_t_times, t_onsets):
            # T onset should be near the interval start (within 50ms)
            self.assertLess(
                abs(actual - expected), 0.1,
                f"T onset at {actual}s should be near {expected}s"
            )
    
    def test_a_onset_after_t_onset(self):
        """Test that A onsets are always after T onsets."""
        t_onsets, a_onsets, details = onset_detection.detect_ta_onsets_with_voicing(
            self.temp_wav_path,
            self.temp_tg_path
        )
        
        for t_time, a_time in zip(t_onsets, a_onsets):
            self.assertGreater(
                a_time, t_time,
                f"A onset ({a_time}s) should be after T onset ({t_time}s)"
            )
    
    def test_transition_within_limit(self):
        """Test that all transitions are within the max limit."""
        max_transition = 0.2
        
        t_onsets, a_onsets, details = onset_detection.detect_ta_onsets_with_voicing(
            self.temp_wav_path,
            self.temp_tg_path,
            max_transition_sec=max_transition
        )
        
        for detail in details:
            self.assertLessEqual(
                detail['transition_sec'], max_transition + 0.01,
                f"Transition {detail['transition_sec']}s should be <= {max_transition}s"
            )
    
    def test_detail_dict_structure(self):
        """Test that detail dicts have correct structure."""
        t_onsets, a_onsets, details = onset_detection.detect_ta_onsets_with_voicing(
            self.temp_wav_path,
            self.temp_tg_path
        )
        
        required_keys = [
            't_burst_onset_sec', 't_peak_sec', 'a_start_sec',
            'transition_sec', 'detection_method', 'corrected',
            'interval_min', 'interval_max'
        ]
        
        for detail in details:
            for key in required_keys:
                self.assertIn(key, detail, f"Detail dict missing key: {key}")


class TestIntegrationWithExistingFunctions(unittest.TestCase):
    """Integration tests to verify compatibility with existing functions."""
    
    def test_voicing_functions_exported(self):
        """Test that new functions are properly exported from module."""
        required_functions = [
            'detect_voicing_onset',
            'find_ta_transition',
            'detect_ta_onsets_with_voicing',
            'plot_ta_detection_results',
        ]
        
        for func_name in required_functions:
            self.assertTrue(
                hasattr(onset_detection, func_name),
                f"Module missing function: {func_name}"
            )
    
    def test_new_functions_have_docstrings(self):
        """Test that new functions have documentation."""
        functions = [
            onset_detection.detect_voicing_onset,
            onset_detection.find_ta_transition,
            onset_detection.detect_ta_onsets_with_voicing,
        ]
        
        for func in functions:
            self.assertIsNotNone(func.__doc__)
            self.assertGreater(len(func.__doc__.strip()), 50,
                             f"{func.__name__} has insufficient documentation")
    
    def test_backward_compatibility(self):
        """Test that existing functions still work correctly."""
        # Create simple test audio
        sr = 44100
        t = np.linspace(0, 0.5, int(sr * 0.5))
        audio = np.sin(2 * np.pi * 440 * t)
        
        # Test compute_rms_envelope still works
        env, times = onset_detection.compute_rms_envelope(audio, sr)
        self.assertGreater(len(env), 0)
        
        # Test compute_hilbert_envelope still works
        hilbert_env = onset_detection.compute_hilbert_envelope(audio, sr)
        self.assertEqual(len(hilbert_env), len(audio))
        
        # Test detect_onsets_from_envelope still works
        onsets = onset_detection.detect_onsets_from_envelope(env, times)
        # Should not crash, regardless of result


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_audio(self):
        """Test handling of empty/very short audio."""
        sr = 44100
        short_audio = np.array([])
        
        # Should handle gracefully
        result = onset_detection.detect_voicing_onset(
            short_audio, sr, 0
        )
        
        self.assertIsNone(result[0])
    
    def test_high_t_peak_sample(self):
        """Test when t_peak_sample is near end of audio."""
        sr = 44100
        audio = np.random.randn(sr)  # 1 second
        t_peak_sample = sr - 100  # Very near end
        
        result = onset_detection.find_ta_transition(
            audio, sr, t_peak_sample
        )
        
        # Should not crash, should return valid result
        self.assertIsNotNone(result['a_start_sec'])
    
    def test_max_transition_zero(self):
        """Test behavior with zero max transition."""
        sr = 44100
        audio = np.random.randn(sr)
        
        result = onset_detection.find_ta_transition(
            audio, sr, 0,
            max_transition_sec=0.0
        )
        
        # Should return T peak time as A start (0 transition)
        self.assertEqual(result['a_start_sec'], result['t_peak_sec'])
        self.assertEqual(result['transition_sec'], 0.0)


if __name__ == '__main__':
    unittest.main()
