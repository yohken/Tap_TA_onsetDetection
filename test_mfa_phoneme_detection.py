"""
Unit tests for MFA-based T-peak and A-onset detection using phoneme boundaries.

This test suite validates the new detect_ta_onsets_from_mfa_phonemes function
that uses MFA phoneme labels and timing information for accurate labeling.

Key functions tested:
- detect_ta_onsets_from_mfa_phonemes()
- plot_mfa_ta_detection_results()
- VOWEL_PHONEMES constant
"""

import unittest
import numpy as np
import os
import tempfile
import scipy.io.wavfile as wavfile
import onset_detection


class TestVowelPhonemeConstants(unittest.TestCase):
    """Test cases for VOWEL_PHONEMES constant."""
    
    def test_vowel_phonemes_exists(self):
        """Test that VOWEL_PHONEMES constant is exported."""
        self.assertTrue(hasattr(onset_detection, 'VOWEL_PHONEMES'))
    
    def test_vowel_phonemes_contains_common_vowels(self):
        """Test that VOWEL_PHONEMES contains common ARPABET vowels."""
        expected_vowels = {'AA', 'AE', 'AH', 'IH', 'IY', 'UH', 'UW'}
        for vowel in expected_vowels:
            self.assertIn(vowel, onset_detection.VOWEL_PHONEMES,
                         f"Expected vowel '{vowel}' in VOWEL_PHONEMES")
    
    def test_vowel_phonemes_contains_japanese_vowels(self):
        """Test that VOWEL_PHONEMES contains Japanese vowel labels."""
        japanese_vowels = {'a', 'i', 'u', 'e', 'o'}
        for vowel in japanese_vowels:
            self.assertIn(vowel, onset_detection.VOWEL_PHONEMES,
                         f"Expected Japanese vowel '{vowel}' in VOWEL_PHONEMES")


class TestDetectTaOnsetsFromMfaPhonemes(unittest.TestCase):
    """Test cases for detect_ta_onsets_from_mfa_phonemes function."""
    
    @classmethod
    def setUpClass(cls):
        """Create synthetic 'ta ta' audio with MFA-style TextGrid."""
        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        audio = np.zeros_like(t)
        
        # Create two 'ta' syllables with clear phoneme boundaries
        # First TA: t from 0.1-0.15, AH from 0.15-0.45
        # Second TA: t from 0.8-0.85, AH from 0.85-1.15
        cls.ta_phonemes = [
            {'t_start': 0.1, 't_end': 0.15, 'vowel': 'AH', 'v_start': 0.15, 'v_end': 0.45},
            {'t_start': 0.8, 't_end': 0.85, 'vowel': 'AH', 'v_start': 0.85, 'v_end': 1.15},
        ]
        
        for ta in cls.ta_phonemes:
            # T burst: high-frequency noise burst
            t_start_sample = int(ta['t_start'] * sr)
            t_end_sample = int(ta['t_end'] * sr)
            burst_duration = t_end_sample - t_start_sample
            
            # Create burst with decay (simulates consonant explosion)
            decay = np.exp(-np.linspace(0, 8, burst_duration))
            # High-frequency noise (white noise filtered to emphasize high frequencies)
            noise = np.random.randn(burst_duration)
            audio[t_start_sample:t_end_sample] = noise * decay * 0.6
            
            # Vowel: voiced signal with harmonics
            v_start_sample = int(ta['v_start'] * sr)
            v_end_sample = int(ta['v_end'] * sr)
            vowel_duration = v_end_sample - v_start_sample
            vowel_t = np.linspace(0, ta['v_end'] - ta['v_start'], vowel_duration)
            
            f0 = 150  # Fundamental frequency
            vowel = (
                0.5 * np.sin(2 * np.pi * f0 * vowel_t) +
                0.3 * np.sin(2 * np.pi * 2 * f0 * vowel_t) +
                0.15 * np.sin(2 * np.pi * 3 * f0 * vowel_t)
            )
            
            # Apply envelope
            fade_len = min(int(0.02 * sr), vowel_duration // 4)
            vowel[:fade_len] *= np.linspace(0, 1, fade_len)
            vowel[-fade_len:] *= np.linspace(1, 0, fade_len)
            
            audio[v_start_sample:v_end_sample] = vowel * 0.7
        
        # Add background noise
        audio += np.random.randn(len(audio)) * 0.01
        audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Save audio to temp file
        audio_int16 = (audio * 32767).astype(np.int16)
        cls.temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        cls.temp_wav_path = cls.temp_wav.name
        cls.temp_wav.close()
        wavfile.write(cls.temp_wav_path, sr, audio_int16)
        
        # Create MFA-style TextGrid file
        cls.temp_tg = tempfile.NamedTemporaryFile(suffix='.TextGrid', delete=False, mode='w')
        cls.temp_tg_path = cls.temp_tg.name
        
        # Write TextGrid content with proper phoneme intervals
        tg_content = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 2.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0.0
        xmax = 2.0
        intervals: size = 7
        intervals [1]:
            xmin = 0.0
            xmax = 0.1
            text = ""
        intervals [2]:
            xmin = 0.1
            xmax = 0.15
            text = "t"
        intervals [3]:
            xmin = 0.15
            xmax = 0.45
            text = "AH"
        intervals [4]:
            xmin = 0.45
            xmax = 0.8
            text = ""
        intervals [5]:
            xmin = 0.8
            xmax = 0.85
            text = "t"
        intervals [6]:
            xmin = 0.85
            xmax = 1.15
            text = "AH"
        intervals [7]:
            xmin = 1.15
            xmax = 2.0
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
    
    def test_function_exists(self):
        """Test that detect_ta_onsets_from_mfa_phonemes function exists."""
        self.assertTrue(hasattr(onset_detection, 'detect_ta_onsets_from_mfa_phonemes'))
    
    def test_basic_detection(self):
        """Test basic detection returns expected number of results."""
        t_peaks, a_onsets, t_bursts, details = onset_detection.detect_ta_onsets_from_mfa_phonemes(
            self.temp_wav_path,
            self.temp_tg_path
        )
        
        # Should detect 2 TA syllables
        self.assertEqual(len(t_peaks), 2, f"Expected 2 T-peaks, got {len(t_peaks)}")
        self.assertEqual(len(a_onsets), 2, f"Expected 2 A-onsets, got {len(a_onsets)}")
        self.assertEqual(len(t_bursts), 2, f"Expected 2 T-bursts, got {len(t_bursts)}")
        self.assertEqual(len(details), 2, f"Expected 2 detail dicts, got {len(details)}")
    
    def test_t_peak_timing(self):
        """Test that T-peaks are within the 't' phoneme intervals."""
        t_peaks, a_onsets, t_bursts, details = onset_detection.detect_ta_onsets_from_mfa_phonemes(
            self.temp_wav_path,
            self.temp_tg_path
        )
        
        for i, (t_peak, ta) in enumerate(zip(t_peaks, self.ta_phonemes)):
            self.assertGreaterEqual(
                t_peak, ta['t_start'],
                f"T-peak {i} at {t_peak}s should be >= t_start {ta['t_start']}s"
            )
            self.assertLessEqual(
                t_peak, ta['t_end'],
                f"T-peak {i} at {t_peak}s should be <= t_end {ta['t_end']}s"
            )
    
    def test_a_onset_from_mfa_boundary(self):
        """Test that A-onsets match MFA vowel boundaries."""
        t_peaks, a_onsets, t_bursts, details = onset_detection.detect_ta_onsets_from_mfa_phonemes(
            self.temp_wav_path,
            self.temp_tg_path
        )
        
        for i, (a_onset, ta) in enumerate(zip(a_onsets, self.ta_phonemes)):
            # A-onset should match the MFA vowel start time exactly
            self.assertAlmostEqual(
                a_onset, ta['v_start'], places=4,
                msg=f"A-onset {i} at {a_onset}s should match MFA vowel start {ta['v_start']}s"
            )
    
    def test_t_peak_before_a_onset(self):
        """Test that T-peaks are always before A-onsets."""
        t_peaks, a_onsets, t_bursts, details = onset_detection.detect_ta_onsets_from_mfa_phonemes(
            self.temp_wav_path,
            self.temp_tg_path
        )
        
        for t_peak, a_onset in zip(t_peaks, a_onsets):
            self.assertLess(
                t_peak, a_onset,
                f"T-peak ({t_peak}s) should be before A-onset ({a_onset}s)"
            )
    
    def test_detail_dict_structure(self):
        """Test that detail dicts contain all required keys."""
        t_peaks, a_onsets, t_bursts, details = onset_detection.detect_ta_onsets_from_mfa_phonemes(
            self.temp_wav_path,
            self.temp_tg_path
        )
        
        required_keys = [
            't_burst_onset_sec', 't_peak_sec', 'a_onset_sec',
            't_interval_start', 't_interval_end',
            'vowel_label', 'vowel_interval_start', 'vowel_interval_end',
            'detection_method'
        ]
        
        for detail in details:
            for key in required_keys:
                self.assertIn(key, detail, f"Detail dict missing key: {key}")
    
    def test_detection_method_is_mfa_phoneme(self):
        """Test that detection_method is 'mfa_phoneme' for all results."""
        t_peaks, a_onsets, t_bursts, details = onset_detection.detect_ta_onsets_from_mfa_phonemes(
            self.temp_wav_path,
            self.temp_tg_path
        )
        
        for detail in details:
            self.assertEqual(
                detail['detection_method'], 'mfa_phoneme',
                "Detection method should be 'mfa_phoneme'"
            )
    
    def test_vowel_label_extracted(self):
        """Test that vowel labels are correctly extracted from TextGrid."""
        t_peaks, a_onsets, t_bursts, details = onset_detection.detect_ta_onsets_from_mfa_phonemes(
            self.temp_wav_path,
            self.temp_tg_path
        )
        
        for detail in details:
            self.assertEqual(
                detail['vowel_label'], 'AH',
                f"Expected vowel label 'AH', got '{detail['vowel_label']}'"
            )
    
    def test_custom_consonant_labels(self):
        """Test detection with custom consonant labels."""
        # Create TextGrid with 'T' instead of 't'
        temp_tg2 = tempfile.NamedTemporaryFile(suffix='.TextGrid', delete=False, mode='w')
        tg_content = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 2.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0.0
        xmax = 2.0
        intervals: size = 3
        intervals [1]:
            xmin = 0.0
            xmax = 0.1
            text = ""
        intervals [2]:
            xmin = 0.1
            xmax = 0.15
            text = "T"
        intervals [3]:
            xmin = 0.15
            xmax = 2.0
            text = "AH"
'''
        temp_tg2.write(tg_content)
        temp_tg2.close()
        
        try:
            # Should detect with default labels (includes 'T')
            t_peaks, a_onsets, t_bursts, details = onset_detection.detect_ta_onsets_from_mfa_phonemes(
                self.temp_wav_path,
                temp_tg2.name
            )
            self.assertEqual(len(t_peaks), 1)
            
            # Should also work with custom labels
            t_peaks2, a_onsets2, t_bursts2, details2 = onset_detection.detect_ta_onsets_from_mfa_phonemes(
                self.temp_wav_path,
                temp_tg2.name,
                consonant_labels=['T']
            )
            self.assertEqual(len(t_peaks2), 1)
        finally:
            os.unlink(temp_tg2.name)
    
    def test_tier_not_found_raises_error(self):
        """Test that missing tier raises ValueError."""
        with self.assertRaises(ValueError) as context:
            onset_detection.detect_ta_onsets_from_mfa_phonemes(
                self.temp_wav_path,
                self.temp_tg_path,
                tier_name='nonexistent_tier'
            )
        
        self.assertIn('not found', str(context.exception).lower())


class TestMfaTaDetectionEdgeCases(unittest.TestCase):
    """Test edge cases for MFA-based TA detection."""
    
    def test_no_consonant_found(self):
        """Test handling when no target consonant is found in TextGrid."""
        # Create audio and TextGrid with no 't' phonemes
        sr = 44100
        audio = np.random.randn(sr) * 0.1
        audio_int16 = (audio * 32767).astype(np.int16)
        
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        wavfile.write(temp_wav.name, sr, audio_int16)
        temp_wav.close()
        
        temp_tg = tempfile.NamedTemporaryFile(suffix='.TextGrid', delete=False, mode='w')
        tg_content = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 1.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0.0
        xmax = 1.0
        intervals: size = 1
        intervals [1]:
            xmin = 0.0
            xmax = 1.0
            text = "AH"
'''
        temp_tg.write(tg_content)
        temp_tg.close()
        
        try:
            t_peaks, a_onsets, t_bursts, details = onset_detection.detect_ta_onsets_from_mfa_phonemes(
                temp_wav.name,
                temp_tg.name
            )
            
            # Should return empty arrays
            self.assertEqual(len(t_peaks), 0)
            self.assertEqual(len(a_onsets), 0)
            self.assertEqual(len(t_bursts), 0)
            self.assertEqual(len(details), 0)
        finally:
            os.unlink(temp_wav.name)
            os.unlink(temp_tg.name)
    
    def test_consonant_without_following_vowel(self):
        """Test handling when consonant has no following vowel."""
        # Create audio and TextGrid with 't' but no following vowel
        sr = 44100
        audio = np.random.randn(sr) * 0.1
        audio_int16 = (audio * 32767).astype(np.int16)
        
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        wavfile.write(temp_wav.name, sr, audio_int16)
        temp_wav.close()
        
        temp_tg = tempfile.NamedTemporaryFile(suffix='.TextGrid', delete=False, mode='w')
        tg_content = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 1.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0.0
        xmax = 1.0
        intervals: size = 2
        intervals [1]:
            xmin = 0.0
            xmax = 0.1
            text = "t"
        intervals [2]:
            xmin = 0.1
            xmax = 1.0
            text = ""
'''
        temp_tg.write(tg_content)
        temp_tg.close()
        
        try:
            t_peaks, a_onsets, t_bursts, details = onset_detection.detect_ta_onsets_from_mfa_phonemes(
                temp_wav.name,
                temp_tg.name
            )
            
            # Should still return T-peak and fallback A-onset (end of T interval)
            self.assertEqual(len(t_peaks), 1)
            self.assertEqual(len(a_onsets), 1)
            
            # A-onset should be at end of t interval (fallback)
            self.assertAlmostEqual(a_onsets[0], 0.1, places=4)
            self.assertEqual(details[0]['vowel_label'], 'unknown')
        finally:
            os.unlink(temp_wav.name)
            os.unlink(temp_tg.name)


class TestPlotMfaTaDetectionResults(unittest.TestCase):
    """Test cases for plot_mfa_ta_detection_results function."""
    
    def test_function_exists(self):
        """Test that plot_mfa_ta_detection_results function exists."""
        self.assertTrue(hasattr(onset_detection, 'plot_mfa_ta_detection_results'))
    
    def test_function_has_docstring(self):
        """Test that function has documentation."""
        self.assertIsNotNone(onset_detection.plot_mfa_ta_detection_results.__doc__)


class TestIntegrationNewFunctions(unittest.TestCase):
    """Integration tests for new MFA phoneme functions."""
    
    def test_new_functions_exported(self):
        """Test that new functions are properly exported from module."""
        required_functions = [
            'detect_ta_onsets_from_mfa_phonemes',
            'plot_mfa_ta_detection_results',
        ]
        
        for func_name in required_functions:
            self.assertTrue(
                hasattr(onset_detection, func_name),
                f"Module missing function: {func_name}"
            )
    
    def test_new_functions_have_docstrings(self):
        """Test that new functions have documentation."""
        functions = [
            onset_detection.detect_ta_onsets_from_mfa_phonemes,
            onset_detection.plot_mfa_ta_detection_results,
        ]
        
        for func in functions:
            self.assertIsNotNone(func.__doc__)
            self.assertGreater(len(func.__doc__.strip()), 100,
                             f"{func.__name__} has insufficient documentation")
    
    def test_backward_compatibility(self):
        """Test that existing functions still work correctly."""
        # Test that old functions are still available
        old_functions = [
            'detect_t_burst_onsets_from_mfa',
            'detect_ta_onsets_with_voicing',
            'detect_voicing_onset',
            'find_ta_transition',
        ]
        
        for func_name in old_functions:
            self.assertTrue(
                hasattr(onset_detection, func_name),
                f"Old function missing: {func_name}"
            )


if __name__ == '__main__':
    unittest.main()
