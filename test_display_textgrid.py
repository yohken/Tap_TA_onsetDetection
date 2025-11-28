"""
Unit tests for display_textgrid module.

This test suite validates:
1. TextGrid structure display functionality
2. Phoneme interval display functionality
3. Command-line argument handling
4. Error handling for missing files and tiers
"""

import unittest
import tempfile
import os
import sys
from io import StringIO

import display_textgrid


class TestDisplayTextGrid(unittest.TestCase):
    """Test cases for TextGrid display functionality."""

    @classmethod
    def setUpClass(cls):
        """Create temporary TextGrid files for testing."""
        # Create a standard test TextGrid file
        cls.temp_tg = tempfile.NamedTemporaryFile(
            mode='w', suffix='.TextGrid', delete=False
        )
        cls.temp_tg.write('''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0 
xmax = 1.0 
tiers? <exists> 
size = 2 
item []: 
    item [1]:
        class = "IntervalTier" 
        name = "phones" 
        xmin = 0 
        xmax = 1.0 
        intervals: size = 5 
        intervals [1]:
            xmin = 0.0 
            xmax = 0.10 
            text = "" 
        intervals [2]:
            xmin = 0.10 
            xmax = 0.15 
            text = "t" 
        intervals [3]:
            xmin = 0.15 
            xmax = 0.45 
            text = "AH" 
        intervals [4]:
            xmin = 0.45 
            xmax = 0.50 
            text = "t" 
        intervals [5]:
            xmin = 0.50 
            xmax = 1.0 
            text = "" 
    item [2]:
        class = "IntervalTier" 
        name = "words" 
        xmin = 0 
        xmax = 1.0 
        intervals: size = 2 
        intervals [1]:
            xmin = 0.0 
            xmax = 0.10 
            text = "" 
        intervals [2]:
            xmin = 0.10 
            xmax = 1.0 
            text = "ta" 
''')
        cls.temp_tg.close()
        cls.temp_tg_path = cls.temp_tg.name

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        if os.path.exists(cls.temp_tg_path):
            os.unlink(cls.temp_tg_path)

    def test_module_imports(self):
        """Test that display_textgrid module can be imported."""
        import display_textgrid
        self.assertTrue(hasattr(display_textgrid, 'display_textgrid_structure'))
        self.assertTrue(hasattr(display_textgrid, 'display_phones_tier'))
        self.assertTrue(hasattr(display_textgrid, 'main'))

    def test_display_textgrid_structure(self):
        """Test TextGrid structure display."""
        from textgrid import TextGrid
        tg = TextGrid.fromFile(self.temp_tg_path)

        # Capture output
        captured_output = StringIO()
        sys.stdout = captured_output
        display_textgrid.display_textgrid_structure(tg)
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        # Verify output contains expected content
        self.assertIn("TextGrid structure:", output)
        self.assertIn("phones", output)
        self.assertIn("words", output)
        self.assertIn("IntervalTier", output)
        self.assertIn("5 intervals", output)
        self.assertIn("2 intervals", output)

    def test_display_phones_tier(self):
        """Test phones tier display."""
        from textgrid import TextGrid
        tg = TextGrid.fromFile(self.temp_tg_path)

        # Capture output
        captured_output = StringIO()
        sys.stdout = captured_output
        display_textgrid.display_phones_tier(tg, tier_name="phones")
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        # Verify output contains expected content
        self.assertIn("Phoneme intervals (phones tier):", output)
        self.assertIn("0.10 - 0.15: t", output)
        self.assertIn("0.15 - 0.45: AH", output)

    def test_display_nonexistent_tier(self):
        """Test handling of nonexistent tier."""
        from textgrid import TextGrid
        tg = TextGrid.fromFile(self.temp_tg_path)

        # Capture output
        captured_output = StringIO()
        sys.stdout = captured_output
        display_textgrid.display_phones_tier(tg, tier_name="nonexistent")
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertIn("not found", output)

    def test_main_with_valid_file(self):
        """Test main function with valid TextGrid file."""
        # Patch sys.argv
        original_argv = sys.argv
        sys.argv = ['display_textgrid.py', self.temp_tg_path]

        # Capture output
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            result = display_textgrid.main()
        finally:
            sys.argv = original_argv
            sys.stdout = sys.__stdout__

        self.assertEqual(result, 0)
        output = captured_output.getvalue()
        self.assertIn("TextGrid structure:", output)
        self.assertIn("phones", output)

    def test_main_with_nonexistent_file(self):
        """Test main function with nonexistent file."""
        # Patch sys.argv
        original_argv = sys.argv
        sys.argv = ['display_textgrid.py', 'nonexistent.TextGrid']

        # Capture output
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            result = display_textgrid.main()
        finally:
            sys.argv = original_argv
            sys.stdout = sys.__stdout__

        self.assertEqual(result, 1)
        output = captured_output.getvalue()
        self.assertIn("Error", output)

    def test_main_with_tier_argument(self):
        """Test main function with --tier argument."""
        # Patch sys.argv
        original_argv = sys.argv
        sys.argv = ['display_textgrid.py', self.temp_tg_path, '--tier', 'words']

        # Capture output
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            result = display_textgrid.main()
        finally:
            sys.argv = original_argv
            sys.stdout = sys.__stdout__

        self.assertEqual(result, 0)
        output = captured_output.getvalue()
        self.assertIn("words tier", output)
        self.assertIn("ta", output)


if __name__ == '__main__':
    unittest.main()
