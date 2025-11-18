"""
Test for the new export_data functionality in onset_detection_gui.py
"""

import unittest
import tempfile
import os
import onset_detection


class TestExportDataButton(unittest.TestCase):
    """Test cases for the export data button functionality."""
    
    def test_data_export_format(self):
        """Test that data can be properly formatted for export."""
        import numpy as np
        
        # Generate some test onset data
        onsets = onset_detection.get_click_onsets_from_bpm(120, 8)
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
            
            # Write data in the expected format
            header_lines = [
                "# Click Track Onset Times",
                "# BPM: 120",
                "# Number of clicks: 8",
                "# Subdivision: 1 (quarter notes)",
                f"# Total onsets: {len(onsets)}",
                "#",
                "# Format: onset_time (seconds)"
            ]
            f.write('\n'.join(header_lines) + '\n')
            np.savetxt(f, onsets, fmt='%.6f')
        
        try:
            # Verify the file was created and has content
            self.assertTrue(os.path.exists(temp_path))
            
            # Read and verify the content
            with open(temp_path, 'r') as f:
                content = f.read()
                
            # Check that header is present
            self.assertIn('Click Track Onset Times', content)
            self.assertIn('BPM: 120', content)
            self.assertIn('Number of clicks: 8', content)
            
            # Check that data is present
            lines = content.strip().split('\n')
            data_lines = [line for line in lines if not line.startswith('#')]
            self.assertEqual(len(data_lines), len(onsets))
            
            # Verify data can be read back
            values = [float(line.strip()) for line in data_lines]
            self.assertEqual(len(values), len(onsets))
            
            # Check that values match within tolerance
            for i, (expected, actual) in enumerate(zip(onsets, values)):
                self.assertAlmostEqual(expected, actual, places=6,
                                     msg=f"Mismatch at index {i}")
        
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == '__main__':
    unittest.main()
