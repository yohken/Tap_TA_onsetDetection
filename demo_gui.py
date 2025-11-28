#!/usr/bin/env python3
"""
Demonstration script for the Onset Detection GUI.

This script shows how to use the GUI application for onset detection.
Run this to launch the graphical interface.

Usage:
    python demo_gui.py

Features:
    - Select audio files using file dialogs
    - Automatic onset detection
    - Visual plots of results
    - Support for multiple detection methods
"""

import sys
import os

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch the onset detection GUI."""
    try:
        import onset_detection_gui
        
        print("=" * 60)
        print("Onset Detection GUI")
        print("=" * 60)
        print("\nStarting graphical user interface...")
        print("\nGUI Features:")
        print("  • File selection dialogs for easy audio file selection")
        print("  • Automatic visualization of detection results")
        print("  • Support for tap and voice segment detection")
        print("  • Click track generation with custom parameters")
        print("\nInstructions:")
        print("  1. Click a button to select a detection method")
        print("  2. Choose your audio file(s) in the file dialog")
        print("  3. View results in the text area")
        print("  4. A plot window will open automatically")
        print("\n" + "=" * 60)
        print()
        
        # Launch the GUI
        onset_detection_gui.main()
        
    except ImportError as e:
        print(f"\nError: Could not import GUI module: {e}")
        print("\nPlease ensure tkinter is installed:")
        print("  Ubuntu/Debian: sudo apt-get install python3-tk")
        print("  macOS: Included with Python")
        print("  Windows: Included with Python")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
