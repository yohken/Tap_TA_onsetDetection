"""
GUI application for onset detection with file selection and result plotting.

This module provides a graphical user interface for:
1. Tap onset detection from WAV files
2. /t/ burst onset detection from WAV files with automatic MFA TextGrid generation

Features:
- File selection dialogs for audio files
- Automatic MFA TextGrid generation for /t/ burst detection
- Automatic visualization of detection results
- User-friendly interface using tkinter

Target Python version: 3.10+
Dependencies: tkinter (standard library), onset_detection module, mfa_onset_pipeline module
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import librosa
import matplotlib
import matplotlib.pyplot as plt
import onset_detection
import onset_hilbert
import os
import tempfile
from pathlib import Path
import mfa_onset_pipeline

# Configure matplotlib backend for GUI
# TkAgg will be used automatically when available


class OnsetDetectionGUI:
    """Main GUI application for onset detection."""
    
    def __init__(self, root: tk.Tk):
        """Initialize the GUI application.
        
        Args:
            root: The root tkinter window.
        """
        self.root = root
        self.root.title("Onset Detection Tool")
        self.root.geometry("800x600")
        
        # Create main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for responsiveness
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title label
        title_label = ttk.Label(
            main_frame, 
            text="Onset Detection Tool",
            font=('Helvetica', 16, 'bold')
        )
        title_label.grid(row=0, column=0, pady=20)
        
        # Description label
        desc_label = ttk.Label(
            main_frame,
            text="Select an onset detection method and choose your audio file(s)",
            wraplength=700
        )
        desc_label.grid(row=1, column=0, pady=10)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=20)
        
        # Tap detection button
        self.tap_button = ttk.Button(
            button_frame,
            text="Detect Tap Onsets",
            command=self.detect_tap_onsets,
            width=30
        )
        self.tap_button.grid(row=0, column=0, pady=10, padx=10)
        
        # /t/ burst detection button
        self.burst_button = ttk.Button(
            button_frame,
            text="Detect /t/ Burst Onsets",
            command=self.detect_t_burst_onsets,
            width=30
        )
        self.burst_button.grid(row=1, column=0, pady=10, padx=10)
        
        # Status label
        self.status_label = ttk.Label(
            main_frame,
            text="Ready",
            font=('Helvetica', 10),
            foreground='green'
        )
        self.status_label.grid(row=3, column=0, pady=10)
        
        # Results text area
        result_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        result_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        main_frame.rowconfigure(4, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        # Text widget with scrollbar
        self.result_text = tk.Text(result_frame, height=10, wrap=tk.WORD)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text['yscrollcommand'] = scrollbar.set
    
    def update_status(self, message: str, color: str = 'black'):
        """Update the status label.
        
        Args:
            message: Status message to display.
            color: Text color (default: black).
        """
        self.status_label.config(text=message, foreground=color)
        self.root.update_idletasks()
    
    def append_result(self, text: str):
        """Append text to the results area.
        
        Args:
            text: Text to append.
        """
        self.result_text.insert(tk.END, text + "\n")
        self.result_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_results(self):
        """Clear the results text area."""
        self.result_text.delete(1.0, tk.END)
    
    def detect_tap_onsets(self):
        """Handle tap onset detection with file selection."""
        self.clear_results()
        self.update_status("Select WAV file(s) for tap detection...", 'blue')
        
        # Open file dialog for WAV file(s) - now supports multiple selection
        wav_paths = filedialog.askopenfilenames(
            title="Select WAV file(s) for tap detection",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if not wav_paths:
            self.update_status("Selection cancelled", 'orange')
            return
        
        # Convert to list
        wav_paths = list(wav_paths)
        
        # Process files starting with the first one
        self._process_tap_files(wav_paths, 0)
    
    def _process_tap_files(self, wav_paths, current_index):
        """Process tap onset detection for a specific file in the list."""
        if current_index >= len(wav_paths):
            self.update_status("All files processed!", 'green')
            return
        
        wav_path = wav_paths[current_index]
        
        try:
            self.clear_results()
            self.update_status("Processing...", 'blue')
            self.append_result(f"Processing file {current_index + 1} of {len(wav_paths)}: {os.path.basename(wav_path)}")
            self.append_result("=" * 60)
            
            # Initial detection with default parameters using Fujii method
            hp_cutoff = 300.0
            threshold_ratio = 0.1
            min_distance_ms = 100.0
            
            onset_times, peak_times = onset_hilbert.detect_tap_onsets_and_peaks(
                wav_path,
                hp_cutoff_hz=hp_cutoff,
                threshold_ratio=threshold_ratio,
                min_distance_ms=min_distance_ms
            )
            
            # Display results
            self.append_result(f"\nDetected {len(onset_times)} tap onsets using Fujii method:")
            for i, (ot, pt) in enumerate(zip(onset_times, peak_times), 1):
                self.append_result(f"  {i}. onset={ot:.3f}s, peak={pt:.3f}s")
            
            # Plot results with interactive controls using Fujii method
            self.append_result("\nGenerating interactive plot...")
            self.append_result("Use the slider to adjust HPF frequency and click 'Re-detect' to update.")
            self.append_result("Cmd+Shift+Click to delete onset/peak markers.")
            self.append_result("Re-detection uses Fujii method (10% threshold, backward search, linear interpolation).")
            
            # Callback for next file navigation
            def on_next_file():
                if current_index + 1 < len(wav_paths):
                    # Process next file in the list
                    self._process_tap_files(wav_paths, current_index + 1)
                else:
                    # Show file dialog for new file selection
                    self.detect_tap_onsets()
            
            onset_hilbert.plot_waveform_and_envelope_interactive(
                wav_path,
                initial_hp_cutoff_hz=hp_cutoff,
                is_click=False,
                threshold_ratio=threshold_ratio,
                min_distance_ms=min_distance_ms,
                title=f"Tap Onset Detection (Fujii Method) - {os.path.basename(wav_path)} ({current_index + 1}/{len(wav_paths)})",
                on_next_callback=on_next_file,
                enable_marker_deletion=True,
                enable_export=True
            )
            
            self.update_status("Detection complete!", 'green')
            self.append_result("\n" + "=" * 60)
            self.append_result("Plot window opened. Close it to continue.")
            
        except Exception as e:
            error_msg = f"Error during tap detection: {str(e)}"
            self.append_result(f"\nERROR: {error_msg}")
            self.update_status("Error occurred", 'red')
            messagebox.showerror("Error", error_msg)
    
    def detect_t_burst_onsets(self):
        """Handle /t/ burst onset detection with file selection and automatic MFA TextGrid generation."""
        self.clear_results()
        self.update_status("Select WAV file for /t/ burst detection...", 'blue')
        
        # Open file dialog for WAV file
        wav_path = filedialog.askopenfilename(
            title="Select WAV file for /t/ burst detection",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if not wav_path:
            self.update_status("Selection cancelled", 'orange')
            return
        
        try:
            self.update_status("Generating TextGrid with MFA...", 'blue')
            self.append_result(f"Processing file: {os.path.basename(wav_path)}")
            self.append_result("=" * 60)
            self.append_result("Step 1: Running MFA alignment to generate TextGrid...")
            self.root.update_idletasks()
            
            # Create MFA pipeline
            wav_path_obj = Path(wav_path)
            
            # Create temporary directories for MFA
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                corpus_dir = temp_dir_path / "corpus"
                output_dir = temp_dir_path / "output"
                
                # Initialize MFA pipeline
                pipeline = mfa_onset_pipeline.MFAOnsetPipeline(
                    output_dir=output_dir,
                    log_level=30  # WARNING level to reduce output
                )
                
                # Check if MFA is available
                if not pipeline.check_mfa_available():
                    error_msg = "MFA (Montreal Forced Aligner) is not installed or not available.\n"
                    error_msg += "Please install MFA: https://montreal-forced-aligner.readthedocs.io/"
                    self.append_result(f"\nERROR: {error_msg}")
                    self.update_status("MFA not available", 'red')
                    messagebox.showerror("MFA Not Available", error_msg)
                    return
                
                # Create corpus structure
                self.append_result("  Creating corpus structure...")
                self.root.update_idletasks()
                if not pipeline.create_corpus_structure([wav_path_obj], corpus_dir, text_content="ta"):
                    error_msg = "Failed to create MFA corpus structure"
                    self.append_result(f"\nERROR: {error_msg}")
                    self.update_status("Error occurred", 'red')
                    messagebox.showerror("Error", error_msg)
                    return
                
                # Run MFA alignment
                self.append_result("  Running MFA alignment (this may take a moment)...")
                self.root.update_idletasks()
                mfa_output_dir = temp_dir_path / "mfa_output"
                if not pipeline.run_mfa_alignment(corpus_dir, mfa_output_dir):
                    error_msg = "MFA alignment failed. Please check that the audio file contains speech."
                    self.append_result(f"\nERROR: {error_msg}")
                    self.update_status("MFA alignment failed", 'red')
                    messagebox.showerror("Error", error_msg)
                    return
                
                # Find the generated TextGrid file
                tg_path = mfa_output_dir / f"{wav_path_obj.stem}.TextGrid"
                if not tg_path.exists():
                    error_msg = f"TextGrid file not found: {tg_path}"
                    self.append_result(f"\nERROR: {error_msg}")
                    self.update_status("TextGrid not found", 'red')
                    messagebox.showerror("Error", error_msg)
                    return
                
                self.append_result("  TextGrid generated successfully!")
                self.append_result("\nStep 2: Detecting /t/ burst onsets...")
                self.update_status("Processing...", 'blue')
                self.root.update_idletasks()
                
                # Load audio for interactive plotting
                y, sr = librosa.load(wav_path, sr=None, mono=True)
                
                # Initial detection with default parameters
                # Note: /t/ burst detection uses RMS envelope method from onset_detection
                # as it's optimized for TextGrid-guided detection
                high_freq_min = 2000.0
                diff_threshold_std = 2.0
                
                onset_times = onset_detection.detect_t_burst_onsets_from_mfa(
                    wav_path,
                    str(tg_path),
                    tier_name="phones",
                    phone_label="t",
                    high_freq_min=high_freq_min,
                    diff_threshold_std=diff_threshold_std
                )
                
                # Display results
                self.append_result(f"\nDetected {len(onset_times)} /t/ burst onsets:")
                for i, t in enumerate(onset_times, 1):
                    self.append_result(f"  {i}. {t:.3f} seconds")
                
                # Plot results with interactive controls
                self.append_result("\nGenerating interactive plot...")
                self.append_result("Use the slider to adjust HPF frequency and click 'Re-detect' to update.")
                self.append_result("Note: /t/ burst uses RMS envelope method (not Fujii method).")
                onset_detection.plot_envelope_with_onsets_interactive(
                    wav_path, y, sr,
                    initial_hp_cutoff=high_freq_min,
                    diff_threshold_std=diff_threshold_std,
                    min_interval_ms=50.0,
                    title=f"/t/ Burst Onset Detection - {os.path.basename(wav_path)}",
                    detection_type="t_burst"
                )
                
                self.update_status("Detection complete!", 'green')
                self.append_result("\n" + "=" * 60)
                self.append_result("Plot window opened. Close it to continue.")
            
        except Exception as e:
            error_msg = f"Error during /t/ burst detection: {str(e)}"
            self.append_result(f"\nERROR: {error_msg}")
            self.update_status("Error occurred", 'red')
            messagebox.showerror("Error", error_msg)
    



def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = OnsetDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
