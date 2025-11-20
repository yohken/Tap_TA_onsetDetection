"""
GUI application for onset detection with file selection and result plotting.

This module provides a graphical user interface for:
1. Tap onset detection from WAV files
2. /t/ burst onset detection from WAV and TextGrid files

Features:
- File selection dialogs for audio and TextGrid files
- Automatic visualization of detection results
- User-friendly interface using tkinter

Target Python version: 3.10+
Dependencies: tkinter (standard library), onset_detection module
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
        
        # Click track generation button
        self.click_button = ttk.Button(
            button_frame,
            text="Generate Click Track",
            command=self.generate_click_track,
            width=30
        )
        self.click_button.grid(row=2, column=0, pady=10, padx=10)
        
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
        """Handle /t/ burst onset detection with file selection."""
        self.clear_results()
        self.update_status("Select files for /t/ burst detection...", 'blue')
        
        # Open file dialog for WAV file
        wav_path = filedialog.askopenfilename(
            title="Select WAV file for /t/ burst detection",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if not wav_path:
            self.update_status("Selection cancelled", 'orange')
            return
        
        # Open file dialog for TextGrid file
        tg_path = filedialog.askopenfilename(
            title="Select TextGrid file",
            filetypes=[("TextGrid files", "*.TextGrid"), ("All files", "*.*")],
            initialdir=os.path.dirname(wav_path)
        )
        
        if not tg_path:
            self.update_status("Selection cancelled", 'orange')
            return
        
        try:
            self.update_status("Processing...", 'blue')
            self.append_result(f"Processing files:")
            self.append_result(f"  WAV: {os.path.basename(wav_path)}")
            self.append_result(f"  TextGrid: {os.path.basename(tg_path)}")
            self.append_result("=" * 60)
            
            # Load audio for interactive plotting
            y, sr = librosa.load(wav_path, sr=None, mono=True)
            
            # Initial detection with default parameters
            # Note: /t/ burst detection uses RMS envelope method from onset_detection
            # as it's optimized for TextGrid-guided detection
            high_freq_min = 2000.0
            diff_threshold_std = 2.0
            
            onset_times = onset_detection.detect_t_burst_onsets_from_mfa(
                wav_path,
                tg_path,
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
    
    def generate_click_track(self):
        """Handle click track generation with parameter input."""
        self.clear_results()
        
        # Create a dialog for input parameters
        dialog = tk.Toplevel(self.root)
        dialog.title("Click Track Parameters")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        frame = ttk.Frame(dialog, padding="20")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # BPM input
        ttk.Label(frame, text="BPM (Beats Per Minute):").grid(row=0, column=0, sticky=tk.W, pady=5)
        bpm_var = tk.StringVar(value="120")
        bpm_entry = ttk.Entry(frame, textvariable=bpm_var, width=20)
        bpm_entry.grid(row=0, column=1, pady=5)
        
        # Number of clicks input
        ttk.Label(frame, text="Number of Clicks:").grid(row=1, column=0, sticky=tk.W, pady=5)
        n_clicks_var = tk.StringVar(value="8")
        n_clicks_entry = ttk.Entry(frame, textvariable=n_clicks_var, width=20)
        n_clicks_entry.grid(row=1, column=1, pady=5)
        
        # Subdivision input
        ttk.Label(frame, text="Subdivision:").grid(row=2, column=0, sticky=tk.W, pady=5)
        subdivision_var = tk.StringVar(value="1")
        subdivision_combo = ttk.Combobox(
            frame, 
            textvariable=subdivision_var,
            values=["1 (quarter notes)", "2 (eighth notes)", "4 (sixteenth notes)"],
            width=18,
            state='readonly'
        )
        subdivision_combo.grid(row=2, column=1, pady=5)
        subdivision_combo.set("1 (quarter notes)")
        
        result_container = {"onsets": None}
        
        def on_generate():
            try:
                bpm = float(bpm_var.get())
                n_clicks = int(n_clicks_var.get())
                subdivision_text = subdivision_var.get()
                subdivision = int(subdivision_text.split()[0])
                
                if bpm <= 0 or n_clicks <= 0 or subdivision <= 0:
                    raise ValueError("Values must be positive")
                
                result_container["onsets"] = onset_detection.get_click_onsets_from_bpm(
                    bpm, n_clicks, subdivision=subdivision
                )
                dialog.destroy()
                
            except ValueError as e:
                messagebox.showerror("Invalid Input", f"Please enter valid numbers: {str(e)}")
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Generate", command=on_generate).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).grid(row=0, column=1, padx=5)
        
        # Wait for dialog to close
        self.root.wait_window(dialog)
        
        # Process results if generated
        if result_container["onsets"] is not None:
            onset_times = result_container["onsets"]
            bpm = float(bpm_var.get())
            n_clicks = int(n_clicks_var.get())
            subdivision_text = subdivision_var.get()
            
            self.append_result(f"Click Track Generation")
            self.append_result("=" * 60)
            self.append_result(f"BPM: {bpm}")
            self.append_result(f"Number of clicks: {n_clicks}")
            self.append_result(f"Subdivision: {subdivision_text}")
            self.append_result(f"\nGenerated {len(onset_times)} click onsets:")
            for i, t in enumerate(onset_times, 1):
                self.append_result(f"  {i}. {t:.3f} seconds")
            
            self.update_status("Click track generated!", 'green')
            self.append_result("\n" + "=" * 60)


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = OnsetDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
