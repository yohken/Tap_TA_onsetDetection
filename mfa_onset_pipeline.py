"""
MFA-based Onset Detection Pipeline

This script provides a comprehensive pipeline for detecting /t/ burst onsets using both:
1. MFA (Montreal Forced Aligner) TextGrid annotations
2. Hilbert-based detection (Fujii method)

The pipeline:
- Accepts multiple WAV files from user
- Automatically runs MFA alignment to generate TextGrid files
- Detects onsets using both MFA and Hilbert methods
- Creates comparison plots showing both results
- Exports results to CSV/JSON with full parameter details
- Logs processing steps and handles errors gracefully

Target Python version: 3.10+
Dependencies: numpy, scipy, librosa, textgrid, matplotlib, pandas, soundfile
External: mfa (Montreal Forced Aligner CLI)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import librosa
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import soundfile as sf
from textgrid import TextGrid

import onset_detection
import onset_hilbert


@dataclass
class MFAParameters:
    """Parameters for MFA-based /t/ burst detection"""
    high_freq_min: float = 2000.0
    frame_length_ms: float = 5.0
    hop_length_ms: float = 1.0
    diff_threshold_std: float = 2.0
    tier_name: str = "phones"
    phone_label: str = "t"
    # New: consonant labels for phoneme-based detection
    consonant_labels: Optional[list[str]] = None


@dataclass
class HilbertParameters:
    """Parameters for Hilbert-based onset detection"""
    target_sr: int = 48000
    hp_cutoff: float = 500.0
    threshold_ratio: float = 0.1
    lookback_points: int = 74
    min_interval_ms: float = 50.0
    prominence_ratio: float = 0.3


@dataclass
class TADetectionResult:
    """Result from TA (T-peak and A-onset) detection using MFA phonemes"""
    method: str
    t_peak_times: list[float]  # Consonant burst explosion points
    a_onset_times: list[float]  # Vowel onset times from MFA boundaries
    t_burst_onset_times: list[float]  # When T burst energy starts rising
    details: list[dict]  # Full detection details for each TA syllable
    parameters: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class DetectionResult:
    """Result from onset detection"""
    method: str
    onset_times: list[float]
    peak_times: Optional[list[float]] = None
    parameters: Optional[dict] = None
    error: Optional[str] = None


class MFAOnsetPipeline:
    """Pipeline for MFA-based onset detection with comparison to Hilbert method"""
    
    def __init__(
        self,
        mfa_params: Optional[MFAParameters] = None,
        hilbert_params: Optional[HilbertParameters] = None,
        mfa_model: str = "english_us_arpa",
        mfa_dictionary: str = "english_us_arpa",
        output_dir: Optional[Path] = None,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the MFA onset detection pipeline.
        
        Args:
            mfa_params: Parameters for MFA-based detection
            hilbert_params: Parameters for Hilbert-based detection
            mfa_model: MFA acoustic model name
            mfa_dictionary: MFA dictionary name
            output_dir: Directory for output files (plots, CSVs, logs)
            log_level: Logging level
        """
        self.mfa_params = mfa_params or MFAParameters()
        self.hilbert_params = hilbert_params or HilbertParameters()
        self.mfa_model = mfa_model
        self.mfa_dictionary = mfa_dictionary
        self.output_dir = output_dir or Path("./mfa_onset_results")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # Create file handler
        log_file = self.output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.logger.info(f"MFA Onset Pipeline initialized. Output directory: {self.output_dir}")
        self.logger.info(f"MFA parameters: {self.mfa_params}")
        self.logger.info(f"Hilbert parameters: {self.hilbert_params}")
    
    def check_mfa_available(self) -> bool:
        """Check if MFA is installed and available"""
        try:
            result = subprocess.run(
                ["mfa", "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.logger.info(f"MFA version: {result.stdout.strip()}")
                return True
            else:
                self.logger.warning("MFA command failed")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.warning(f"MFA not available: {e}")
            return False
    
    def create_corpus_structure(
        self,
        wav_files: list[Path],
        corpus_dir: Path,
        text_content: str = "ta"
    ) -> bool:
        """
        Create MFA corpus directory structure with WAV files and text files.
        
        Args:
            wav_files: List of WAV file paths
            corpus_dir: Corpus directory path
            text_content: Default text content for alignment (default: "ta" for /t/ burst)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            corpus_dir.mkdir(exist_ok=True, parents=True)
            
            for wav_file in wav_files:
                # Copy WAV file
                dest_wav = corpus_dir / wav_file.name
                shutil.copy2(wav_file, dest_wav)
                self.logger.info(f"Copied {wav_file.name} to corpus")
                
                # Create corresponding text file
                text_file = corpus_dir / f"{wav_file.stem}.txt"
                with open(text_file, 'w') as f:
                    f.write(text_content)
                self.logger.info(f"Created text file for {wav_file.name}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error creating corpus structure: {e}")
            return False
    
    def run_mfa_alignment(
        self,
        corpus_dir: Path,
        output_dir: Path
    ) -> bool:
        """
        Run MFA alignment on corpus directory.
        
        Args:
            corpus_dir: Corpus directory with WAV and text files
            output_dir: Output directory for TextGrid files
        
        Returns:
            True if successful, False otherwise
        """
        try:
            output_dir.mkdir(exist_ok=True, parents=True)
            
            cmd = [
                "mfa", "align",
                str(corpus_dir),
                self.mfa_dictionary,
                self.mfa_model,
                str(output_dir),
                "--clean"
            ]
            
            self.logger.info(f"Running MFA alignment: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                self.logger.info("MFA alignment completed successfully")
                self.logger.debug(f"MFA output: {result.stdout}")
                return True
            else:
                self.logger.error(f"MFA alignment failed: {result.stderr}")
                return False
        
        except subprocess.TimeoutExpired:
            self.logger.error("MFA alignment timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error running MFA alignment: {e}")
            return False
    
    def detect_mfa_onsets(
        self,
        wav_path: Path,
        textgrid_path: Path
    ) -> DetectionResult:
        """
        Detect /t/ burst onsets using MFA TextGrid.
        
        Args:
            wav_path: Path to WAV file
            textgrid_path: Path to TextGrid file
        
        Returns:
            DetectionResult with MFA-based onsets
        """
        try:
            onsets = onset_detection.detect_t_burst_onsets_from_mfa(
                str(wav_path),
                str(textgrid_path),
                tier_name=self.mfa_params.tier_name,
                phone_label=self.mfa_params.phone_label,
                high_freq_min=self.mfa_params.high_freq_min,
                frame_length_ms=self.mfa_params.frame_length_ms,
                hop_length_ms=self.mfa_params.hop_length_ms,
                diff_threshold_std=self.mfa_params.diff_threshold_std
            )
            
            self.logger.info(f"MFA detection: {len(onsets)} onsets found in {wav_path.name}")
            
            return DetectionResult(
                method="MFA",
                onset_times=onsets.tolist(),
                parameters=asdict(self.mfa_params)
            )
        
        except Exception as e:
            self.logger.error(f"Error in MFA detection for {wav_path.name}: {e}")
            return DetectionResult(
                method="MFA",
                onset_times=[],
                error=str(e)
            )
    
    def detect_ta_from_mfa_phonemes(
        self,
        wav_path: Path,
        textgrid_path: Path
    ) -> TADetectionResult:
        """
        Detect T-peak and A-onset using MFA phoneme boundaries.
        
        This method maximally leverages MFA's phoneme segmentation:
        - T-peak: High-frequency energy maximum within 't' phoneme interval
        - A-onset: Start time of following vowel phoneme from MFA TextGrid
        
        Args:
            wav_path: Path to WAV file
            textgrid_path: Path to TextGrid file
        
        Returns:
            TADetectionResult with T-peaks, A-onsets, and detailed information
        """
        try:
            # Determine consonant labels
            consonant_labels = self.mfa_params.consonant_labels
            if consonant_labels is None:
                consonant_labels = [self.mfa_params.phone_label, self.mfa_params.phone_label.upper()]
            
            t_peaks, a_onsets, t_bursts, details = onset_detection.detect_ta_onsets_from_mfa_phonemes(
                str(wav_path),
                str(textgrid_path),
                tier_name=self.mfa_params.tier_name,
                consonant_labels=consonant_labels,
                high_freq_min=self.mfa_params.high_freq_min,
                frame_length_ms=self.mfa_params.frame_length_ms,
                hop_length_ms=self.mfa_params.hop_length_ms
            )
            
            self.logger.info(
                f"MFA phoneme detection: {len(t_peaks)} T-peaks, "
                f"{len(a_onsets)} A-onsets found in {wav_path.name}"
            )
            
            return TADetectionResult(
                method="MFA_phoneme",
                t_peak_times=t_peaks.tolist(),
                a_onset_times=a_onsets.tolist(),
                t_burst_onset_times=t_bursts.tolist(),
                details=details,
                parameters=asdict(self.mfa_params)
            )
        
        except Exception as e:
            self.logger.error(f"Error in MFA phoneme detection for {wav_path.name}: {e}")
            return TADetectionResult(
                method="MFA_phoneme",
                t_peak_times=[],
                a_onset_times=[],
                t_burst_onset_times=[],
                details=[],
                error=str(e)
            )

    def detect_hilbert_onsets(
        self,
        wav_path: Path
    ) -> DetectionResult:
        """
        Detect onsets using Hilbert-based method (Fujii method).
        
        Args:
            wav_path: Path to WAV file
        
        Returns:
            DetectionResult with Hilbert-based onsets and peaks
        """
        try:
            # Load audio
            y, sr = sf.read(str(wav_path))
            if y.ndim > 1:
                y = y[:, 0]  # Take first channel if stereo
            
            # Resample if needed
            if sr != self.hilbert_params.target_sr:
                y = librosa.resample(
                    y,
                    orig_sr=sr,
                    target_sr=self.hilbert_params.target_sr
                )
                sr = self.hilbert_params.target_sr
            
            # Apply high-pass filter
            y_filtered = onset_hilbert.highpass_filter(
                y,
                sr,
                self.hilbert_params.hp_cutoff
            )
            
            # Compute Hilbert envelope
            env = onset_hilbert.hilbert_envelope(
                y_filtered,
                sr,
                smooth_ms=0.5  # Light smoothing
            )
            
            # Detect onsets and peaks
            min_distance_ms = self.hilbert_params.min_interval_ms
            onsets, peaks = onset_hilbert.detect_onsets_and_peaks_from_envelope(
                env,
                sr,
                threshold_ratio=self.hilbert_params.threshold_ratio,
                min_distance_ms=min_distance_ms,
                global_min_height_ratio=0.2
            )
            
            self.logger.info(f"Hilbert detection: {len(onsets)} onsets, {len(peaks)} peaks in {wav_path.name}")
            
            return DetectionResult(
                method="Hilbert",
                onset_times=onsets.tolist(),
                peak_times=peaks.tolist(),
                parameters=asdict(self.hilbert_params)
            )
        
        except Exception as e:
            self.logger.error(f"Error in Hilbert detection for {wav_path.name}: {e}")
            return DetectionResult(
                method="Hilbert",
                onset_times=[],
                error=str(e)
            )
    
    def plot_comparison(
        self,
        wav_path: Path,
        mfa_result: DetectionResult,
        hilbert_result: DetectionResult,
        textgrid_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Create comparison plot showing both MFA and Hilbert results.
        
        Creates a four-panel plot when TextGrid is available:
        - Panel 1: Waveform with both onset markers
        - Panel 2: MFA high-frequency RMS envelope
        - Panel 3: Hilbert envelope
        - Panel 4: MFA phoneme tier visualization (if TextGrid available)
        
        Args:
            wav_path: Path to WAV file
            mfa_result: MFA detection results
            hilbert_result: Hilbert detection results
            textgrid_path: Optional path to TextGrid (for phoneme visualization)
        
        Returns:
            Path to saved PNG file, or None if failed
        """
        try:
            # Load audio
            y, sr = librosa.load(str(wav_path), sr=None, mono=True)
            time_axis = np.arange(len(y)) / sr
            
            # Determine if we should show phoneme panel
            show_phoneme_panel = textgrid_path and textgrid_path.exists()
            n_panels = 4 if show_phoneme_panel else 3
            
            # Create figure with subplots
            fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels), sharex=True)
            ax1, ax2, ax3 = axes[0], axes[1], axes[2]
            ax4 = axes[3] if show_phoneme_panel else None
            
            # Plot 1: Waveform with both onset markers
            ax1.plot(time_axis, y, alpha=0.6, linewidth=0.5, color='gray', label='Waveform')
            ax1.set_ylabel('Amplitude')
            ax1.set_title(f'Onset Detection Comparison: {wav_path.name}')
            ax1.grid(True, alpha=0.3)
            
            # Mark MFA onsets
            for onset_t in mfa_result.onset_times:
                ax1.axvline(x=onset_t, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            
            # Mark Hilbert onsets
            for onset_t in hilbert_result.onset_times:
                ax1.axvline(x=onset_t, color='blue', linestyle='--', alpha=0.7, linewidth=1.5)
            
            # Mark Hilbert peaks
            if hilbert_result.peak_times:
                for peak_t in hilbert_result.peak_times:
                    ax1.axvline(x=peak_t, color='green', linestyle=':', alpha=0.5, linewidth=1.0)
            
            # Add legend with markers
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', linestyle='--', label=f'MFA Onsets (n={len(mfa_result.onset_times)})'),
                Line2D([0], [0], color='blue', linestyle='--', label=f'Hilbert Onsets (n={len(hilbert_result.onset_times)})'),
            ]
            if hilbert_result.peak_times:
                legend_elements.append(
                    Line2D([0], [0], color='green', linestyle=':', label=f'Hilbert Peaks (n={len(hilbert_result.peak_times)})')
                )
            ax1.legend(handles=legend_elements, loc='upper right')
            
            # Plot 2: MFA high-frequency RMS envelope
            if textgrid_path and textgrid_path.exists():
                # Compute high-frequency RMS envelope for visualization
                env_mfa, times_mfa = onset_detection.compute_rms_envelope(
                    y, sr,
                    band=(self.mfa_params.high_freq_min, None),
                    frame_length_ms=self.mfa_params.frame_length_ms,
                    hop_length_ms=self.mfa_params.hop_length_ms
                )
                ax2.plot(times_mfa, env_mfa, color='darkred', linewidth=1.0, label=f'MFA RMS (≥{self.mfa_params.high_freq_min:.0f} Hz)')
                
                # Mark MFA onsets
                for onset_t in mfa_result.onset_times:
                    ax2.axvline(x=onset_t, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                
                ax2.set_ylabel('RMS Amplitude')
                ax2.legend(loc='upper right')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'MFA TextGrid not available', 
                        transform=ax2.transAxes, ha='center', va='center')
                ax2.set_ylabel('MFA Envelope')
            
            # Plot 3: Hilbert envelope
            # Resample for visualization
            if sr != self.hilbert_params.target_sr:
                y_vis = librosa.resample(y, orig_sr=sr, target_sr=self.hilbert_params.target_sr)
                sr_vis = self.hilbert_params.target_sr
            else:
                y_vis = y
                sr_vis = sr
            
            # Apply high-pass filter and compute Hilbert envelope
            y_filtered = onset_hilbert.highpass_filter(
                y_vis, sr_vis, self.hilbert_params.hp_cutoff
            )
            env_hilbert = onset_hilbert.hilbert_envelope(
                y_filtered, sr_vis, smooth_ms=0.5
            )
            time_hilbert = np.arange(len(env_hilbert)) / sr_vis
            
            ax3.plot(time_hilbert, env_hilbert, color='darkblue', linewidth=1.0, 
                    label=f'Hilbert Envelope (HP {self.hilbert_params.hp_cutoff:.0f} Hz)')
            
            # Mark Hilbert onsets
            for onset_t in hilbert_result.onset_times:
                ax3.axvline(x=onset_t, color='blue', linestyle='--', alpha=0.7, linewidth=1.5)
            
            # Mark Hilbert peaks
            if hilbert_result.peak_times:
                for peak_t in hilbert_result.peak_times:
                    ax3.axvline(x=peak_t, color='green', linestyle=':', alpha=0.5, linewidth=1.0)
            
            ax3.set_ylabel('Hilbert Amplitude')
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: MFA phoneme tier (if TextGrid available)
            if show_phoneme_panel and ax4 is not None:
                ax4.set_ylabel('Phoneme')
                ax4.set_xlabel('Time (s)')
                ax4.set_ylim(0, 1)
                ax4.set_yticks([])
                
                try:
                    tg = TextGrid.fromFile(str(textgrid_path))
                    tier = None
                    for t in tg.tiers:
                        if t.name == self.mfa_params.tier_name:
                            tier = t
                            break
                    
                    if tier is not None:
                        for interval in tier:
                            if interval.mark:  # Only show non-empty labels
                                # Draw interval rectangle
                                rect_start = interval.minTime
                                rect_width = interval.maxTime - interval.minTime
                                
                                # Color based on phoneme type
                                if interval.mark in onset_detection.VOWEL_PHONEMES:
                                    color = 'lightblue'
                                elif interval.mark.lower() == 't':
                                    color = 'lightcoral'
                                else:
                                    color = 'lightgray'
                                
                                rect = Rectangle((rect_start, 0.1), rect_width, 0.8,
                                                facecolor=color, edgecolor='black',
                                                linewidth=0.5, alpha=0.7)
                                ax4.add_patch(rect)
                                
                                # Add label text
                                mid_time = (interval.minTime + interval.maxTime) / 2
                                ax4.text(mid_time, 0.5, interval.mark,
                                        ha='center', va='center', fontsize=10, fontweight='bold')
                        
                        # Add onset markers on phoneme tier
                        for onset_t in mfa_result.onset_times:
                            ax4.axvline(x=onset_t, color='red', linestyle='--', alpha=0.7, linewidth=1.2)
                        for onset_t in hilbert_result.onset_times:
                            ax4.axvline(x=onset_t, color='blue', linestyle='--', alpha=0.7, linewidth=1.2)
                    else:
                        ax4.text(0.5, 0.5, f'Tier "{self.mfa_params.tier_name}" not found in TextGrid',
                                transform=ax4.transAxes, ha='center', va='center')
                except Exception as e:
                    ax4.text(0.5, 0.5, f'Error loading TextGrid: {e}',
                            transform=ax4.transAxes, ha='center', va='center')
                
                ax4.grid(True, alpha=0.3, axis='x')
            else:
                # Set xlabel on ax3 when no phoneme panel
                ax3.set_xlabel('Time (s)')
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = self.output_dir / f"{wav_path.stem}_comparison.png"
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved comparison plot: {plot_filename}")
            
            plt.close(fig)
            
            return plot_filename
        
        except Exception as e:
            self.logger.error(f"Error creating comparison plot for {wav_path.name}: {e}")
            return None
    
    def export_results(
        self,
        wav_path: Path,
        mfa_result: DetectionResult,
        hilbert_result: DetectionResult
    ) -> tuple[Optional[Path], Optional[Path]]:
        """
        Export detection results to CSV and JSON.
        
        Args:
            wav_path: Path to WAV file
            mfa_result: MFA detection results
            hilbert_result: Hilbert detection results
        
        Returns:
            Tuple of (csv_path, json_path) or (None, None) if failed
        """
        try:
            base_name = wav_path.stem
            
            # Prepare data for CSV
            csv_rows = []
            
            # Add MFA results
            for onset_t in mfa_result.onset_times:
                csv_rows.append({
                    'method': 'MFA',
                    'onset_time_s': onset_t,
                    'peak_time_s': None,
                    'file': wav_path.name
                })
            
            # Add Hilbert results
            for i, onset_t in enumerate(hilbert_result.onset_times):
                peak_t = hilbert_result.peak_times[i] if hilbert_result.peak_times and i < len(hilbert_result.peak_times) else None
                csv_rows.append({
                    'method': 'Hilbert',
                    'onset_time_s': onset_t,
                    'peak_time_s': peak_t,
                    'file': wav_path.name
                })
            
            # Save CSV
            csv_path = self.output_dir / f"{base_name}_results.csv"
            df = pd.DataFrame(csv_rows)
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved CSV results: {csv_path}")
            
            # Prepare JSON with full details
            json_data = {
                'file': wav_path.name,
                'timestamp': datetime.now().isoformat(),
                'mfa_detection': {
                    'onsets': mfa_result.onset_times,
                    'count': len(mfa_result.onset_times),
                    'parameters': mfa_result.parameters,
                    'error': mfa_result.error
                },
                'hilbert_detection': {
                    'onsets': hilbert_result.onset_times,
                    'peaks': hilbert_result.peak_times,
                    'count': len(hilbert_result.onset_times),
                    'parameters': hilbert_result.parameters,
                    'error': hilbert_result.error
                }
            }
            
            # Save JSON
            json_path = self.output_dir / f"{base_name}_results.json"
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            self.logger.info(f"Saved JSON results: {json_path}")
            
            return csv_path, json_path
        
        except Exception as e:
            self.logger.error(f"Error exporting results for {wav_path.name}: {e}")
            return None, None
    
    def process_wav_file(
        self,
        wav_path: Path,
        textgrid_path: Optional[Path] = None,
        skip_mfa: bool = False
    ) -> dict:
        """
        Process a single WAV file through the complete pipeline.
        
        Args:
            wav_path: Path to WAV file
            textgrid_path: Optional pre-existing TextGrid path
            skip_mfa: Skip MFA alignment if True
        
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing {wav_path.name}")
        
        results = {
            'file': wav_path.name,
            'success': False,
            'mfa_result': None,
            'hilbert_result': None,
            'plot_path': None,
            'csv_path': None,
            'json_path': None
        }
        
        # Detect using Hilbert method (always possible)
        hilbert_result = self.detect_hilbert_onsets(wav_path)
        results['hilbert_result'] = hilbert_result
        
        # Detect using MFA method (if TextGrid available)
        if textgrid_path and textgrid_path.exists():
            mfa_result = self.detect_mfa_onsets(wav_path, textgrid_path)
            results['mfa_result'] = mfa_result
        elif not skip_mfa:
            self.logger.warning(f"TextGrid not provided for {wav_path.name}, MFA detection skipped")
            mfa_result = DetectionResult(method="MFA", onset_times=[], error="TextGrid not available")
            results['mfa_result'] = mfa_result
        else:
            mfa_result = DetectionResult(method="MFA", onset_times=[], error="MFA skipped")
            results['mfa_result'] = mfa_result
        
        # Create comparison plot
        plot_path = self.plot_comparison(
            wav_path,
            mfa_result,
            hilbert_result,
            textgrid_path
        )
        results['plot_path'] = plot_path
        
        # Export results
        csv_path, json_path = self.export_results(
            wav_path,
            mfa_result,
            hilbert_result
        )
        results['csv_path'] = csv_path
        results['json_path'] = json_path
        
        results['success'] = True
        return results
    
    def process_multiple_files(
        self,
        wav_files: list[Path],
        run_mfa: bool = True,
        text_content: str = "ta"
    ) -> list[dict]:
        """
        Process multiple WAV files through the pipeline.
        
        Args:
            wav_files: List of WAV file paths
            run_mfa: Whether to run MFA alignment
            text_content: Text content for MFA alignment
        
        Returns:
            List of processing results for each file
        """
        all_results = []
        
        # Check MFA availability
        mfa_available = self.check_mfa_available() if run_mfa else False
        
        if run_mfa and not mfa_available:
            self.logger.warning("MFA not available. Skipping MFA alignment.")
            run_mfa = False
        
        # Run MFA alignment if requested
        textgrid_paths = {}
        if run_mfa and mfa_available:
            with tempfile.TemporaryDirectory() as tmpdir:
                corpus_dir = Path(tmpdir) / "corpus"
                mfa_output_dir = Path(tmpdir) / "output"
                
                # Create corpus structure
                if self.create_corpus_structure(wav_files, corpus_dir, text_content):
                    # Run MFA alignment
                    if self.run_mfa_alignment(corpus_dir, mfa_output_dir):
                        # Copy TextGrid files to output directory
                        for wav_file in wav_files:
                            tg_file = mfa_output_dir / f"{wav_file.stem}.TextGrid"
                            if tg_file.exists():
                                dest_tg = self.output_dir / f"{wav_file.stem}.TextGrid"
                                shutil.copy2(tg_file, dest_tg)
                                textgrid_paths[wav_file] = dest_tg
                                self.logger.info(f"Copied TextGrid: {dest_tg}")
                            else:
                                self.logger.warning(f"TextGrid not generated for {wav_file.name}")
                    else:
                        self.logger.error("MFA alignment failed")
                else:
                    self.logger.error("Failed to create corpus structure")
        
        # Process each WAV file
        for wav_file in wav_files:
            try:
                textgrid_path = textgrid_paths.get(wav_file)
                result = self.process_wav_file(
                    wav_file,
                    textgrid_path=textgrid_path,
                    skip_mfa=not run_mfa
                )
                all_results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {wav_file.name}: {e}")
                all_results.append({
                    'file': wav_file.name,
                    'success': False,
                    'error': str(e)
                })
        
        # Create summary report
        self.create_summary_report(all_results)
        
        return all_results
    
    def create_summary_report(self, all_results: list[dict]) -> None:
        """Create a summary report of all processing results"""
        summary_path = self.output_dir / "processing_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MFA Onset Detection Pipeline - Processing Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Total files processed: {len(all_results)}\n\n")
            
            f.write("MFA Parameters:\n")
            f.write(f"  high_freq_min: {self.mfa_params.high_freq_min} Hz\n")
            f.write(f"  frame_length_ms: {self.mfa_params.frame_length_ms} ms\n")
            f.write(f"  hop_length_ms: {self.mfa_params.hop_length_ms} ms\n")
            f.write(f"  diff_threshold_std: {self.mfa_params.diff_threshold_std}\n\n")
            
            f.write("Hilbert Parameters:\n")
            f.write(f"  target_sr: {self.hilbert_params.target_sr} Hz\n")
            f.write(f"  hp_cutoff: {self.hilbert_params.hp_cutoff} Hz\n")
            f.write(f"  threshold_ratio: {self.hilbert_params.threshold_ratio}\n")
            f.write(f"  lookback_points: {self.hilbert_params.lookback_points}\n")
            f.write(f"  min_interval_ms: {self.hilbert_params.min_interval_ms} ms\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("File-by-File Results:\n")
            f.write("-" * 80 + "\n\n")
            
            for result in all_results:
                f.write(f"File: {result['file']}\n")
                f.write(f"  Success: {result['success']}\n")
                
                if result.get('error'):
                    f.write(f"  Error: {result['error']}\n")
                
                if result.get('mfa_result'):
                    mfa = result['mfa_result']
                    f.write(f"  MFA Onsets: {len(mfa.onset_times)}\n")
                    if mfa.error:
                        f.write(f"  MFA Error: {mfa.error}\n")
                
                if result.get('hilbert_result'):
                    hilbert = result['hilbert_result']
                    f.write(f"  Hilbert Onsets: {len(hilbert.onset_times)}\n")
                    if hilbert.peak_times:
                        f.write(f"  Hilbert Peaks: {len(hilbert.peak_times)}\n")
                    if hilbert.error:
                        f.write(f"  Hilbert Error: {hilbert.error}\n")
                
                if result.get('plot_path'):
                    f.write(f"  Plot: {result['plot_path']}\n")
                if result.get('csv_path'):
                    f.write(f"  CSV: {result['csv_path']}\n")
                if result.get('json_path'):
                    f.write(f"  JSON: {result['json_path']}\n")
                
                f.write("\n")
        
        self.logger.info(f"Created summary report: {summary_path}")


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(
        description="MFA-based Onset Detection Pipeline - Automatically detects /t/ bursts from singing audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Process single file (MFA runs automatically)
  python mfa_onset_pipeline.py file1.wav
  
  # Process multiple files (MFA runs automatically)
  python mfa_onset_pipeline.py file1.wav file2.wav file3.wav
  
  # Process without MFA (if TextGrid files already exist)
  python mfa_onset_pipeline.py file1.wav --no-mfa
  
  # Process with custom parameters
  python mfa_onset_pipeline.py file.wav --mfa-high-freq 2500 --hilbert-hp-cutoff 600
  
  # Process with custom output directory
  python mfa_onset_pipeline.py file.wav -o my_results/
        """
    )
    
    # Input files
    parser.add_argument(
        'wav_files',
        nargs='+',
        type=Path,
        help='WAV files containing singing audio to process'
    )
    
    # MFA options
    parser.add_argument(
        '--no-mfa',
        action='store_true',
        help='Skip MFA alignment (use if TextGrid files already exist in output directory)'
    )
    parser.add_argument(
        '--mfa-model',
        default='english_us_arpa',
        help='MFA acoustic model (default: english_us_arpa)'
    )
    parser.add_argument(
        '--mfa-dictionary',
        default='english_us_arpa',
        help='MFA dictionary (default: english_us_arpa)'
    )
    parser.add_argument(
        '--text-content',
        default='ta',
        help='Text content for MFA alignment - automatically set for /t/ burst detection (default: ta)'
    )
    
    # MFA detection parameters
    parser.add_argument(
        '--mfa-high-freq',
        type=float,
        default=2000.0,
        help='MFA high-frequency minimum in Hz (default: 2000)'
    )
    parser.add_argument(
        '--mfa-frame-length',
        type=float,
        default=5.0,
        help='MFA frame length in ms (default: 5.0)'
    )
    parser.add_argument(
        '--mfa-hop-length',
        type=float,
        default=1.0,
        help='MFA hop length in ms (default: 1.0)'
    )
    parser.add_argument(
        '--mfa-threshold',
        type=float,
        default=2.0,
        help='MFA diff threshold std (default: 2.0)'
    )
    
    # Hilbert detection parameters
    parser.add_argument(
        '--hilbert-sr',
        type=int,
        default=48000,
        help='Hilbert target sampling rate in Hz (default: 48000)'
    )
    parser.add_argument(
        '--hilbert-hp-cutoff',
        type=float,
        default=500.0,
        help='Hilbert high-pass cutoff in Hz (default: 500)'
    )
    parser.add_argument(
        '--hilbert-threshold',
        type=float,
        default=0.1,
        help='Hilbert threshold ratio (default: 0.1)'
    )
    parser.add_argument(
        '--hilbert-lookback',
        type=int,
        default=74,
        help='Hilbert lookback points (default: 74)'
    )
    parser.add_argument(
        '--hilbert-min-interval',
        type=float,
        default=50.0,
        help='Hilbert minimum interval in ms (default: 50.0)'
    )
    
    # Output options
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=Path('./mfa_onset_results'),
        help='Output directory (default: ./mfa_onset_results)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate WAV files
    wav_files = []
    for wav_file in args.wav_files:
        if not wav_file.exists():
            print(f"Error: File not found: {wav_file}", file=sys.stderr)
            sys.exit(1)
        if wav_file.suffix.lower() not in ['.wav', '.wave']:
            print(f"Warning: {wav_file} may not be a WAV file", file=sys.stderr)
        wav_files.append(wav_file)
    
    # Setup parameters
    mfa_params = MFAParameters(
        high_freq_min=args.mfa_high_freq,
        frame_length_ms=args.mfa_frame_length,
        hop_length_ms=args.mfa_hop_length,
        diff_threshold_std=args.mfa_threshold
    )
    
    hilbert_params = HilbertParameters(
        target_sr=args.hilbert_sr,
        hp_cutoff=args.hilbert_hp_cutoff,
        threshold_ratio=args.hilbert_threshold,
        lookback_points=args.hilbert_lookback,
        min_interval_ms=args.hilbert_min_interval
    )
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Create pipeline
    pipeline = MFAOnsetPipeline(
        mfa_params=mfa_params,
        hilbert_params=hilbert_params,
        mfa_model=args.mfa_model,
        mfa_dictionary=args.mfa_dictionary,
        output_dir=args.output_dir,
        log_level=log_level
    )
    
    # Process files
    print(f"Processing {len(wav_files)} file(s)...")
    # By default, run MFA unless --no-mfa is specified
    run_mfa = not args.no_mfa
    results = pipeline.process_multiple_files(
        wav_files,
        run_mfa=run_mfa,
        text_content=args.text_content
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print(f"Processed {len(results)} file(s)")
    
    success_count = sum(1 for r in results if r['success'])
    print(f"Successful: {success_count}/{len(results)}")
    
    if success_count < len(results):
        print("\nFiles with errors:")
        for r in results:
            if not r['success']:
                print(f"  - {r['file']}: {r.get('error', 'Unknown error')}")
    
    print("\nCheck the summary report for details:")
    print(f"  {args.output_dir / 'processing_summary.txt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
