"""
Tests for MFA onset detection pipeline.

This test suite validates the MFA-based onset detection pipeline including:
- Basic pipeline functionality
- Detection result formats
- Plot generation
- CSV/JSON export
- Error handling

Target Python version: 3.10+
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from mfa_onset_pipeline import (
    MFAOnsetPipeline,
    MFAParameters,
    HilbertParameters,
    DetectionResult,
)


def create_test_audio(duration: float = 2.0, sr: int = 48000, click_times: list[float] = None):
    """Create a simple test audio file with clicks."""
    if click_times is None:
        click_times = [0.3, 0.8, 1.5]
    
    t = np.linspace(0, duration, int(sr * duration))
    y = np.zeros_like(t)
    
    for click_t in click_times:
        idx = int(click_t * sr)
        burst_len = int(0.02 * sr)  # 20ms burst
        y[idx:idx+burst_len] = np.random.randn(burst_len) * 0.5
    
    return y, sr


def test_pipeline_initialization():
    """Test that pipeline initializes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = MFAOnsetPipeline(output_dir=Path(tmpdir))
        assert pipeline.output_dir.exists()
        assert isinstance(pipeline.mfa_params, MFAParameters)
        assert isinstance(pipeline.hilbert_params, HilbertParameters)
        print("✓ Pipeline initialization test passed")


def test_hilbert_detection():
    """Test Hilbert-based onset detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test audio
        y, sr = create_test_audio()
        wav_path = tmpdir / "test.wav"
        sf.write(wav_path, y, sr)
        
        # Initialize pipeline
        pipeline = MFAOnsetPipeline(output_dir=tmpdir / "results")
        
        # Run Hilbert detection
        result = pipeline.detect_hilbert_onsets(wav_path)
        
        # Verify result
        assert result.method == "Hilbert"
        assert len(result.onset_times) > 0
        assert result.peak_times is not None
        assert len(result.peak_times) == len(result.onset_times)
        assert result.error is None
        
        # Verify timing is reasonable
        for onset_t in result.onset_times:
            assert 0 <= onset_t <= 2.0  # Within audio duration
        
        print(f"✓ Hilbert detection test passed ({len(result.onset_times)} onsets detected)")


def test_plot_generation():
    """Test comparison plot generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test audio
        y, sr = create_test_audio()
        wav_path = tmpdir / "test.wav"
        sf.write(wav_path, y, sr)
        
        # Initialize pipeline
        pipeline = MFAOnsetPipeline(output_dir=tmpdir / "results")
        
        # Create mock results
        hilbert_result = pipeline.detect_hilbert_onsets(wav_path)
        mfa_result = DetectionResult(method="MFA", onset_times=[0.35, 0.85, 1.55])
        
        # Generate plot
        plot_path = pipeline.plot_comparison(wav_path, mfa_result, hilbert_result)
        
        # Verify plot was created
        assert plot_path is not None
        assert plot_path.exists()
        assert plot_path.suffix == ".png"
        
        print("✓ Plot generation test passed")


def test_csv_export():
    """Test CSV export functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test audio
        y, sr = create_test_audio()
        wav_path = tmpdir / "test.wav"
        sf.write(wav_path, y, sr)
        
        # Initialize pipeline
        pipeline = MFAOnsetPipeline(output_dir=tmpdir / "results")
        
        # Create mock results
        hilbert_result = DetectionResult(
            method="Hilbert",
            onset_times=[0.3, 0.8, 1.5],
            peak_times=[0.31, 0.81, 1.51]
        )
        mfa_result = DetectionResult(
            method="MFA",
            onset_times=[0.35, 0.85, 1.55]
        )
        
        # Export results
        csv_path, json_path = pipeline.export_results(wav_path, mfa_result, hilbert_result)
        
        # Verify CSV was created
        assert csv_path is not None
        assert csv_path.exists()
        
        # Verify CSV contents
        df = pd.read_csv(csv_path)
        assert len(df) == 6  # 3 MFA + 3 Hilbert
        assert set(df.columns) == {'method', 'onset_time_s', 'peak_time_s', 'file'}
        assert set(df['method'].unique()) == {'MFA', 'Hilbert'}
        
        # Verify JSON was created
        assert json_path is not None
        assert json_path.exists()
        
        # Verify JSON contents
        with open(json_path) as f:
            data = json.load(f)
        
        assert 'file' in data
        assert 'mfa_detection' in data
        assert 'hilbert_detection' in data
        assert data['mfa_detection']['count'] == 3
        assert data['hilbert_detection']['count'] == 3
        
        print("✓ CSV/JSON export test passed")


def test_complete_pipeline():
    """Test complete pipeline with single file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test audio
        y, sr = create_test_audio()
        wav_path = tmpdir / "test.wav"
        sf.write(wav_path, y, sr)
        
        # Initialize pipeline
        pipeline = MFAOnsetPipeline(output_dir=tmpdir / "results")
        
        # Process file (without MFA)
        result = pipeline.process_wav_file(wav_path, skip_mfa=True)
        
        # Verify result
        assert result['success'] is True
        assert result['file'] == "test.wav"
        assert result['hilbert_result'] is not None
        assert len(result['hilbert_result'].onset_times) > 0
        assert result['plot_path'] is not None
        assert result['plot_path'].exists()
        assert result['csv_path'] is not None
        assert result['csv_path'].exists()
        assert result['json_path'] is not None
        assert result['json_path'].exists()
        
        print("✓ Complete pipeline test passed")


def test_multiple_files():
    """Test pipeline with multiple files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create multiple test audio files
        wav_files = []
        for i in range(3):
            y, sr = create_test_audio()
            wav_path = tmpdir / f"test_{i}.wav"
            sf.write(wav_path, y, sr)
            wav_files.append(wav_path)
        
        # Initialize pipeline
        pipeline = MFAOnsetPipeline(output_dir=tmpdir / "results")
        
        # Process files
        results = pipeline.process_multiple_files(wav_files, run_mfa=False)
        
        # Verify results
        assert len(results) == 3
        for result in results:
            assert result['success'] is True
            assert result['plot_path'] is not None
            assert result['plot_path'].exists()
        
        # Verify summary report was created
        summary_path = tmpdir / "results" / "processing_summary.txt"
        assert summary_path.exists()
        
        print("✓ Multiple files test passed")


def test_error_handling():
    """Test error handling for invalid inputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create pipeline
        pipeline = MFAOnsetPipeline(output_dir=tmpdir / "results")
        
        # Test with non-existent file
        fake_path = tmpdir / "nonexistent.wav"
        result = pipeline.detect_hilbert_onsets(fake_path)
        
        # Should have error
        assert result.error is not None
        assert len(result.onset_times) == 0
        
        print("✓ Error handling test passed")


def test_custom_parameters():
    """Test pipeline with custom parameters."""
    custom_mfa = MFAParameters(
        high_freq_min=2500.0,
        frame_length_ms=10.0,
        hop_length_ms=2.0,
        diff_threshold_std=3.0
    )
    
    custom_hilbert = HilbertParameters(
        target_sr=44100,
        hp_cutoff=600.0,
        threshold_ratio=0.15,
        lookback_points=50,
        min_interval_ms=100.0
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = MFAOnsetPipeline(
            mfa_params=custom_mfa,
            hilbert_params=custom_hilbert,
            output_dir=Path(tmpdir)
        )
        
        # Verify parameters were set
        assert pipeline.mfa_params.high_freq_min == 2500.0
        assert pipeline.hilbert_params.hp_cutoff == 600.0
        
        print("✓ Custom parameters test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Running MFA Pipeline Tests")
    print("=" * 70)
    
    test_pipeline_initialization()
    test_hilbert_detection()
    test_plot_generation()
    test_csv_export()
    test_complete_pipeline()
    test_multiple_files()
    test_error_handling()
    test_custom_parameters()
    
    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
