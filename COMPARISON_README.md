# Envelope Comparison Framework

This document provides detailed information about the envelope comparison framework for onset detection, including metrics explanation, parameter effects, and usage guidelines.

## Overview

The envelope comparison framework allows systematic evaluation of different envelope computation methods and detection parameters. This is useful for:

- **Method Selection**: Choose the best envelope type and parameters for your specific audio
- **Parameter Tuning**: Understand how each parameter affects detection results
- **Reproducible Research**: Document and replicate onset detection configurations
- **Quality Assessment**: Compare detection accuracy across different configurations

## Core Concepts

### Fujii-Style Detection Method

The framework implements the Fujii-style onset detection method as the core algorithm:

1. **Envelope Computation**: Extract amplitude envelope from audio signal
2. **Peak Detection**: Find prominent peaks in the envelope (potential events)
3. **Onset Localization**: For each peak, search backward to find when amplitude first exceeds 10% of peak value
4. **Threshold Crossing**: Use linear interpolation for sub-sample precision

This method is based on perceptual principles: onsets are defined relative to local peak amplitude rather than absolute thresholds.

### Envelope Variants

#### Hilbert Envelope

**Method**: `compute_hilbert_envelope_variant()`

- Computes instantaneous amplitude using Hilbert transform: `E(t) = |hilbert(signal)|`
- Provides sample-level temporal resolution
- Optional smoothing with moving average window

**When to use**:
- Fast transients (clicks, taps)
- Need precise temporal resolution
- Prefer mathematical rigor

**Parameters**:
- `hpf_cutoff`: High-pass filter cutoff (Hz). Removes low-frequency content.
- `smooth_ms`: Smoothing window (ms). Reduces noise but decreases temporal precision.

#### RMS Envelope

**Method**: `compute_rms_envelope_variant()`

- Computes root-mean-square amplitude over short frames
- Frame-based computation (typically 5ms frames, 1ms hop)
- More robust to noise than Hilbert for some signals

**When to use**:
- Noisy signals
- Longer, sustained events
- Need noise robustness

**Parameters**:
- `hpf_cutoff`: High-pass filter cutoff (Hz)
- `frame_ms`: Analysis frame length (ms). Larger = smoother but less precise.
- `hop_ms`: Hop size between frames (ms). Smaller = finer temporal resolution.

## Metrics Explained

Each variant produces the following metrics:

### Event Count Metrics

**`n_events`**: Total number of detected onsets

- Higher values may indicate false positives or closely-spaced events
- Lower values may indicate missed events or over-filtering

### Temporal Metrics

**`mean_onset_to_peak_ms`**: Average time from onset to peak (milliseconds)

- Measure of event "attack time"
- Influenced by signal characteristics and envelope smoothing
- Typical range: 0.1-5 ms for percussive sounds

**`rise_time_10_90_ms`**: Average rise time from 10% to 90% of peak amplitude (milliseconds)

- Standard measure of transient sharpness
- Shorter = sharper transient
- Affected by smoothing and signal characteristics
- Typical range: 0.1-10 ms

**`median_rise_time_ms`**: Median rise time (milliseconds)

- More robust to outliers than mean
- Useful when events have varying characteristics

### Slope Metrics

**`slope_10_90`**: Normalized amplitude slope (1/seconds)

- Measures how quickly amplitude rises relative to peak
- Formula: `(A_peak - A_onset) / (t_peak - t_onset) / A_peak`
- Higher values = steeper rise
- Normalized by peak amplitude for comparability across events

## Parameters and Their Effects

### High-Pass Filter Cutoff (`hpf_cutoff`)

**Purpose**: Remove low-frequency content that may interfere with detection

| Value | Effect | Use Case |
|-------|--------|----------|
| 0 or None | No filtering | Full-band signals, known clean input |
| 300 Hz | Mild filtering | Finger taps, low-frequency rejection |
| 500 Hz | Moderate filtering | General percussive sounds |
| 1000 Hz | Strong filtering | Metronome clicks, high-frequency transients |

**Trade-offs**:
- ↑ cutoff → Removes more low-frequency content, may miss low-pitched events
- ↓ cutoff → Retains more signal content, may introduce low-frequency artifacts

### Smoothing Window (`smooth_ms`, Hilbert only)

**Purpose**: Reduce envelope noise and stabilize detection

| Value | Effect | Use Case |
|-------|--------|----------|
| 0 or None | No smoothing | Need maximum temporal precision |
| 0.1-0.5 ms | Light smoothing | Balance precision and stability |
| 0.5-1.0 ms | Moderate smoothing | Noisy signals, reduce false positives |
| 1.0+ ms | Heavy smoothing | Very noisy signals, smooth events |

**Trade-offs**:
- ↑ smoothing → Reduces false positives, increases rise time, decreases temporal precision
- ↓ smoothing → Sharper onsets, more false positives from noise

### Global Min Height Ratio (`global_min_height_ratio`)

**Purpose**: Filter out small peaks relative to overall signal amplitude

| Value | Effect | Use Case |
|-------|--------|----------|
| 0.0 | No global filtering | Detect all events regardless of amplitude |
| 0.05-0.1 | Light filtering | Remove tiny artifacts |
| 0.2-0.3 | Moderate filtering | Keep only significant events |
| 0.5+ | Strong filtering | Only loudest events |

**Trade-offs**:
- ↑ ratio → Fewer detections, removes minor events, reduces false positives
- ↓ ratio → More detections, includes quieter events, may include noise

### Minimum Distance (`min_distance_ms`)

**Purpose**: Prevent multiple detections for same event

| Value | Effect | Use Case |
|-------|--------|----------|
| 20-50 ms | Very short | Fast drumming, rapid taps |
| 50-100 ms | Short | Normal tapping, speech syllables |
| 100-200 ms | Moderate | Slower rhythms, distinct events |
| 200+ ms | Long | Well-separated events |

**Trade-offs**:
- ↑ distance → Merges closely-spaced events, prevents spurious double-detections
- ↓ distance → May split single events, allows closer event spacing

### Lookback Points (`lookback_points`)

**Purpose**: Enforce quiet baseline before onset (reduces false positives)

| Value | Effect | Use Case |
|-------|--------|----------|
| 0 | Disabled | Standard Fujii method |
| 74 (~2ms @ 48kHz) | Standard | Typical quiet-period enforcement |
| 150+ | Extended | Very strict baseline requirement |

**Trade-offs**:
- ↑ lookback → Stricter onset criterion, fewer false positives, may miss soft onsets
- 0 (disabled) → More permissive, detects all threshold crossings

## Example Usage

### Basic Comparison

Compare Hilbert and RMS envelopes with default parameters:

```bash
python compare_envelopes.py --wav audio.wav --export_plots
```

### Parameter Sweep: HPF Cutoffs

Test different high-pass filter cutoffs:

```bash
python compare_envelopes.py --wav audio.wav \
    --hpf_cutoffs 0,300,500,1000 \
    --export_plots --out_dir results
```

### Parameter Sweep: Smoothing Windows

Compare smoothing effects:

```bash
python compare_envelopes.py --wav audio.wav \
    --smooth_ms 0,0.1,0.5,1.0 \
    --hpf_cutoffs 300 \
    --export_plots --out_dir results
```

### Comprehensive Grid Search

Test multiple parameter combinations:

```bash
python compare_envelopes.py --wav audio.wav \
    --hpf_cutoffs 0,300,500 \
    --smooth_ms 0,0.5 \
    --global_min_height_ratios 0,0.1,0.2 \
    --min_distance_ms 50,100 \
    --lookback_points 0,74 \
    --export_plots --out_dir results \
    --label tap
```

This generates 2×2×3×2×2 = 48 Hilbert variants + 24 RMS variants = 72 total configurations.

### Audio Type-Specific Recommendations

**Metronome Clicks**:
```bash
python compare_envelopes.py --wav clicks.wav \
    --hpf_cutoffs 1000 \
    --smooth_ms 0,0.5 \
    --min_distance_ms 100 \
    --label click
```

**Finger Taps**:
```bash
python compare_envelopes.py --wav taps.wav \
    --hpf_cutoffs 300,500 \
    --smooth_ms 0,0.5 \
    --lookback_points 0,74 \
    --min_distance_ms 100 \
    --label tap
```

**Speech /t/ Bursts**:
```bash
python compare_envelopes.py --wav speech.wav \
    --hpf_cutoffs 500,1000,2000 \
    --smooth_ms 0 \
    --min_distance_ms 50 \
    --label burst
```

## Interpreting Results

### CSV Output

The metrics CSV contains one row per variant with all parameters and computed metrics.

**Key columns**:
- `variant_id`: Unique identifier encoding all parameters
- `n_events`: Number of detected onsets
- `mean_onset_to_peak_ms`: Average onset-to-peak time
- `rise_time_10_90_ms`: Average 10-90% rise time
- `slope_10_90`: Average normalized slope

**Metadata** (in CSV header comments):
- Git commit hash for reproducibility
- Total number of variants tested

### Plots

**Top panel**: Waveform with onset markers (vertical lines) for each variant

**Bottom panel**: Overlaid envelopes with onset markers (scatter points)

**Color coding**: Each variant has unique color (colorblind-friendly palette)

**Legend**: Shows variant ID for each color

### Selecting Best Configuration

Consider these factors:

1. **Detection Count**: Does `n_events` match expected number?
2. **Temporal Precision**: Are onsets placed at perceptually correct locations?
3. **Consistency**: Do similar events have similar metrics?
4. **False Positives**: Are there spurious detections in quiet regions?
5. **False Negatives**: Are real events missed?

**Trade-off principle**: There's no universally "best" configuration. Balance:
- **Sensitivity** (detect all events) vs. **Specificity** (avoid false positives)
- **Temporal precision** vs. **Robustness to noise**
- **Generalizability** vs. **Signal-specific optimization**

## Comparison with Standard Methods

### vs. Librosa `onset_detect()`

- **Librosa**: Uses spectral flux or other frequency-domain features
- **This framework**: Uses time-domain amplitude envelope (Fujii method)
- **Advantage**: Simpler, more interpretable, better for percussive sounds
- **Limitation**: May miss tonal onsets without sharp amplitude rise

### vs. Energy-Based Threshold

- **Simple threshold**: Fixed or adaptive amplitude threshold
- **Fujii method**: Relative to local peak (10% rule)
- **Advantage**: Automatically adapts to varying event amplitudes
- **Use case**: Signals with varying loudness levels

## Performance Considerations

### Caching

The script caches high-pass filtered signals:
- Each HPF cutoff is computed once
- Reused across multiple smoothing/detection parameter combinations
- Significantly speeds up large parameter sweeps

### Memory Usage

For large parameter sweeps:
- Each variant stores full envelope (same length as audio)
- Consider limiting `max_variants_to_plot` for long recordings
- CSV metrics are compact (one row per variant)

## Reproducibility

### Git Commit Hash

Every CSV output includes the git commit hash:
```
# Git commit: a1b2c3d
```

This allows exact reproduction of results using the same code version.

### Random Seed

For test generation with synthetic audio, set random seed:
```python
np.random.seed(42)
```

## Future Extensions (TODO)

See inline `TODO` comments in code for planned features:

1. **Reference Onset Import**: Compare against ground truth CSV
   - Compute precision, recall, F1-score
   - Calculate onset time errors (mean, std)

2. **JSON Config Input**: Support large grid sweeps via JSON file
   ```json
   {
     "hpf_cutoffs": [0, 300, 500, 1000],
     "smooth_ms": [0, 0.1, 0.5, 1.0],
     ...
   }
   ```

3. **Batch Processing**: Process directory of files
   - Generate summary statistics across dataset
   - Identify best parameters for entire corpus

4. **Statistical Summary**: Mean differences between variants
   - ANOVA or t-tests for significance
   - Confidence intervals for metrics

## References

- **Fujii Method**: Based on perceptual onset detection principles
- **Hilbert Transform**: Signal processing standard for envelope extraction
- **RMS Envelope**: Common audio feature in music information retrieval

## Support

For questions or issues:
1. Check this documentation
2. Review example commands above
3. Inspect output CSV and plots
4. Consult main README.md for package overview
