# Repressilator Analysis

A Python package for analyzing time-series fluorescence microscopy images of bacterial cells expressing the Repressilator genetic circuit.

## Overview

The Repressilator is a synthetic genetic regulatory network consisting of three transcription repressors that form a cyclic negative feedback loop. This package provides tools to:

1. Load and process time-series fluorescence microscopy images
2. Extract fluorescence values for individual cells and proteins
3. Convert pixel intensities to protein quantities using calibration data
4. Infer ODE model parameters using Bayesian inference (PINTS)

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Package Structure

- `image_loader.py` - Load and sort time-series images
- `fluorescence_extraction.py` - Segment cells and extract fluorescence values
- `calibration.py` - Convert pixel values to protein quantities
- `ode_inference.py` - Parameter inference using PINTS
- `pipeline.py` - Main analysis pipeline script

## Usage

```python
from repressilator_analysis import pipeline

# Run the full analysis pipeline
results = pipeline.run_analysis(
    intensity_dir="images/intensity",
    phase_dir="images/phase",
    output_dir="results"
)
```

Or use the command-line interface:

```bash
python -m repressilator_analysis.pipeline
```

## Data Format

### Images
- **Intensity images**: RGB PNG files in `images/intensity/` with fluorescence data
- **Phase contrast images**: PNG files in `images/phase/` for cell segmentation
- Time-series naming: `sample_t+{time}m.png` (time in minutes)

### Calibration Data
- Located in `docs/` directory
- Two proteins with calibration curves: Nuclear repressor 1 (66 kDa) and Cytosolic repressor (53 kDa)
- Format: Mass (ng) vs fluorescence intensity (arbitrary units)
- Conversion factor: 1e7 between pixel intensities and calibration arbitrary units

## References

See `docs/35002125.pdf` for details on the Repressilator system.
