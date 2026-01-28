"""
Main analysis pipeline for Repressilator fluorescence microscopy data.

This script orchestrates the complete analysis workflow:
1. Load time-series images
2. Segment cells and extract fluorescence
3. Convert to protein quantities
4. Infer ODE parameters using PINTS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List
import pickle

from . import image_loader
from . import fluorescence_extraction
from . import calibration
from . import ode_inference

def extract_protein_numbers_from_tracks(
    tracks,
    intensity_dir: str = "images/intensity",
    phase_dir: str = "images/phase",
    weights: list=[],
    calibration_files=[],
    calibration_headers=0,
    track_labels=["n_intensity", "c_intensity"],
    output_dir: str = None,
    show_masks: bool = True,
):
    """
    Run the complete Repressilator analysis pipeline.

    Args:
        intensity_dir: Directory with fluorescence intensity images
        phase_dir: Directory with phase contrast images
        docs_dir: Directory with calibration files
        output_dir: Directory to save results
        nuclear_channel: Color channel for nuclear fluorescence
        cytoplasmic_channel: Color channel for cytoplasmic fluorescence
        min_cell_area: Minimum cell area in pixels
        n_mcmc_iterations: Number of MCMC iterations for parameter inference
        save_intermediate: Save intermediate results

    Returns:
        Dictionary containing all analysis results
    """

    # Create output directory
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    calibrators=[]
    for i in range(0, len(calibration_files)):
        calibrators.append(calibration.ProteinCalibration(calibration_files[i], weights[i], header=calibration_headers))
    # Step 1: Load images
    print("\n[1/5] Loading time-series images...")
    timepoints, intensity_images, phase_images = image_loader.load_timeseries(
        intensity_dir, phase_dir
    )
    get_all_cell_ids=[x["cell_id"] for x in tracks[0]]
    intensities=np.zeros((len(timepoints), len(tracks[0]), 2))
    molecule_numbers=np.zeros((len(timepoints), len(tracks[0]), 2))
    for i in range(0, len(tracks)):#Each row is a timepoint
        tracks[i]=fluorescence_extraction.extract_nuclear_cytoplasmic(
                intensity_images[i],
                tracks[i])
        
        for j in range(0, len(tracks[i])):#Each column is  a cell
            track_item=[tracks[i][x] for x in range(0, len(tracks[i])) if tracks[i][x]["cell_id"]==j][0]
            intensities[i,j,0]=track_item[track_labels[0]]
            intensities[i,j,1]=track_item[track_labels[1]]
    for m in range(0,2):
        for j in range(0, len(tracks[0])):#
            molecule_numbers[:,j, m]=calibrators[m].pixel_intensities_to_molecules(intensities[:,j,m])
    return molecule_numbers


        

    
    


   