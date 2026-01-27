"""
Calibration utilities for converting fluorescence intensities to protein quantities.

This module provides functions to load calibration data and convert pixel
intensities to protein mass (nanograms) or molecule counts.
"""

import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Dict, Tuple
from scipy import stats
class ProteinCalibration:
    """
    Calibration curve for converting fluorescence to protein quantity.
    """
    PIXEL_TO_AU_FACTOR = 1e7
    def __init__(self, filename, weight,header=0):
        """
        Initialize calibration curve.

        Args:
            mass_ng: Protein mass in nanograms
            fluorescence_au: Fluorescence in arbitrary units
            protein_name: Name of the protein
        """
        self.weight=weight
        self.avogadro=6.022*10**23
        with open(filename, "r") as f:
            data=np.loadtxt(f, skiprows=header)
        mass_ng=data[:,0]
        fluorescence_values=data[:,1:]
        self.slope_real, self.intercept_real, r_value_real, p_value, std_err = stats.linregress(np.repeat(mass_ng,3), fluorescence_values.flatten())
    def pixel_intensities_to_molecules(
        self,
        pixel_array: float,
    ) -> float:
        """
        Convert pixel intensity to number of molecules.

        Args:
            pixel_intensity: Mean pixel intensity from image
            molecular_weight_kda: Molecular weight in kDa

        Returns:
            Number of molecules
        """
        weight_in_nanograms=(pixel_array-self.intercept_real)/self.slope_real
        moles=(weight_in_nanograms*1e-9/self.PIXEL_TO_AU_FACTOR) / (self.weight * 1000) 
        molecules=moles* self.avogadro
        molecules[np.where(molecules<0)]=0
        return molecules


