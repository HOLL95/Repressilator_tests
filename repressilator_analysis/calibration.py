"""
Calibration utilities for converting fluorescence intensities to protein quantities.

This module provides functions to load calibration data and convert pixel
intensities to protein mass (nanograms) or molecule counts.
"""

import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Dict, Tuple


# Conversion factor between pixel intensities and arbitrary units in calibration
PIXEL_TO_AU_FACTOR = 1e7


class ProteinCalibration:
    """
    Calibration curve for converting fluorescence to protein quantity.
    """

    def __init__(self, mass_ng: np.ndarray, fluorescence_au: np.ndarray, protein_name: str):
        """
        Initialize calibration curve.

        Args:
            mass_ng: Protein mass in nanograms
            fluorescence_au: Fluorescence in arbitrary units
            protein_name: Name of the protein
        """
        self.mass_ng = mass_ng
        self.fluorescence_au = fluorescence_au
        self.protein_name = protein_name

        # Create interpolation function (fluorescence -> mass)
        self.interp_func = interp1d(
            fluorescence_au,
            mass_ng,
            kind='linear',
            fill_value='extrapolate'
        )

    def fluorescence_to_mass(self, fluorescence_au: float) -> float:
        """
        Convert fluorescence (arbitrary units) to protein mass (nanograms).

        Args:
            fluorescence_au: Fluorescence in arbitrary units

        Returns:
            Protein mass in nanograms
        """
        return float(self.interp_func(fluorescence_au))

    def pixel_intensity_to_mass(self, pixel_intensity: float) -> float:
        """
        Convert pixel intensity to protein mass (nanograms).

        Args:
            pixel_intensity: Mean pixel intensity from image

        Returns:
            Protein mass in nanograms
        """
        fluorescence_au = pixel_intensity * PIXEL_TO_AU_FACTOR
        return self.fluorescence_to_mass(fluorescence_au)

    def pixel_intensity_to_molecules(
        self,
        pixel_intensity: float,
        molecular_weight_kda: float,
    ) -> float:
        """
        Convert pixel intensity to number of molecules.

        Args:
            pixel_intensity: Mean pixel intensity from image
            molecular_weight_kda: Molecular weight in kDa

        Returns:
            Number of molecules
        """
        mass_ng = self.pixel_intensity_to_mass(pixel_intensity)

        # Convert ng to grams
        mass_g = mass_ng * 1e-9

        # Convert kDa to g/mol
        molecular_weight_g_per_mol = molecular_weight_kda * 1000

        # Calculate moles
        moles = mass_g / molecular_weight_g_per_mol

        # Convert to molecules (Avogadro's number)
        avogadro = 6.02214076e23
        molecules = moles * avogadro

        return molecules


def load_calibration_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load calibration data from a text file.

    Expected format:
        Mass/ng repeat 1/A.U. repeat 2/A.U. repeat 3/A.U.
        2.500e+01 5.969e+02 6.228e+02 8.278e+02
        ...

    Args:
        filepath: Path to calibration file

    Returns:
        Tuple of (mass_ng, mean_fluorescence_au)
    """
    data = np.loadtxt(filepath, skiprows=1)

    mass_ng = data[:, 0]
    repeats = data[:, 1:]

    # Average across repeats
    mean_fluorescence_au = np.mean(repeats, axis=1)

    return mass_ng, mean_fluorescence_au


def load_calibrations(docs_dir: str = "docs") -> Dict[str, ProteinCalibration]:
    """
    Load all calibration curves from the docs directory.

    Args:
        docs_dir: Directory containing calibration files

    Returns:
        Dictionary mapping protein name to ProteinCalibration object
    """
    docs_path = Path(docs_dir)

    calibrations = {}

    # Load nuclear repressor calibration
    nuclear_file = docs_path / "Nuclear repressor 1 (66 kDa) calibration.txt"
    if nuclear_file.exists():
        mass, fluor = load_calibration_file(str(nuclear_file))
        calibrations['nuclear'] = ProteinCalibration(
            mass, fluor, "Nuclear repressor 1 (66 kDa)"
        )

    # Load cytosolic repressor calibration
    cyto_file = docs_path / "Cytosolic repressor (53 kDa) calibration.txt"
    if cyto_file.exists():
        mass, fluor = load_calibration_file(str(cyto_file))
        calibrations['cytoplasmic'] = ProteinCalibration(
            mass, fluor, "Cytosolic repressor (53 kDa)"
        )

    return calibrations


def convert_cell_fluorescence_to_mass(
    cell_fluorescence: Dict[str, float],
    calibrations: Dict[str, ProteinCalibration],
) -> Dict[str, float]:
    """
    Convert cell fluorescence values to protein masses.

    Args:
        cell_fluorescence: Dict with keys like 'nuclear', 'cytoplasmic' -> pixel intensity
        calibrations: Dict mapping protein type to ProteinCalibration

    Returns:
        Dict with same keys -> protein mass in nanograms
    """
    results = {}

    for protein_type, pixel_intensity in cell_fluorescence.items():
        if protein_type in calibrations:
            calibration = calibrations[protein_type]
            mass_ng = calibration.pixel_intensity_to_mass(pixel_intensity)
            results[protein_type] = mass_ng
        else:
            results[protein_type] = np.nan

    return results


def convert_cell_fluorescence_to_molecules(
    cell_fluorescence: Dict[str, float],
    calibrations: Dict[str, ProteinCalibration],
    molecular_weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Convert cell fluorescence values to molecule counts.

    Args:
        cell_fluorescence: Dict with keys like 'nuclear', 'cytoplasmic' -> pixel intensity
        calibrations: Dict mapping protein type to ProteinCalibration
        molecular_weights: Dict mapping protein type to molecular weight in kDa

    Returns:
        Dict with same keys -> number of molecules
    """
    results = {}

    for protein_type, pixel_intensity in cell_fluorescence.items():
        if protein_type in calibrations and protein_type in molecular_weights:
            calibration = calibrations[protein_type]
            mw = molecular_weights[protein_type]
            molecules = calibration.pixel_intensity_to_molecules(pixel_intensity, mw)
            results[protein_type] = molecules
        else:
            results[protein_type] = np.nan

    return results
