#!/usr/bin/env python3
"""
Example usage of the Repressilator analysis package.

This script demonstrates how to use the package to analyze
fluorescence microscopy images.
"""

from repressilator_analysis import pipeline
from repressilator_analysis import image_loader
from repressilator_analysis import calibration
import matplotlib.pyplot as plt


def main():
    """Run the analysis pipeline with custom parameters."""

    print("Example 1: Running the full pipeline")
    print("-" * 50)

    # Run the complete analysis pipeline
    results = pipeline.run_analysis(
        intensity_dir="images/intensity",
        phase_dir="images/phase",
        docs_dir="docs",
        output_dir="results",
        nuclear_channel="green",       # Color channel for nuclear fluorescence
        cytoplasmic_channel="red",     # Color channel for cytoplasmic fluorescence
        min_cell_area=50,              # Minimum cell size in pixels
        n_mcmc_iterations=1000,        # MCMC iterations (increase for better convergence)
        save_intermediate=True,        # Save intermediate results
    )

    print("\nExample 2: Using individual modules")
    print("-" * 50)

    # Load images manually
    print("Loading images...")
    timepoints, intensity_images, phase_images = image_loader.load_timeseries(
        "images/intensity",
        "images/phase"
    )
    print(f"Loaded {len(timepoints)} timepoints")

    # Load calibration data
    print("Loading calibrations...")
    calibrations = calibration.load_calibrations("docs")
    for name, calib in calibrations.items():
        print(f"  {name}: {calib.protein_name}")

    # Example: Convert a single fluorescence value to mass
    example_pixel_value = 100.0
    if 'nuclear' in calibrations:
        mass_ng = calibrations['nuclear'].pixel_intensity_to_mass(example_pixel_value)
        molecules = calibrations['nuclear'].pixel_intensity_to_molecules(
            example_pixel_value, 66  # molecular weight in kDa
        )
        print(f"\nExample conversion for pixel value {example_pixel_value}:")
        print(f"  Mass: {mass_ng:.2f} ng")
        print(f"  Molecules: {molecules:.2e}")

    # Visualize calibration curves
    print("\nGenerating calibration curve plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (name, calib) in enumerate(calibrations.items()):
        ax = axes[idx]
        ax.plot(calib.fluorescence_au, calib.mass_ng, 'o-', label='Calibration data')
        ax.set_xlabel('Fluorescence (A.U.)')
        ax.set_ylabel('Mass (ng)')
        ax.set_title(f'{name.capitalize()} Protein Calibration')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig("results/calibration_curves.png", dpi=150)
    print("  Saved to results/calibration_curves.png")
    plt.close()

    print("\nExample 3: Quick analysis summary")
    print("-" * 50)
    print(f"Total timepoints analyzed: {len(results['timepoints'])}")
    print(f"Time range: {results['timepoints'][0]:.0f} - {results['timepoints'][-1]:.0f} minutes")
    print(f"Total cell observations: {len(results['all_cell_data'])}")
    print(f"Tracked cells: {len(results['tracks'])}")
    print(f"Cells with inferred parameters: {len(results['inference_results'])}")

    if len(results['inference_results']) > 0:
        print("\nExample parameter estimates (first cell):")
        first_result = list(results['inference_results'].values())[0]
        param_names = first_result['parameter_names']
        param_means = first_result['parameter_means']

        for name, value in zip(param_names, param_means):
            print(f"  {name:10s}: {value:.4f}")

    print("\nAnalysis complete! Check the 'results/' directory for outputs.")


if __name__ == "__main__":
    main()
