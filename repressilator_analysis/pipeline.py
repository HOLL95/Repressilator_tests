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


def run_analysis(
    intensity_dir: str = "images/intensity",
    phase_dir: str = "images/phase",
    docs_dir: str = "docs",
    output_dir: str = "results",
    nuclear_channel: str = "green",
    cytoplasmic_channel: str = "red",
    min_cell_area: int = 50,
    n_mcmc_iterations: int = 1000,
    save_intermediate: bool = True,
) -> Dict:
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
    print("=" * 60)
    print("REPRESSILATOR ANALYSIS PIPELINE")
    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Step 1: Load images
    print("\n[1/5] Loading time-series images...")
    timepoints, intensity_images, phase_images = image_loader.load_timeseries(
        intensity_dir, phase_dir
    )
    print(f"  Loaded {len(timepoints)} timepoints")
    print(f"  Time range: {timepoints[0]:.0f} - {timepoints[-1]:.0f} minutes")

    # Step 2: Segment cells and track across time
    print("\n[2/5] Segmenting cells and tracking across timepoints...")
    tracks, labeled_images = fluorescence_extraction.track_cells_across_time(
        phase_images, min_cell_area
    )
    print(f"  Identified {len(tracks)} cell tracks")

    # Step 3: Extract fluorescence from segmented cells
    print("\n[3/5] Extracting fluorescence from tracked cells...")
    all_cell_data = []

    for t_idx, (t, intensity_img, labeled) in enumerate(
        zip(timepoints, intensity_images, labeled_images)
    ):
        # Extract fluorescence
        cell_fluorescence = fluorescence_extraction.extract_nuclear_cytoplasmic(
            intensity_img, labeled, nuclear_channel, cytoplasmic_channel
        )

        # Get cell IDs from labeled image
        cell_ids = np.unique(labeled)
        cell_ids = cell_ids[cell_ids > 0]

        # Store data
        for cell_id, fluor_data in zip(cell_ids, cell_fluorescence):
            all_cell_data.append({
                'timepoint': t,
                'timepoint_idx': t_idx,
                'cell_id': int(cell_id),
                'nuclear_pixels': fluor_data['nuclear'],
                'cytoplasmic_pixels': fluor_data['cytoplasmic'],
            })

    print(f"  Extracted fluorescence from {len(all_cell_data)} cell observations")

    if save_intermediate:
        df_cells = pd.DataFrame(all_cell_data)
        df_cells.to_csv(output_path / "raw_fluorescence.csv", index=False)
        print(f"  Saved raw fluorescence to {output_path / 'raw_fluorescence.csv'}")

    # Step 4: Convert fluorescence to protein quantities
    print("\n[4/5] Converting fluorescence to protein quantities...")
    calibrations = calibration.load_calibrations(docs_dir)
    print(f"  Loaded {len(calibrations)} calibration curves")

    # Molecular weights from calibration file names
    molecular_weights = {
        'nuclear': 66,  # kDa
        'cytoplasmic': 53,  # kDa
    }

    # Convert all measurements
    for data_point in all_cell_data:
        fluor_dict = {
            'nuclear': data_point['nuclear_pixels'],
            'cytoplasmic': data_point['cytoplasmic_pixels'],
        }

        # Convert to mass
        mass_dict = calibration.convert_cell_fluorescence_to_mass(fluor_dict, calibrations)
        data_point['nuclear_mass_ng'] = mass_dict.get('nuclear', np.nan)
        data_point['cytoplasmic_mass_ng'] = mass_dict.get('cytoplasmic', np.nan)

        # Convert to molecules
        mol_dict = calibration.convert_cell_fluorescence_to_molecules(
            fluor_dict, calibrations, molecular_weights
        )
        data_point['nuclear_molecules'] = mol_dict.get('nuclear', np.nan)
        data_point['cytoplasmic_molecules'] = mol_dict.get('cytoplasmic', np.nan)

    if save_intermediate:
        df_converted = pd.DataFrame(all_cell_data)
        df_converted.to_csv(output_path / "protein_quantities.csv", index=False)
        print(f"  Saved protein quantities to {output_path / 'protein_quantities.csv'}")

    # Step 5: Parameter inference for each cell track
    print("\n[5/5] Running parameter inference (this may take a while)...")
    inference_results = {}

    for track_id, track in tracks.items():
        if len(track) < 10:  # Skip short tracks
            continue

        print(f"\n  Processing track {track_id} ({len(track)} timepoints)...")

        # Collect data for this track
        track_times = []
        track_nuclear = []
        track_cytoplasmic = []

        for t_idx, cell_id in track:
            # Find corresponding data
            for data_point in all_cell_data:
                if (data_point['timepoint_idx'] == t_idx and
                    data_point['cell_id'] == cell_id):
                    track_times.append(data_point['timepoint'])
                    track_nuclear.append(data_point['nuclear_molecules'])
                    track_cytoplasmic.append(data_point['cytoplasmic_molecules'])
                    break

        # Check for valid data
        if len(track_times) < 10 or np.any(np.isnan(track_nuclear + track_cytoplasmic)):
            print(f"    Skipping track {track_id}: insufficient valid data")
            continue

        # Run inference
        try:
            cell_data = {
                'nuclear': track_nuclear,
                'cytoplasmic': track_cytoplasmic,
            }

            result = ode_inference.run_inference_for_cell(
                np.array(track_times),
                cell_data,
                n_iterations=n_mcmc_iterations,
            )

            inference_results[track_id] = result

            # Save individual track results
            if save_intermediate:
                track_file = output_path / f"track_{track_id}_inference.pkl"
                with open(track_file, 'wb') as f:
                    pickle.dump(result, f)

        except Exception as e:
            print(f"    Error in inference for track {track_id}: {e}")

    print(f"\n  Successfully inferred parameters for {len(inference_results)} cells")

    # Save summary results
    print("\n[6/6] Saving results...")
    results = {
        'timepoints': timepoints,
        'all_cell_data': all_cell_data,
        'tracks': tracks,
        'inference_results': inference_results,
        'calibrations': calibrations,
    }

    with open(output_path / "complete_results.pkl", 'wb') as f:
        pickle.dump(results, f)

    print(f"\n  Complete results saved to {output_path / 'complete_results.pkl'}")

    # Generate summary plots
    print("\n[7/7] Generating summary plots...")
    generate_summary_plots(results, output_path)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return results


def generate_summary_plots(results: Dict, output_dir: Path):
    """Generate summary plots for the analysis results."""

    # Plot 1: Example cell tracks
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    df = pd.DataFrame(results['all_cell_data'])
    tracks = results['tracks']

    # Plot first few tracks
    for i, (track_id, track) in enumerate(list(tracks.items())[:5]):
        track_data = []
        for t_idx, cell_id in track:
            for data_point in results['all_cell_data']:
                if (data_point['timepoint_idx'] == t_idx and
                    data_point['cell_id'] == cell_id):
                    track_data.append(data_point)
                    break

        if len(track_data) > 0:
            track_df = pd.DataFrame(track_data)
            axes[0].plot(track_df['timepoint'], track_df['nuclear_molecules'],
                        label=f'Track {track_id}', alpha=0.7)
            axes[1].plot(track_df['timepoint'], track_df['cytoplasmic_molecules'],
                        label=f'Track {track_id}', alpha=0.7)

    axes[0].set_xlabel('Time (minutes)')
    axes[0].set_ylabel('Nuclear Repressor (molecules)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Time (minutes)')
    axes[1].set_ylabel('Cytoplasmic Repressor (molecules)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "example_tracks.png", dpi=150)
    plt.close()

    print(f"  Saved example_tracks.png")

    # Plot 2: Parameter distributions
    if len(results['inference_results']) > 0:
        param_names = ['alpha', 'alpha0', 'beta', 'n', 'gamma_m', 'gamma_p']
        all_params = []

        for track_id, result in results['inference_results'].items():
            all_params.append(result['parameter_means'][:-1])  # Exclude sigma

        all_params = np.array(all_params)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for i, name in enumerate(param_names):
            axes[i].hist(all_params[:, i], bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(name)
            axes[i].set_ylabel('Count')
            axes[i].set_title(f'{name} distribution')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "parameter_distributions.png", dpi=150)
        plt.close()

        print(f"  Saved parameter_distributions.png")


if __name__ == "__main__":
    # Run the pipeline with default parameters
    results = run_analysis()
