import unittest
from unittest.mock import patch
import pytest
import numpy as np
import os
import repressilator_analysis as ra
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

@pytest.fixture(scope="module")
def test_images(request):
    """Load test images once and share across all tests in this module."""
    intensity_dir = os.path.join(os.path.dirname(__file__), os.pardir, "images", "intensity")
    phase_dir = os.path.join(os.path.dirname(__file__), os.pardir, "images", "phase")

    timepoints, intensity_images, phase_images = ra.image_loader.load_timeseries(
        intensity_dir, phase_dir
    )
    testdata_path = os.path.join(os.path.dirname(__file__), "testdata", "F_vs_amount.txt")
    all_true_data= np.loadtxt(testdata_path)
   
    return {
        'timepoints': timepoints,
        'intensity_images': intensity_images,
        'phase_images': phase_images,
        "data":all_true_data
    }

@pytest.mark.unit
class TestFluorescenceExtraction:
    """Unit tests for fluorescence extraction functions."""

    def test_segment_cells(self, test_images):
        """Test that segment_cells correctly identifies cells and their properties."""
        phase_images = test_images['phase_images']
        
        segment_list = ra.fluorescence_extraction.segment_cells(phase_images[0], 5)

        # Check return type
        if isinstance(segment_list, np.ndarray) and len(segment_list.shape) == 2:
            raise TypeError("Unittest implementation expects 1D list of dictionaries - LLM implementation returns the whole image. If you are using this implementation then run `pytest -m integration`")
        elif isinstance(segment_list[0], dict) is False:
            raise TypeError("Unittest implementation expects 1D list of dictionaries - LLM implementation returns the whole image. If you are using this implementation then run `pytest -m integration`")

        # Check expected number of cells
        if len(segment_list) != 80:
            raise ValueError("Segmentation error:There are 80 cells in the image, not {0} at timepoint {1}".format(len(segment_list),test_images["timepoints"][i]))
        for j in range(0, len(segment_list)):
            for key in ["cell_id", "nmask", "mask", "bbox"]:
                if key not in segment_list[j]:
                    raise KeyError(f"Segment_cells list needs every element to contain {['cell_id', 'nmask', 'mask', 'bbox', "centre"]}, but index {i} is missing {key}")




@pytest.mark.integration
class TestFluorescenceIntegration:
    """Integration tests for fluorescence extraction pipeline."""
    def test_extract_nuclear_cytoplasmic(self, test_images):
        phase_images = test_images['phase_images']
        intensity_images = test_images['intensity_images']

        # Segment cells
        segmented_arg = ra.fluorescence_extraction.segment_cells(phase_images[0], 5)

        # Extract fluorescence - note: this modifies segmented_arg in place
        results=ra.fluorescence_extraction.extract_nuclear_cytoplasmic(
            intensity_images[0],
            segmented_arg,
        )

        # Determine which keys are used for fluorescence data
        if "cytoplasmic" in results[0]:
            keys = ["nuclear", "cytoplasmic"]
        elif "n_intensity" in results[0]:
            keys = ["n_intensity", "c_intensity"]
        else:
            raise KeyError(f"Expected fluorescence keys not found in segmented data. Available keys: {results[0].keys()}")

        # Check we have the expected number of cells
        if len(results) != 80:
            raise ValueError("There are 80 cells in the image, not {0}".format(len(results)))

        # Extract test values (nucleus, cell) for each segment
        test_vals = np.column_stack((
            np.array([s[keys[0]] for s in results]),
            np.array([s[keys[1]] for s in results])
        ))
        test_fs=test_images["data"][:80, [0,3]]
        
        for j in range(0, 2):
            cost_matrix = np.abs(test_vals[:,j][:, np.newaxis] - test_fs[:,j][np.newaxis, :])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            best_distances=[cost_matrix[x] for x in zip(row_ind,col_ind)]
            if np.mean(best_distances)>5 or np.std(best_distances)>5:
                raise ValueError(f"Fluorescence value extraction in {keys[j]} over distance threshold to true value (mean difference {np.mean(best_distances)}, s.d. {np.std(best_distances)})")

