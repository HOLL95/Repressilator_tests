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
    def test_track_cells(self, test_images):
        tracks=ra.fluorescence_extraction.track_cells_across_time(test_images["phase_images"], 5)
        if len(tracks)==2:
            labelled_images=tracks[1]
            new_tracks=[[] for x in range(0, len(test_images["phase_images"]))]
            for i in range(0, len(test_images["phase_images"])):
                for key in tracks[0].keys():
                    for elem in tracks[0][key]:
                        if elem[0]==i:
                            new_tracks[i].append({"cell_id":int(key), "centre":list(elem[2])})
            tracks=new_tracks
        actual_data=test_images["data"]
        for i in range(0, len(tracks)):
            positions=actual_data[i*80:(i+1)*80,8:]
            indices=list(actual_data[i*80:(i+1)*80,6])
            if len(tracks[i])!=80:
                raise ValueError(f"At timepoint {i}, the number of cells in the segmented image is {len(tracks[i])}, not 80")
            cost_matrix=np.zeros((len(tracks[i]), len(tracks[i])))
            for r in range(0, positions.shape[0]):
                for q in range(0, len(tracks[i])):
                    cost_matrix[r, q]=np.linalg.norm(positions[r,:]-tracks[i][q]["centre"])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            actual_cost_values=[cost_matrix[x,y] for x,y in zip(row_ind, col_ind)]
            errors=[x>7 for x in actual_cost_values]
            num_errors=len([x for x in errors if x is True])
            if i==0:
                if any(errors):
                    raise ValueError(f"At the first timepoint, {num_errors} cell centres have been assigned incorrectly")
                mapping={tracks[i][y]["cell_id"]:indices[x] for x,y in zip(row_ind, col_ind)}   
                backmapping={indices[x]:tracks[i][y]["cell_id"] for x,y in zip(row_ind, col_ind)}   
            else:
                for j in range(0, len(col_ind)):
                    cidx=col_ind[j]
                    #improper assignment. 
                    if mapping[tracks[i][cidx]["cell_id"]]!=indices[row_ind[j]]:
                        raise ValueError(f"At time {i}, cell_id {backmapping[indices[row_ind[j]]]} (true position {positions[row_ind[j]]}) has been incorectly assigned to cell_id {tracks[i][cidx]["cell_id"]}")
                    distance=actual_cost_values[j]
                    if distance>7:
                        raise ValueError(f"At time {i}, cell_id {backmapping[indices[row_ind[j]]]} (true position {positions[row_ind[j]]}) has been given the position {tracks[i][col_ind[j]]["centre"]}, (error:{actual_cost_values[j]:.2f} pixels)")

