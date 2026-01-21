import unittest
from unittest.mock import patch
import pytest
import numpy as np
import os
import repressilator_analysis as ra
import matplotlib.pyplot as plt

# Fixture to load test images - shared across all test classes
@pytest.fixture(scope="module")
def test_images():
    """Load test images once and share across all tests in this module."""
    intensity_dir = os.path.join(os.path.dirname(__file__), os.pardir, "images", "intensity")
    phase_dir = os.path.join(os.path.dirname(__file__), os.pardir, "images", "phase")

    timepoints, intensity_images, phase_images = ra.image_loader.load_timeseries(
        intensity_dir, phase_dir
    )

    return {
        'timepoints': timepoints,
        'intensity_images': intensity_images,
        'phase_images': phase_images
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
            raise ValueError("There are 80 cells in the image, not {0}".format(len(segment_list)))

        # Load reference centroids
        testdata_path = os.path.join(os.path.dirname(__file__), "testdata", "segment_centroids.txt")
        all_true_centroids = np.loadtxt(testdata_path)

        # Calculate centroids from segmented cells
        cell_centroids = np.array([ra.utils.get_centroid(cell["mask"]) for cell in segment_list])
        nuclear_centroids = np.array([ra.utils.get_centroid(cell["nmask"]) for cell in segment_list])
        all_test_centroids = np.column_stack((cell_centroids, nuclear_centroids))

        # Sort both reference and test centroids by first coordinate
        true_sort = np.argsort(all_true_centroids[:, 0])
        sorted_centres = np.array([all_true_centroids[x, :] for x in true_sort])
        test_sort = np.argsort(all_test_centroids[:, 0])
        sorted_test_centres = np.array([all_test_centroids[x, :] for x in test_sort])

        mean_difference = np.zeros(len(segment_list))
        labels = ["Cells", "Nuclei"]
        sizes = [5, 10]

        for j in range(0, 2):
            start = j * 2
            end = (j + 1) * 2
            for i in range(0, len(segment_list)):
                # Check required keys on first iteration
                if j == 0:
                    for key in ["cell_id", "nmask", "mask", "bbox"]:
                        if key not in segment_list[i]:
                            raise KeyError(f"Segment_cells list needs every element to contain {['cell_id', 'nmask', 'mask', 'bbox']}, but index {i} is missing {key}")

                # Calculate distance between sorted centroids
                mean_difference[i] = np.linalg.norm(sorted_centres[i, start:end] - sorted_test_centres[i, start:end])

            # Check if mean difference exceeds tolerance
            if np.mean(mean_difference) > sizes[j]:
                raise ValueError(f"Mean difference between test centroid in {labels[j]} is too large at {round(np.mean(mean_difference), 2)} pixels, problems with your segmentation!")


@pytest.mark.integration
class TestFluorescenceIntegration:
    """Integration tests for fluorescence extraction pipeline."""

    def test_extraction(self, test_images):
        """Test that fluorescence extraction produces correct values."""
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

        # Load reference data: [nucleus_mean, nucleus_std, cell_mean, cell_std] per row
        testdata_path = os.path.join(os.path.dirname(__file__), "testdata", "segment_fluorescence.txt")
        testdata = np.loadtxt(testdata_path)

        # Load centroids for error reporting
        centroid_path = os.path.join(os.path.dirname(__file__), "testdata", "segment_centroids.txt")
        all_true_centroids = np.loadtxt(centroid_path)

        # Check both nucleus (j=0) and cell (j=1) fluorescence
        for j in range(0, 2):
            close_enough = []
            # For each reference cell
            for i in range(0, testdata.shape[0]):
                # Find test cells within 1 std of this reference value
                mean_idx = j * 2  # 0 for nucleus, 2 for cell
                std_idx = (j * 2) + 1  # 1 for nucleus, 3 for cell

                any_close = [x for x in range(0, test_vals.shape[0])
                            if abs(testdata[i, mean_idx] - test_vals[x, j]) < testdata[i, std_idx]]

                if not any_close:
                    raise ValueError(f"Key {keys[j]} (known position {all_true_centroids[i, j*2:(j+1)*2]}) extracted fluorescence value not within one s.d. of any true known value at t=0")

                close_enough.append(any_close)

            # Check that all test cells matched at least one reference cell
            close_enough = set(np.array(close_enough).flatten())
            not_found = set(range(0, 80)) - close_enough

            if not_found:
                not_found_list = list(not_found)
                positions = [all_true_centroids[x, j*2:(j+1)*2] for x in not_found_list]
                str_positions = [str(x) for x in positions]
                str_positions = "\n" + "\n".join(str_positions) + "\n"
                raise ValueError(f"The following {keys[j]} centroids fluorescence values were not found in the extracted dataset:{str_positions}")

            


# You can add more test classes that all use the same test_images fixture:
# @pytest.mark.slow
# class TestFluorescencePerformance:
#     def test_segmentation_speed(self, test_images):
#         # Performance tests using the same data
#         pass
