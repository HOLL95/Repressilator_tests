"""
Fluorescence extraction utilities for Repressilator analysis.

This module provides functions to segment cells from phase contrast images
and extract fluorescence intensity values for each cell and protein.
"""

import numpy as np
from typing import List, Dict, Tuple
from skimage import filters, measure, morphology, segmentation
from scipy import ndimage




def segment_cells(phase_image: np.ndarray, max_nuc_area: int = 5) -> np.ndarray:
    """
    Segment individual cells from a phase contrast image.

    Args:
        phase_image: Phase contrast image (grayscale or RGB)
        max_nuc_area: Minimum cell area in pixels

    Returns:
        list of cell dictionaries with bbox (bounding box) nuclear and cytosolic masks (nmask, mask) and ID, 
    """
    # Convert to grayscale if RGB
    if phase_image.ndim == 3:
        gray = np.mean(phase_image, axis=2)
    else:
        gray = phase_image

    # Apply Otsu's thresholding
    threshed=filters.threshold_multiotsu(phase_image, classes=3)
    labelled = np.digitize(phase_image, bins=threshed) 
    #darkest is nuclei, middle is cytoplasm
    nuclei=(labelled == 0)
    cytoplasm=(labelled == 1)
    cell=(labelled<2)

    regions = measure.regionprops(measure.label(cytoplasm)) 
    #store for each segment
    saved_cells=[]
    cell_id_counter = 0
    #iterate through each contiguous cell element
    for i, region in enumerate(regions):    
        minr, minc, maxr, maxc = region.bbox 
        #just the cell
        phase_crop = phase_image[minr-0:maxr+0, minc-0:maxc+0]
        cell_crop=cell[minr-0:maxr+0, minc-0:maxc+0]
        nuc_crop=nuclei[minr-0:maxr+0, minc-0:maxc+0]
        connected_labels = measure.label(nuc_crop)   
        #count the number of nuclei     
        n_regions = connected_labels.max()  
        actual=0
        #check it's not a spurious pixel
        for z in range(0, n_regions):
            cluster=np.sum(connected_labels==(z+1))
            if cluster>max_nuc_area:
                actual+=1
        #more than one nucleus per region -> a bunch of collided cells
        if actual>1:
                threshed=filters.threshold_otsu(phase_crop)
                cell_mask=  labelled[minr-0:maxr+0, minc-0:maxc+0]<2
                distance = ndimage.distance_transform_edt(cell_mask)
                #seperate the cells using distances from the nuclei
                watershed_labels = segmentation.watershed(-distance, connected_labels, mask=cell_mask)
                for watershed_id in range(1, watershed_labels.max() + 1):
                    cell_specific_mask = (watershed_labels == watershed_id)
                    nuclei_crop = nuclei[minr-0:maxr+0, minc-0:maxc+0]
                    nuclei_specific_mask = nuclei_crop & cell_specific_mask

                    # Convert masks to full image coordinates
                    cell_indices = np.argwhere(cell_specific_mask)
                    #store as index to main image
                    cell_indices[:, 0] += minr
                    cell_indices[:, 1] += minc
                    nuclei_indices = np.argwhere(nuclei_specific_mask)
                    nuclei_indices[:, 0] += minr
                    nuclei_indices[:, 1] += minc
                    saved_cells.append({
                        'bbox': (minr, minc, maxr, maxc),
                        'mask': cell_indices,
                        'nmask': nuclei_indices,
                        'cell_id': cell_id_counter
                    })
                    cell_id_counter += 1
        else:
            nuclei_crop = nuclei[minr-0:maxr+0, minc-0:maxc+0]
            nuclei_specific_mask = nuclei_crop & nuc_crop
            # store as index to main image
            cell_indices = np.argwhere(cell_crop)
            cell_indices[:, 0] += minr
            cell_indices[:, 1] += minc
            nuclei_indices = np.argwhere(nuclei_specific_mask)
            nuclei_indices[:, 0] += minr
            nuclei_indices[:, 1] += minc
            saved_cells.append({
                'bbox': (minr, minc, maxr, maxc),
                'mask': cell_indices,
                'nmask': nuclei_indices,
                'cell_id': cell_id_counter
            })
            cell_id_counter += 1
    return saved_cells



def extract_nuclear_cytoplasmic(
    intensity_image: np.ndarray,
    labeled_cells: np.ndarray,
    nuclear_channel: str = 'red',
    cytoplasmic_channel: str = 'green',
) -> Dict[int, Dict[str, float]]:
    """
    Extract nuclear and cytoplasmic fluorescence for each cell.

    This assumes:
    - Nuclear repressor is in one fluorescence channel
    - Cytosolic repressor is in another fluorescence channel

    Args:
        intensity_image: RGB fluorescence image
        labeled_cells: List of cells with cytosolic and nuclear masks
        nuclear_channel: Color channel for nuclear fluorescence
        cytoplasmic_channel: Color channel for cytoplasmic fluorescence

    Returns:
        Updated labeled_cells lists to include the nuclear and cytosolic intensities
    """
    channel_map = {'red': 0, 'green': 1, 'blue': 2}

    
    nuclear_idx = channel_map[nuclear_channel.lower()]
    nuclear_image = intensity_image[:, :, nuclear_idx]
    cyto_idx = channel_map[cytoplasmic_channel.lower()]
    cyto_image = intensity_image[:, :, cyto_idx]
    for i in range(0,len(labeled_cells)):
        cell_mask = labeled_cells[i]["mask"]
        nuclear_mask=labeled_cells[i]["nmask"]
        nuclear_intensity = float(np.mean(nuclear_image[nuclear_mask]))       
        cyto_intensity = float(np.mean(cyto_image[cell_mask]))

        labeled_cells[i]["n_intensity"]=nuclear_intensity
        labeled_cells[i]["c_intensity"]=cyto_intensity

    return labeled_cells


def track_cells_across_time(
    labeled_images: List[np.ndarray],
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Track cell identities across time points based on spatial overlap.

    This is a simple tracking algorithm based on maximum overlap between
    consecutive frames.

    Args:
        labeled_images: List of labeled cell images (one per timepoint)

    Returns:
        Dictionary mapping track_id -> [(timepoint_idx, cell_label), ...]
    """
    if len(labeled_images) == 0:
        return {}

    # Initialize tracks with cells from first frame
    tracks = {}
    cell_ids = np.unique(labeled_images[0])
    cell_ids = cell_ids[cell_ids > 0]

    for track_id, cell_id in enumerate(cell_ids):
        tracks[track_id] = [(0, int(cell_id))]

    next_track_id = len(tracks)

    # Process subsequent frames
    for t in range(1, len(labeled_images)):
        prev_labels = labeled_images[t - 1]
        curr_labels = labeled_images[t]

        curr_cell_ids = np.unique(curr_labels)
        curr_cell_ids = curr_cell_ids[curr_cell_ids > 0]

        assigned = set()

        # For each current cell, find best match in previous frame
        for curr_id in curr_cell_ids:
            curr_mask = curr_labels == curr_id

            # Find overlap with previous frame cells
            overlaps = {}
            for prev_id in np.unique(prev_labels[curr_mask]):
                if prev_id == 0:
                    continue
                prev_mask = prev_labels == prev_id
                overlap = np.sum(curr_mask & prev_mask)
                overlaps[prev_id] = overlap

            # Assign to track with maximum overlap
            if overlaps:
                best_prev_id = max(overlaps, key=overlaps.get)

                # Find which track this previous cell belongs to
                for track_id, track_list in tracks.items():
                    if (t - 1, best_prev_id) in track_list:
                        tracks[track_id].append((t, int(curr_id)))
                        assigned.add(curr_id)
                        break
            else:
                # New cell appeared
                tracks[next_track_id] = [(t, int(curr_id))]
                next_track_id += 1
                assigned.add(curr_id)

        # Handle unassigned cells (new cells)
        for curr_id in curr_cell_ids:
            if curr_id not in assigned:
                tracks[next_track_id] = [(t, int(curr_id))]
                next_track_id += 1

    return tracks
