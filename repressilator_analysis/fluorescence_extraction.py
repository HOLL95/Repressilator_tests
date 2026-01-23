"""
Fluorescence extraction utilities for Repressilator analysis.

This module provides functions to segment cells from phase contrast images
and extract fluorescence intensity values for each cell and protein.
"""

import numpy as np
from typing import List, Dict, Tuple
from skimage import filters, measure, morphology, segmentation, feature
from scipy import ndimage
import repressilator_analysis as ra
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def segment_cells(phase_image: np.ndarray, max_nuc_area: int = 5, show_arg=False) -> np.ndarray:
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
    labelled_cells=[]

    cell_id_counter = 0
    counter=1
    #iterate through each contiguous cell element
    for i, region in enumerate(regions):    
        minr, minc, maxr, maxc = region.bbox 
        #just the cell
        phase_crop = phase_image[minr-0:maxr+0, minc-0:maxc+0]
        cell_crop=cell[minr-0:maxr+0, minc-0:maxc+0]
        nuc_crop=nuclei[minr-0:maxr+0, minc-0:maxc+0]
        connected_labels = measure.label(nuc_crop)   
        connected_labels = segmentation.clear_border(connected_labels)
        #count the number of nuclei     
        n_regions = connected_labels.max()  
        actual=0
        #check it's not a spurious pixel
        needs_splitting=[]
        for z in range(0, n_regions):
            cluster=np.sum(connected_labels==(z+1))
            if cluster>max_nuc_area:
                actual+=1
                nucleus_region = (connected_labels == (z+1))
                props = measure.regionprops(measure.label(nucleus_region))[0] 
                if props.eccentricity>0.74:
                    actual+=1
                    needs_splitting+=[z+1]
        #if there's two merged nuclei they need to be split
        for label_id in needs_splitting:                                                                                                                                                                             
            nucleus_mask = (connected_labels == label_id)
            distance = ndimage.distance_transform_edt(nucleus_mask) 
            coords = feature.peak_local_max(distance, footprint=np.ones((3, 3)), labels=nucleus_mask)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndimage.label(mask)
            watershed_labels = segmentation.watershed(-distance, markers, mask=nucleus_mask)
            if watershed_labels.max() > 1:                                                                                                                                                                               
                connected_labels[nucleus_mask] = 0  # Clear old label                                                                                                                                                    
                max_label = connected_labels.max()                                                                                                                                                                       
                for new_id in range(1, watershed_labels.max() + 1):                                                                                                                                                      
                    connected_labels[watershed_labels == new_id] = max_label + new_id
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
                    position= nuclei_indices.mean(axis=0)
                    if ra.utils.check_position_dupes(position, [s["centre"] for s in labelled_cells]) is False:
                        if len(cell_indices)!=0 and len(nuclei_indices)!=0:
                            labelled_cells.append({
                                'bbox': (minr, minc, maxr, maxc),
                                'mask': cell_indices,
                                'nmask': nuclei_indices,
                                'cell_id': cell_id_counter,
                                "centre" :position
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
            position= nuclei_indices.mean(axis=0)
            if ra.utils.check_position_dupes(position, [s["centre"] for s in labelled_cells]) is False:
                if len(cell_indices)!=0 and len(nuclei_indices)!=0:
                    labelled_cells.append({
                        'bbox': (minr, minc, maxr, maxc),
                        'mask': cell_indices,
                        'nmask': nuclei_indices,
                        'cell_id': cell_id_counter,
                        "centre" :position
                    })
                    cell_id_counter += 1
    if show_arg==True:
        plt.imshow(phase_image, cmap='gray')
        all_ids=[c["cell_id"] for c in labelled_cells]
        for old_idx, cell_data in enumerate(labelled_cells):
            centroid = cell_data['centre']
            
            plt.text(centroid[1], centroid[0], cell_data["cell_id"],
                    color='red', fontsize=12, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        plt.show()
    return labelled_cells



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
    phase_images: List[np.ndarray],
    min_nuc_area: int = 5,
):
    """
    Track cell identities across time points based on spatial overlap.

    This is a simple tracking algorithm based on maximum overlap between
    consecutive frames.

    Args:
        labeled_images: List of phase contrast images

    Returns:
        Dictionary containing list of cells at each timepoint, where each cell is tagged with a persistent cellID
    """
    saved_tracks=[]
    labelled_cells=ra.fluorescence_extraction.segment_cells(phase_images[0], min_nuc_area)
    saved_tracks.append(labelled_cells)
    for i in range(1, len(phase_images)):
        labelled_cells=ra.fluorescence_extraction.segment_cells(phase_images[i], min_nuc_area)
        cost_matrix=np.zeros((len(saved_tracks[i-1]), len(labelled_cells)))
        for q in range(0, len(saved_tracks[i-1])):
            old_centre=saved_tracks[i-1][q]["centre"]
            for j in range(0, len(labelled_cells)):
                new_centre=labelled_cells[j]["centre"]
                cost_matrix[q,j]=np.linalg.norm(new_centre-old_centre)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # Assign IDs based on optimal matching
        for old_idx, new_idx in zip(row_ind, col_ind):
            labelled_cells[new_idx]["cell_id"] = saved_tracks[i-1][old_idx]["cell_id"]
        # Remove spurious cells with duplicate IDs (keep closest to old position)
        id_to_cells = {}
        for idx, cell in enumerate(labelled_cells):
            cid = cell["cell_id"]
            if cid not in id_to_cells:
                id_to_cells[cid] = []
            id_to_cells[cid].append(idx)
        removal_indices = []
        for cid, indices in id_to_cells.items():
            if len(indices) > 1:
                # Find the old cell's position for this ID
                old_cells = [c for c in saved_tracks[i-1] if c["cell_id"] == cid]
                if old_cells:
                    old_centre = old_cells[0]["centre"]
                    # Calculate distance from each duplicate to old position
                    distances = [
                        np.linalg.norm(labelled_cells[idx]["centre"] - old_centre)
                        for idx in indices
                    ]
                    # Keep the closest, mark others for removal
                    closest_idx = indices[np.argmin(distances)]
                    removal_indices.extend([idx for idx in indices if idx != closest_idx])

        # Remove spurious cells (in reverse order to preserve indices)
        
        for idx in sorted(removal_indices, reverse=True):
            del labelled_cells[idx]
        saved_tracks.append(labelled_cells)
    return saved_tracks
        