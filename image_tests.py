import repressilator_analysis as ra
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Qt5Agg')
from PIL import Image 
from scipy import ndimage
from skimage.filters import sobel                                                                                                                                                                                
 
from skimage import filters, measure, morphology, segmentation


def is_duplicate_mask(new_bbox, new_mask, saved_cells, threshold=0.95):
    """
    Check if a mask is already saved (>threshold overlap).

    Args:
        new_bbox: (minr, minc, maxr, maxc) - bounding box in full image coords
        new_mask: Boolean mask (cropped to bbox region)
        saved_cells: List of previously saved cell dictionaries
        threshold: IoU threshold for duplicate detection (default 0.95)

    Returns: True if duplicate found, False otherwise
    """
    new_minr, new_minc, new_maxr, new_maxc = new_bbox

    for cell in saved_cells:
        old_minr, old_minc, old_maxr, old_maxc = cell['bbox']

        # Check if bounding boxes overlap at all
        if (new_maxr < old_minr or new_minr > old_maxr or
            new_maxc < old_minc or new_minc > old_maxc):
            continue  # No overlap, skip

        # Calculate overlap region
        overlap_minr = max(new_minr, old_minr)
        overlap_minc = max(new_minc, old_minc)
        overlap_maxr = min(new_maxr, old_maxr)
        overlap_maxc = min(new_maxc, old_maxc)

        # Extract the overlapping portions from both masks
        # Convert bbox coordinates to local mask coordinates
        new_overlap_minr = overlap_minr - new_minr
        new_overlap_minc = overlap_minc - new_minc
        new_overlap_maxr = overlap_maxr - new_minr
        new_overlap_maxc = overlap_maxc - new_minc

        old_overlap_minr = overlap_minr - old_minr
        old_overlap_minc = overlap_minc - old_minc
        old_overlap_maxr = overlap_maxr - old_minr
        old_overlap_maxc = overlap_maxc - old_minc

        # Get overlapping regions from both masks
        new_overlap_region = new_mask[new_overlap_minr:new_overlap_maxr,
                                       new_overlap_minc:new_overlap_maxc]
        old_overlap_region = cell['mask'][old_overlap_minr:old_overlap_maxr,
                                           old_overlap_minc:old_overlap_maxc]

        # Calculate IoU = intersection / union
        intersection = np.sum(new_overlap_region & old_overlap_region)
        union = np.sum(new_mask) + np.sum(cell['mask']) - intersection

        if union > 0:
            iou = intersection / union
            if iou >= threshold:
                return True

    return False


intensity_dir="images/tests/intensity/jammed"
phase_dir="images/tests/phase/jammed"
intensity_dir="images/intensity/"
phase_dir="images/phase/"
timepoints, intensity_images, phase_images = ra.image_loader.load_timeseries(
        intensity_dir, phase_dir
    )

#print(timepoints)
#fig,ax=plt.subplots(1,2)
idx=5
#ax[0].imshow(phase_images[idx], cmap="gray")
#plt.show()
for m in range(0, len(phase_images),10):
    total=0
    saved_cells = []  # Reset for each frame
    cell_id_counter = 0
    min_cell_area=10
    threshed=filters.threshold_multiotsu(phase_images[m], classes=3)
    labelled = np.digitize(phase_images[m], bins=threshed) 
    #labelled = morphology.remove_small_objects(labelled, max_size=min_cell_area)
    #labelled = morphology.remove_small_holes(labelled, max_size=min_cell_area)
    nuclei=(labelled == 0)
    cytoplasm=(labelled == 1)
    cell=(labelled<2)
    

    from skimage.measure import regionprops                                                                                                                                                                          
                                                                                                                                                                                            
    regions = regionprops(measure.label(cytoplasm))        
    cols=10
    rows=7                                                                                                                                                                                                        
    #fig, axes1 = plt.subplots(rows, cols)      
    fig2, axes2= plt.subplots(rows, cols)                                                                                                                                                      #axed=[axes1, axes2]
    #axed=[axes1, axes2]                         
    minsize=4                                                                                                                                                                                       
    for i, region in enumerate(regions):    
        
    # Get bounding box coordinates   
        axes=axes2#axed[j]        
        ax=axes[i//10, i%10]                                                                                                                                                    
        minr, minc, maxr, maxc = region.bbox                                                                                                                                                                         
                                                                                                                                                                                
        # Crop to bounding box
        phase_crop = phase_images[m][minr-0:maxr+0, minc-0:maxc+0]
        cell_crop=nuclei[minr-0:maxr+0, minc-0:maxc+0]
        #threshed = filters.threshold_multiotsu(phase_crop, classes=3)
        #labelled_cell = np.digitize(phase_crop, bins=threshed) 
        
        #nuclei_mask = (labelled_cell == 0)  # or == 2, depending on what you want                                                                                                                                        
                                                                                                                                                                                
        connected_labels = measure.label(cell_crop)        
                                                                                                                                                                                                                                                                            
        n_regions = connected_labels.max()  
        actual=0
        
        for z in range(0, n_regions):
            cluster=np.sum(connected_labels==(z+1))
            if cluster>minsize:
                actual+=1
        

        if actual>1:
            threshed=filters.threshold_otsu(phase_crop)

            cell_mask=  labelled[minr-0:maxr+0, minc-0:maxc+0]<2
            distance = ndimage.distance_transform_edt(cell_mask)
            watershed_labels = segmentation.watershed(-distance, connected_labels, mask=cell_mask)
            ax.imshow(watershed_labels, cmap="Spectral")

            # Process each watershed-segmented cell
            for watershed_id in range(1, watershed_labels.max() + 1):
                # Extract mask for this specific cell
                cell_specific_mask = (watershed_labels == watershed_id)

                # Skip if too small
                if np.sum(cell_specific_mask) < minsize:
                    continue

                # Check for duplicate
                if not is_duplicate_mask((minr, minc, maxr, maxc), cell_specific_mask, saved_cells):
                    saved_cells.append({
                        'bbox': (minr, minc, maxr, maxc),
                        'mask': cell_specific_mask.copy(),
                        'cell_id': cell_id_counter
                    })
                    cell_id_counter += 1

        else:
            ax.imshow(phase_crop, cmap='gray')

            # For single cell case
            if actual == 1:
                # Check for duplicate
                if not is_duplicate_mask((minr, minc, maxr, maxc), cell_crop, saved_cells):
                    saved_cells.append({
                        'bbox': (minr, minc, maxr, maxc),
                        'mask': cell_crop.copy(),
                        'cell_id': cell_id_counter
                    })
                    cell_id_counter += 1

        ax.set_title(actual)
        total+=actual                                                                                                                         
        ax.axis('off')     
                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                                                                                    
                                                                                                                                                                                
    print(f"Frame {m}: Total count = {total}, Unique cells saved = {len(saved_cells)}, Duplicates removed = {total - len(saved_cells)}")
    plt.tight_layout()
    plt.show()
#ax[1].imshow(labelled,cmap='nipy_spectral')
#plt.show()
