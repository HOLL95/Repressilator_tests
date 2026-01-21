import repressilator_analysis as ra
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Qt5Agg')
from PIL import Image 
from scipy import ndimage
from skimage.filters import sobel     
 
from skimage import filters, measure, morphology, segmentation
def calculate_intersection(new_bbox, new_mask, old_bbox, old_mask):
    """
    Calculate IoU between two masks.
    new_mask and old_mask are now numpy arrays of shape (N, 2) containing [row, col] indices
    in full image coordinates.
    """
    # Convert indices to sets of tuples for efficient intersection
    new_pixels = set(map(tuple, new_mask))
    old_pixels = set(map(tuple, old_mask))

    intersection = len(new_pixels & old_pixels)
    union = len(new_pixels | old_pixels)

    if union > 0:
        iou = intersection / union
        return iou
    else:
        return 0
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
        intersection_pc=calculate_intersection(new_bbox, new_mask,cell['bbox'], cell["mask"])
        if intersection_pc>threshold:
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
save_trace=[]
for m in range(0, 20):
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
    #fig2, axes2= plt.subplots(rows, cols) 
    #axed=[axes1, axes2]      
    minsize=4            
    for i, region in enumerate(regions):    
        
    # Get bounding box coordinates   
        #axes=axes2#axed[j]        
        #ax=axes[i//10, i%10]               
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
            #ax.imshow(phase_crop, cmap="Spectral")

            # Process each watershed-segmented cell
            for watershed_id in range(1, watershed_labels.max() + 1):
                # Extract mask for this specific cell
                cell_specific_mask = (watershed_labels == watershed_id)


                # Check for duplicate
                if not is_duplicate_mask((minr, minc, maxr, maxc), cell_specific_mask, saved_cells):
                                        # Extract nuclei mask for this cell
                                        nuclei_crop = nuclei[minr-0:maxr+0, minc-0:maxc+0]
                                        nuclei_specific_mask = nuclei_crop & cell_specific_mask

                                        # Convert masks to full image coordinates
                                        cell_indices = np.argwhere(cell_specific_mask)
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

        # For single cell case
        if actual == 1:
            # Check for duplicate
            if not is_duplicate_mask((minr, minc, maxr, maxc), cell_crop, saved_cells):
                # Extract nuclei mask for this cell
                nuclei_crop = nuclei[minr-0:maxr+0, minc-0:maxc+0]
                nuclei_specific_mask = nuclei_crop & cell_crop

                # Convert masks to full image coordinates
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

    if m==0:
        save_trace.append(saved_cells)
    else:
        # Store both euclidean distance and area difference
        # Shape: (old_cells, new_cells, 2) where last dimension is [distance, area_diff]
        all_assignments=np.zeros((len(save_trace[m-1]), len(saved_cells), 2))
        for q in range(0, len(save_trace[m-1])):
            # Calculate centroid and area for old cell
            # Use nuclei mask for centroid, cell mask for area
            old_nmask = save_trace[m-1][q]["nmask"]
            old_mask = save_trace[m-1][q]["mask"]
            old_centroid = old_nmask.mean(axis=0) if len(old_nmask) > 0 else old_mask.mean(axis=0)
            old_area = len(old_mask)
            save_trace[m-1][q]["centre"]=old_centroid

            for j in range(0, len(saved_cells)):
                # Calculate centroid and area for new cell
                # Use nuclei mask for centroid, cell mask for area
                new_nmask = saved_cells[j]["nmask"]
                new_mask = saved_cells[j]["mask"]
                new_centroid = new_nmask.mean(axis=0) if len(new_nmask) > 0 else new_mask.mean(axis=0)
                new_area = len(new_mask)
                saved_cells[j]["centre"]=new_centroid

                # Calculate euclidean distance between centroids
                euclidean_dist = np.linalg.norm(old_centroid - new_centroid)

                # Calculate difference in cell area
                area_diff = np.abs(old_area - new_area)

                all_assignments[q,j,0] = euclidean_dist
                all_assignments[q,j,1] = area_diff         

        # Simple assignment: each old cell to its nearest new cell
        n_old = len(save_trace[m-1])
        n_new = len(saved_cells)
        print(n_old, n_new)
        assignments = {}  # old_idx -> new_idx
        for old_idx in range(n_old):
            argmin = np.argmin(all_assignments[old_idx, :, 0])
            saved_cells[argmin]["cell_id"]=save_trace[m-1][old_idx]["cell_id"]

        # Visualization for debugging
        
        removal_idx=[]
        
        for r in range(0, n_old):
            #r is the row index of all_assignments
            dupes=[x for x in range(0, n_new) if saved_cells[x]["cell_id"]==r]
            
            if len(dupes)>1:
               
                old_centre=[cell["centre"] for cell in save_trace[m-1] if cell["cell_id"]==r]

                minima=[
                    np.linalg.norm(saved_cells[y]["centre"]-old_centre) for y in dupes
                ]
                #for z in range(0, len(dupes)):
                #    other_distances=[np.linalg.norm(saved_cells[dupes[z]]["centre"]-saved_cells[x]["centre"]) for x in range(0, len(saved_cells)) if x!=dupes[z]]
                #    print(min(other_distances))
                minima=np.argmin(minima)
               
                del dupes[minima]
                
                removal_idx+=dupes
        saved_cells=[saved_cells[x] for x in range(0, len(saved_cells)) if x not in removal_idx]#
       
       
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        # Show t-1 frame with cell numbers
        ax1.imshow(phase_images[m-1], cmap='gray')
        ax1.set_title(f'Frame {m-1} (t-1)')
        for old_idx, cell_data in enumerate(save_trace[m-1]):
            centroid = cell_data['centre']
            ax1.text(centroid[1], centroid[0], cell_data["cell_id"],
                    color='red', fontsize=12, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Show t frame with cell numbers and assignments
        ax2.imshow(phase_images[m], cmap='gray')
        ax2.set_title(f'Frame {m} (t)')
        for new_idx, cell_data in enumerate(saved_cells):
            centroid = cell_data['centre']

            # Find which old cell maps to this new cell
            
            label= cell_data["cell_id"]

            ax2.text(centroid[1], centroid[0], label,
                    color='blue', fontsize=10, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax1.axis('off')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
        plt.pause(0.5)  # Pause to see each frame

        save_trace.append(saved_cells)
        for m in range(0, len(save_trace[-1])):
            save_trace[-1][m]["identified_flag"]=True
        full_set=set(range(0,80))
        all_cell_ids=set([c["cell_id"] for c in saved_cells])  
        missing=list(full_set-all_cell_ids ) 
        for m in range(0, len(missing)):
            found=[cell for cell in save_trace[-2] if cell["cell_id"]==missing[m]]
            found[0]["identified_flag"]=False
            save_trace[-1].append(found[0])
               

                
    #plt.tight_layout()
    #plt.show()
#ax[1].imshow(labelled,cmap='nipy_spectral')
#plt.show()
