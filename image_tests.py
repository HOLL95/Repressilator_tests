import repressilator_analysis as ra
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Qt5Agg')
from PIL import Image 
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.filters import sobel     
 
from skimage import filters, measure, morphology, segmentation,feature
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
            print("Fired")
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
actual_data=np.loadtxt("/home/henryll/Documents/ClaudeRepressilator/tests/testdata/F_vs_amount.txt")
for m in range(0, len(timepoints)):
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
        #ax=axes2[i//10, i%10]               
        minr, minc, maxr, maxc = region.bbox                 
     
        # Crop to bounding box
        phase_crop = phase_images[m][minr-0:maxr+0, minc-0:maxc+0]
        
        cell_crop=nuclei[minr-0:maxr+0, minc-0:maxc+0]
        all_cell_crop=cell[minr-0:maxr+0, minc-0:maxc+0]
        #threshed = filters.threshold_multiotsu(phase_crop, classes=3)
        #labelled_cell = np.digitize(phase_crop, bins=threshed) 
        #ax.imshow(cell_crop, cmap="gray")
        #nuclei_mask = (labelled_cell == 0)  # or == 2, depending on what you want   
     
        connected_labels = measure.label(cell_crop)        
        connected_labels = segmentation.clear_border(connected_labels)
  
        n_regions = connected_labels.max()  
        actual=0
        needs_splitting=[]
        for z in range(0, n_regions):
            cluster=np.sum(connected_labels==(z+1))
            if cluster>minsize:
                actual+=1
                nucleus_region = (connected_labels == (z+1))
                props = measure.regionprops(measure.label(nucleus_region))[0] 
                if props.eccentricity>0.8:
                    actual+=1
                    needs_splitting+=[z+1]
        for label_id in needs_splitting:                                                                                                                                                                             
            nucleus_mask = (connected_labels == label_id)
            
            distance = ndimage.distance_transform_edt(nucleus_mask) 
            coords = feature.peak_local_max(distance, footprint=np.ones((3, 3)), labels=nucleus_mask)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndimage.label(mask)
            #fig,ax=plt.subplots(1,3)
            #ax[0].imshow(nucleus_mask)                                                                                                                                                        
            
            watershed_labels = segmentation.watershed(-distance, markers, mask=nucleus_mask)
            if watershed_labels.max() > 1:                                                                                                                                                                               
                connected_labels[nucleus_mask] = 0  # Clear old label                                                                                                                                                    
                max_label = connected_labels.max()                                                                                                                                                                       
                for new_id in range(1, watershed_labels.max() + 1):                                                                                                                                                      
                    connected_labels[watershed_labels == new_id] = max_label + new_id
    
        if actual>1:
            threshed=filters.threshold_otsu(phase_crop)

            cell_mask=  labelled[minr-0:maxr+0, minc-0:maxc+0]<2
            distance = ndimage.distance_transform_edt(cell_mask)
            watershed_labels = segmentation.watershed(-distance, connected_labels, mask=cell_mask)

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
                                        if len(cell_indices)!=0 and len(nuclei_indices)!=0:
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
        # Load actual cell positions and IDs from ground truth data
       

        # Extract first timepoint data (assuming 80 cells per timepoint)
        num_cells_expected = 80
        first_timepoint_data = actual_data[:num_cells_expected]

        # Extract actual positions (columns 8=y, 9=x) and IDs (column 6)
        actual_positions = first_timepoint_data[:, 8:]  # columns 8 and 9 [y, x]
        actual_ids = first_timepoint_data[:, 6].astype(int)  # column 6
        print(len(saved_cells))
        # Calculate centroids for detected cells
        detected_centroids = []
        for cell in saved_cells:
            nmask = cell["nmask"]
            centroid = nmask.mean(axis=0) if len(nmask) > 0 else cell["mask"].mean(axis=0)
            detected_centroids.append(centroid)
            cell["centre"] = centroid

        detected_centroids = np.array(detected_centroids)  # shape: (n_detected, 2) [y, x]

        # Create cost matrix: distance between detected and actual centroids
        n_detected = len(saved_cells)
        n_actual = len(actual_positions)
        cost_matrix = np.zeros((n_detected, n_actual))

        for i in range(n_detected):
            for j in range(n_actual):
                # Calculate Euclidean distance
                cost_matrix[i, j] = np.linalg.norm(detected_centroids[i] - actual_positions[j])

        # Use Hungarian algorithm for optimal matching
        if n_detected > 0 and n_actual > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Assign cell IDs based on matched actual positions
            for detected_idx, actual_idx in zip(row_ind, col_ind):
                saved_cells[detected_idx]["cell_id"] = actual_ids[actual_idx]

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

        # Optimal one-to-one assignment using Hungarian algorithm
        n_old = len(save_trace[m-1])
        n_new = len(saved_cells)

        # Create cost matrix (distances only)
        cost_matrix = all_assignments[:, :, 0]

        # Handle case where n_old != n_new by padding if necessary
        if n_old > 0 and n_new > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Assign IDs based on optimal matching
            for old_idx, new_idx in zip(row_ind, col_ind):
                saved_cells[new_idx]["cell_id"] = save_trace[m-1][old_idx]["cell_id"]

        # Remove spurious cells with duplicate IDs (keep closest to old position)
        id_to_cells = {}
        for idx, cell in enumerate(saved_cells):
            cid = cell["cell_id"]
            if cid not in id_to_cells:
                id_to_cells[cid] = []
            id_to_cells[cid].append(idx)

        removal_indices = []
        for cid, indices in id_to_cells.items():
            if len(indices) > 1:
                # Find the old cell's position for this ID
                old_cells = [c for c in save_trace[m-1] if c["cell_id"] == cid]
                if old_cells:
                    old_centre = old_cells[0]["centre"]
                    # Calculate distance from each duplicate to old position
                    distances = [
                        np.linalg.norm(saved_cells[idx]["centre"] - old_centre)
                        for idx in indices
                    ]
                    # Keep the closest, mark others for removal
                    closest_idx = indices[np.argmin(distances)]
                    removal_indices.extend([idx for idx in indices if idx != closest_idx])

        # Remove spurious cells (in reverse order to preserve indices)
        
        for idx in sorted(removal_indices, reverse=True):
            del saved_cells[idx]
        
        # Visualization for debugging
        positions=actual_data[m*80:(m+1)*80,8:]
        indices=list(actual_data[m*80:(m+1)*80,6])
        cytosol=actual_data[m*80:(m+1)*80,3]
        nucleus=actual_data[m*80:(m+1)*80,0]
        
        


            
        actual_n_areas=actual_data[:,3]
        all_nmasks=[np.mean(intensity_images[m][s["nmask"][:,0], s["nmask"][:,1]]) for s in saved_cells]
        sorted_actual=sorted(actual_n_areas)
        sorted_recovered=sorted(all_nmasks)

        nuclear_image = intensity_images[m][:, :, 0]
        cyto_image = intensity_images[m][:, :, 1]
        
        thresh=0.05*255
        bad=True
        distances=[]
        for i, cell in enumerate(saved_cells):
            cell_id=cell["cell_id"]
            cell_mask = cell["mask"]
            nuclear_mask=cell["nmask"]
            nuclear_intensity = float(np.mean(nuclear_image[nuclear_mask[:,0], nuclear_mask[:,1]]))                                                                                                                           
            cyto_intensity = float(np.mean(cyto_image[cell_mask[:,0], cell_mask[:,1]]))                                                                                                                                       
                                                                              
        
            index=indices.index(cell_id)
            true_pos=positions[index,:]
            distance=np.linalg.norm(true_pos-cell["centre"])
            c_f_distance=abs(cyto_intensity-cytosol[index])
            n_f_distance=abs(nuclear_intensity-nucleus[index])
            distances+=[n_f_distance,c_f_distance]
            if distance>5:
                bad=True
                print(distance, cell_id)

            #    ax2.scatter(true_pos[1], true_pos[0], marker="x", color="green")
            #    ax2.scatter(cell["centre"][1], cell["centre"][0], marker="x", color="red")
        #print(np.mean(distances), max(distances), np.std(distances))
        if bad==True:
            print(len(save_trace[-1]), len(saved_cells))
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
                        color='blue', fontsize=12, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            


            ax1.axis('off')
            ax2.axis('off')
            plt.tight_layout()
            plt.show()
            

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
               
#tasks
#Get trace
#Get fluourescence channel
#write tests
#
                
    #plt.tight_layout()
    #plt.show()
#ax[1].imshow(labelled,cmap='nipy_spectral')
#plt.show()
