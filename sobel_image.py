import repressilator_analysis as ra
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Qt5Agg')
from PIL import Image 
from scipy import ndimage
from skimage.filters import sobel                                                                                                                                                                                
 
from skimage import filters, measure, morphology, segmentation

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
    labelled=ra.fluorescence_extraction.segment_cells(phase_images[m], 5)
    from skimage.measure import regionprops                                                                                                                                                                          
                                                                                                                                                                                            
    regions = regionprops(labelled)        
    cols=10
    rows=7                                                                                                                                                                                                        
    #fig, axes1 = plt.subplots(rows, cols)      
    fig2, axes2= plt.subplots(rows, cols)                                                                                                                                                      #axed=[axes1, axes2]
    #axed=[axes1, axes2]                         
    minsize=4                                                                                                                                                                                       
    for i, region in enumerate(regions):    
        for j in range(1, 2):
        # Get bounding box coordinates   
            axes=axes2#axed[j]        
            ax=axes[i//10, i%10]                                                                                                                                                    
            minr, minc, maxr, maxc = region.bbox                                                                                                                                                                         
                                                                                                                                                                                    
            # Crop to bounding box
            phase_crop = phase_images[m][minr-0:maxr+0, minc-0:maxc+0]
           
            threshed = filters.threshold_multiotsu(phase_crop, classes=3)
            labelled_cell = np.digitize(phase_crop, bins=threshed) 
            
            nuclei_mask = (labelled_cell == 0)  # or == 2, depending on what you want                                                                                                                                        
                                                                                                                                                                                    
            connected_labels = measure.label(nuclei_mask)        
                                                                                                                                                                                                                                                                                
            n_regions = connected_labels.max()  
            actual=0
            
            for z in range(0, n_regions):
                cluster=np.sum(connected_labels==(z+1))
                if cluster>minsize:
                    actual+=1
            grad=sobel(labelled_cell)
            cell_mask=  labelled[minr-0:maxr+0, minc-0:maxc+0] == region.label 
            watershed_labels = segmentation.watershed(-grad, connected_labels, mask=cell_mask)
            if j==1:
                ax.imshow(watershed_labels, cmap='gray')                                                                                                                                                                     
          
                
            total+=actual                                                                                                                         
            ax.axis('off')     
                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                                                                                    
                                                                                                                                                                                
    print(total)  
    plt.tight_layout()    
    plt.show()
#ax[1].imshow(labelled,cmap='nipy_spectral')
#plt.show()
