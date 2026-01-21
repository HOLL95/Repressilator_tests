import numpy as np
def map_mask_to_image(mask, image, value):
    rows=mask[:,0]
    cols=mask[:,1]
    copy_img=image.copy()
    copy_img[rows, cols]=value
    return copy_img
def get_centroid(mask):
    centroid_row = np.mean(mask[:, 0])                                                                                                                                                                
    centroid_col = np.mean(mask[:, 1])
    return [centroid_row, centroid_col]