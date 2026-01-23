import repressilator_analysis as ra
import matplotlib.pyplot as plt
import numpy as np
import os
intensity_dir = os.path.join(os.path.dirname(__file__),  "images", "intensity")
phase_dir = os.path.join(os.path.dirname(__file__),  "images", "phase")

timepoints, intensity_images, phase_images = ra.image_loader.load_timeseries(
    intensity_dir, phase_dir
)
true_values=np.loadtxt("/home/henryll/Documents/ClaudeRepressilator/tests/testdata/F_vs_amount.txt")
segment_list = ra.fluorescence_extraction.segment_cells(phase_images[0], 5)
get_all_centroids=np.array([ra.utils.get_centroid(s["nmask"]) for s in segment_list])
argsorts=[np.argsort(x[:,0]) for x in [get_all_centroids, true_values[:80,8:]]]

for i in range(0, 80):
    print(get_all_centroids[argsorts[0][i],:], true_values[argsorts[1][i], 8:])
