import repressilator_analysis as ra
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.use('Qt5Agg')
intensity_dir="images/intensity/"
phase_dir="images/phase/"
timepoints, intensity_images, phase_images = ra.image_loader.load_timeseries(
        intensity_dir, phase_dir
    )
segments=ra.fluorescence_extraction.segment_cells(phase_images[0], 5)

# Store centroids and fluorescence values
centroids=np.zeros((len(segments), 4))
fluorescence_data=np.zeros((len(segments), 4))  # [nucleus_red, cell_green]

# Extract first intensity image
intensity_img = intensity_images[0]

for i in range(0, len(segments)):
    # Extract centroids
    centroids[i,:2]=ra.utils.get_centroid(segments[i]["mask"])
    centroids[i,2:]=ra.utils.get_centroid(segments[i]["nmask"])

    # Extract fluorescence values
    # Nucleus fluorescence from red channel (channel 0)
    nuc_rows = segments[i]["nmask"][:, 0]
    nuc_cols = segments[i]["nmask"][:, 1]
    fluorescence_data[i, 0] = np.mean(intensity_img[nuc_rows, nuc_cols, 0])  # Red channel
    fluorescence_data[i, 1] = np.std(intensity_img[nuc_rows, nuc_cols, 0])
    # Whole cell fluorescence from green channel (channel 1)
    cell_rows = segments[i]["mask"][:, 0]
    cell_cols = segments[i]["mask"][:, 1]
    fluorescence_data[i, 2] = np.mean(intensity_img[cell_rows, cell_cols, 1])  # Green channel
    fluorescence_data[i, 3] = np.std(intensity_img[cell_rows, cell_cols, 1]) 

# Save to testdata directory
testdata_dir = "/home/henryll/Documents/ClaudeRepressilator/tests/testdata"
os.makedirs(testdata_dir, exist_ok=True)

with open(os.path.join(testdata_dir, "segment_centroids.txt"), "w") as f:
    np.savetxt(f, centroids)

with open(os.path.join(testdata_dir, "segment_fluorescence.txt"), "w") as f:
    np.savetxt(f, fluorescence_data)

print(f"Saved {len(segments)} cells:")
print(f"  - Centroids saved to segment_centroids.txt")
print(f"  - Fluorescence data saved to segment_fluorescence.txt")
print(f"\nFluorescence summary:")
print(f"  - Nucleus (red): mean={np.mean(fluorescence_data[:, 0]):.2f}, std={np.std(fluorescence_data[:, 0]):.2f}")
print(f"  - Cell (green): mean={np.mean(fluorescence_data[:, 1]):.2f}, std={np.std(fluorescence_data[:, 1]):.2f}")
    
