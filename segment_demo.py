import repressilator_analysis as ra
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import random
matplotlib.use('Qt5Agg')
intensity_dir = "images/intensity/"
phase_dir = "images/phase/"
timepoints, intensity_images, phase_images = ra.image_loader.load_timeseries(
    intensity_dir, phase_dir
)
segments = ra.fluorescence_extraction.segment_cells(phase_images[0], 5)

print(f"Segmented {len(segments)} cells")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Get the phase image and convert to RGB for overlay
phase_img = phase_images[0]
if phase_img.ndim == 2:
    # Convert grayscale to RGB
    cell_overlay = np.stack([phase_img, phase_img, phase_img], axis=-1)
    nuc_overlay = np.stack([phase_img, phase_img, phase_img], axis=-1)
else:
    cell_overlay = phase_img.copy()
    nuc_overlay = phase_img.copy()

# Normalize to 0-1 range for blending
cell_overlay = cell_overlay.astype(float) / cell_overlay.max()
nuc_overlay = nuc_overlay.astype(float) / nuc_overlay.max()

# Generate colors for each cell
colors = plt.cm.tab20(np.linspace(0, 1, len(segments)))

random.shuffle(colors)
# Overlay cell masks in varying colors (left half)
for i in range(len(segments)):
    cell_rows = segments[i]["mask"][:, 0]
    cell_cols = segments[i]["mask"][:, 1]

    # Blend color with original image (70% color, 30% original)
    cell_overlay[cell_rows, cell_cols] = np.random.rand(3) #* 0.7 + cell_overlay[cell_rows, cell_cols] * 0.3

# Overlay nucleus masks in varying colors (right half)
for i in range(len(segments)):
    nuc_rows = segments[i]["nmask"][:, 0]
    nuc_cols = segments[i]["nmask"][:, 1]

    # Blend color with original image (70% color, 30% original)
    nuc_overlay[nuc_rows, nuc_cols] = np.random.rand(3) #* 0.7 + nuc_overlay[nuc_rows, nuc_cols] * 0.3

# Display cell masks
ax1.imshow(cell_overlay, cmap="Spectral")
ax1.set_title(f'Cell Masks ({len(segments)} cells)', fontsize=14, fontweight='bold')
ax1.axis('off')

# Display nucleus masks
ax2.imshow(nuc_overlay, cmap="Spectral")
ax2.set_title(f'Nuclear Masks ({len(segments)} nuclei)', fontsize=14, fontweight='bold')
ax2.axis('off')

plt.tight_layout()
plt.show()

print("Visualization complete!")
