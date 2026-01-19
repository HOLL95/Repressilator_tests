# Challenges and Unclear Aspects of the Repressilator Analysis Project

This document outlines the challenges encountered and uncertainties faced during the implementation of the Repressilator fluorescence microscopy analysis package.

## 1. Image Format and Channel Assignment

### Challenge
The instructions state that "two of the three transcription repressors are detected via different fluorescence channels: one in the cytoplasm and one in the nucleus." However, several aspects were unclear:

### Uncertainties
- **Which color channel corresponds to which protein?** The intensity images are RGB (3-channel) but the mapping between RGB channels and the two fluorescent proteins (nuclear vs cytoplasmic) is not specified.
  - **Assumption made**: Green channel = nuclear repressor, Red channel = cytoplasmic repressor
  - This is a common convention but should be verified experimentally

- **Are the proteins truly nuclear vs cytoplasmic, or are they just differently localized?** The CLAUDE.md mentions "one in the nucleus" but bacterial cells don't have nuclei.
  - **Assumption made**: "Nuclear" likely means "localized to nucleoid region" or possibly a misnomer. Treated as two distinct cellular compartments.

- **How to handle the RGB channels?** Are both fluorophores present in all channels with different intensities, or is each fluorophore exclusively in one channel?
  - **Assumption made**: Each fluorophore is primarily in one channel, with minimal crosstalk

## 2. Cell Segmentation

### Challenge
Phase contrast images need to be segmented to identify individual bacterial cells, but the optimal segmentation approach was unclear.

### Uncertainties
- **Segmentation algorithm choice**: Otsu thresholding is a simple approach, but may not be optimal for densely packed or dividing cells
  - **Implementation**: Used Otsu + morphological operations + border clearing
  - **Better alternatives might include**: Watershed segmentation, deep learning-based segmentation, or manual verification

- **Cell boundaries**: Bacterial cells can be very small and close together, making accurate boundary detection challenging
  - **Assumption made**: Set minimum cell area to 50 pixels, which may exclude small cells or include artifacts

- **Dividing cells**: How to handle cells that are in the process of dividing?
  - **Current approach**: They may be segmented as single cells or as separate cells depending on constriction state

## 3. Cell Tracking Across Time

### Challenge
Individual cells need to be tracked across the 35+ hour time-series to get single-cell trajectories.

### Uncertainties
- **Cell divisions**: Cells divide during the experiment, making lineage tracking complex
  - **Current implementation**: Simple overlap-based tracking that doesn't explicitly handle divisions
  - **Limitation**: When a cell divides, the two daughter cells may be assigned to different tracks or one may be lost

- **Cells leaving/entering field of view**: Cells can move in/out of the imaging area
  - **Current approach**: New tracks are created for cells that appear
  - **Limitation**: Lost tracks when cells leave the field

- **Tracking robustness**: The simple overlap-based method may fail when cells move significantly between frames
  - **Better alternatives**: More sophisticated tracking algorithms (Hungarian algorithm, Kalman filtering, or specialized bacterial tracking software)

## 4. Fluorescence to Protein Conversion

### Challenge
Converting pixel intensities to protein quantities requires careful calibration.

### Uncertainties
- **Conversion factor validity**: The docs mention a 1e7 conversion factor between pixel intensities and calibration arbitrary units
  - **Question**: Was this factor determined under the same imaging conditions (exposure, gain, etc.) as the experimental images?
  - **Assumption made**: The conversion factor is directly applicable

- **Background subtraction**: The calibration docs mention "background subtracting the total fluorescence from each cell in the absence of induction"
  - **Current implementation**: No explicit background subtraction applied to experimental images
  - **Concern**: This could lead to overestimation of protein levels

- **Per-cell vs per-population calibration**: The calibration was done on bulk lysates (total fluorescence from all cells)
  - **Question**: Is it valid to apply this to individual cell measurements?
  - **Concern**: Cell-to-cell variability in fluorophore maturation, expression, etc.

- **Fluorophore saturation**: At high protein levels, fluorescence may not be linear with concentration
  - **Current implementation**: Assumes linear relationship via interpolation with extrapolation
  - **Risk**: May overestimate very high or underestimate very low concentrations

## 5. ODE Model Specification

### Challenge
The exact form of the Repressilator ODE model to use for parameter inference was not specified.

### Uncertainties
- **Model structure**: The classical Repressilator has 6 state variables (3 mRNAs, 3 proteins)
  - **Assumption made**: Used standard Hill function repression with cyclic topology
  - **Question**: Is this the appropriate model for this specific system?

- **Protein dimerization/oligomerization**: Transcription repressors often function as dimers or higher-order oligomers
  - **Current model**: Does not explicitly model oligomerization
  - **Limitation**: May not capture cooperative binding effects accurately

- **Third unobserved protein**: One repressor has no fluorescence tag
  - **Current approach**: Model includes it in dynamics but it's not directly constrained by data
  - **Concern**: Parameters related to the third protein may be poorly constrained

- **Initial conditions**: Where do cells start in the oscillation cycle?
  - **Current approach**: Uses fixed initial conditions
  - **Better approach**: Could treat initial conditions as additional parameters to infer

## 6. Parameter Identifiability

### Challenge
With only 2 of 3 proteins observed, and potential model complexity, parameters may not be uniquely identifiable.

### Uncertainties
- **Structural identifiability**: Can all parameters be uniquely determined from the observed data?
  - **Concern**: Multiple parameter combinations might produce similar observed dynamics
  - **Current approach**: Uses broad uniform priors, but no identifiability analysis performed

- **Practical identifiability**: Even if theoretically identifiable, are parameters well-constrained by noisy experimental data?
  - **Suggestion**: Should examine posterior distributions for correlations and wide uncertainties

## 7. Experimental Details Not Specified

### Missing Information
- **Imaging interval**: Images are at 15-minute intervals, but is this adequate for capturing fast oscillations?
  - If oscillation period is ~30-60 minutes (as mentioned in some Repressilator literature), this should be okay
  - But phase sampling could be an issue

- **Growth medium and conditions**: Temperature, nutrients, induction state
  - These affect protein expression and degradation rates

- **Cell age/lifecycle**: Are cells synchronized? Are we imaging exponential growth phase?
  - This affects which model (growing vs non-growing) is appropriate

- **Image acquisition settings**: Exposure time, gain, microscope setup
  - Important for quantitative fluorescence analysis

## 8. Computational Challenges

### Performance Concerns
- **MCMC convergence**: Bayesian inference with PINTS can be computationally expensive
  - **Current setting**: 1000 iterations per cell (may be insufficient for convergence)
  - **Suggestion**: Should check convergence diagnostics (Gelman-Rubin, trace plots)

- **Processing time**: With 144 timepoints and multiple cells per timepoint, analyzing all cells could take hours/days
  - **Consideration**: May need to parallelize or use more efficient inference methods

## 9. Data Quality Issues

### Potential Problems Not Addressed
- **Photobleaching**: Fluorophores can bleach over long imaging periods
  - **Current approach**: No correction applied
  - **Risk**: Systematic decrease in apparent protein levels over time

- **Autofluorescence**: Background cellular autofluorescence contributes to signal
  - **Current approach**: No explicit autofluorescence subtraction
  - **Mitigation**: Could use cells without fluorescent proteins as controls

- **Out-of-focus cells**: Some cells may be partially out of the focal plane
  - **Current approach**: No focus quality filtering
  - **Suggestion**: Could filter cells based on sharpness or contrast

## 10. Validation and Quality Control

### Missing Components
- **No ground truth**: Without independent protein measurements, can't validate fluorescence conversion

- **No synthetic data testing**: Haven't tested parameter inference on simulated data with known parameters
  - **Suggestion**: Should generate synthetic oscillating data and verify parameter recovery

- **No model comparison**: Only implemented one ODE model variant
  - **Alternative models**: Could include noise in promoter switching, different cooperativity assumptions, etc.

## 11. Software Dependencies and Compatibility

### Technical Concerns
- **PINTS version compatibility**: The package requires `pints>=0.5.0` but API may change

- **Image file format assumptions**: Assumes PNG format at 512x512 resolution
  - **Limitation**: Won't work if image format or resolution changes

- **Memory usage**: Loading all images into memory could be problematic for very large datasets
  - **Suggestion**: Could implement lazy loading or chunked processing

## Recommendations for Future Work

1. **Validate channel assignments** by checking against microscope metadata or testing with single-protein controls

2. **Implement proper background subtraction** using dark frames or non-fluorescent control cells

3. **Improve cell tracking** with specialized bacterial tracking software or manual curation

4. **Perform identifiability analysis** to determine which parameters can be reliably estimated

5. **Test on synthetic data** to validate the inference pipeline

6. **Implement convergence diagnostics** for MCMC chains

7. **Add photobleaching correction** using reference regions or exponential decay models

8. **Create visualization tools** for inspecting segmentation quality and fit quality

9. **Add statistical tests** for detecting oscillations vs noise

10. **Consult original paper** (docs/35002125.pdf) for specific implementation details used in published work

## Summary

The implementation makes several reasonable assumptions to create a functional pipeline, but many details would need experimental validation or clarification from domain experts. The most critical uncertainties are:

1. RGB channel to protein mapping
2. Fluorescence-to-protein conversion accuracy
3. Appropriate ODE model structure
4. Parameter identifiability with partial observability
5. Cell tracking through divisions

These should be addressed before drawing biological conclusions from the analysis results.
