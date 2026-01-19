# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository analyzes time-series fluorescence microscopy images of bacterial cells expressing the Repressilator (see `docs/35002125.pdf`). Two of the three transcription repressors are detected via different fluorescence channels: one in the cytoplasm and one in the nucleus. One of the proteins is not associated with a flouresence signal. 

**Pipeline steps:**
1. Read image files in time order
2. Extract fluorescence values for each protein and each cell. Cell division events do not need to be modelled. 
3. Convert the pixel values to protein quantities using the calibration values provided for each protein in docs/
3. Determine ODE model parameters for each cell using PINTS (Bayesian inference)


