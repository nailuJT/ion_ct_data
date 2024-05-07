# ion_ct_data
This project aims to be a reference for data augmentation as well as simple projections for generating synthetic pretraining for machine learning-based ion image registration.

## Files and Modules
### CT data handling and projection
- straight_projection.py
  This is the entry point for loading CT data into a dataclass. This class is tailored for the dataclass I work with, but it can be generalized quite simply. The data can then be handled by a projector class to generate straight projection sinograms.
### Data augmentation and deformation sampling
- data_transform.py
  This file holds all the methods for introducing Gaussian deformation onto the data to simulate anatomical changes, as well as to make trained machine learning algorithms more robust to data variation.
- deformation_sampling.py
  This module contains the GaussianParameterSampler class for sampling deformation parameters given suited configurations. This is used to generate data automatically.
- sample_data.py
  This could be considered an example main script for using this package, loading the data, sampling deformations, generating projections, and subsequently storing these in chunked form as training examples.
