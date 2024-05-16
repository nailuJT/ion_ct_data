# Ion CT Data Simulation and Handling

This project demonstrates the handling and generation of simulated ion CT data, useful for research and development in the field of medical imaging. The codebase includes modules for data representation, transformation, and projection, as well as automated data augmentation.

## Table of Contents
- [Project Structure](#project-structure)
- [Modules](#modules)
- [License](#license)


## Project Structure

- **patient_data.py**: Contains the `PatientData` class representing CT data.
- **straight_projection.py**: Defines the `Projector` class, generating CT projections from CT images as simulated sonogram data.
- **sample_data.py**: Script for sampling the data.
- **data_transform.py**: Includes methods for applying caution transformations to images.
- **deformation_sampling.py**: Contains the `Sampler` class which samples parameters for transformations to automatically generate augmented data.

## Modules

### patient_data.py

Defines a class to represent CT data, encapsulating all relevant attributes and methods to manage patient-specific imaging data.

### straight_projection.py

Implements a projector class to simulate CT projections, converting 3D CT images into 2D sonogram data for analysis and further processing.

### sample_data.py

Handles the sampling of data, integrating various modules to create a seamless workflow from raw data to processed outputs.

### data_transform.py

Provides a suite of functions for applying geometric and intensity transformations to CT images, essential for data augmentation and robustness testing.

### deformation_sampling.py

Implements a sampler class to automate the generation of transformation parameters, enabling large-scale data augmentation for training and validation of machine learning models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to adjust any sections to better fit your needs or add additional information specific to your project. This README should give potential employers and collaborators a clear overview of your project and how to use it.
