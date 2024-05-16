# Ion CT Data Simulation and Handling

This project demonstrates the handling and generation of simulated ion CT data, useful for research and development in the field of medical imaging. The codebase includes modules for data representation, transformation, and projection, as well as automated data augmentation.

## Table of Contents
- [Project Structure](#project-structure)
- [License](#license)

## Project Structure

- **patient_data.py**: Contains the `PatientData` class representing CT data. It encapsulates all relevant attributes and methods to manage patient-specific imaging data.
- **straight_projection.py**: Defines the `Projector` class, generating CT projections from CT images as simulated sonogram data.
- **sample_data.py**: Script for sampling the data, integrating various modules to create a seamless workflow from raw data to processed outputs.
- **data_transform.py**: Includes methods for applying geometric and intensity transformations to CT images, essential for data augmentation and robustness testing.
- **deformation_sampling.py**: Contains the `Sampler` class which samples parameters for transformations to automatically generate augmented data.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This version combines the project structure and module explanations into a single cohesive section, providing clarity without redundancy.
