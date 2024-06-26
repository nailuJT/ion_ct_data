import matplotlib.pyplot as plt
import numpy as np
import os
from warnings import warn
from deformation_sampling import GaussianParameterSampler, transform_ct
from straight_projection import Projector
import json
from patient_data import PatientCTData, PatientDataLoader, PATIENTS
from tqdm import tqdm

SLICES = 76
DATA_DIR = "/project/med6/IONCT/julian_titze/data/medical"
if not os.path.exists(DATA_DIR):
    warn(f"Folder {DATA_DIR} does not exist creating it now")
    os.makedirs(DATA_DIR, exist_ok=True)


def sample_data():
    """
    :saves:
    transformed_ionct_chunk_{patient_name}_{i}.npy:numpy.array[slices, x, y]:  label for machine learning
    ct_chunk_{patient_name}_{i}.npy:numpy.array[slices, x, y]:  input for machine learning
    projection_angles_chunk_{patient_name}_{i}.npy:numpy.array[n_angles, slices, x]:  projection angles for machine learning
    system_matrices.pt:torch.tensor[n_angles, n_slices, n_pixels]:  system matrices for projection
    """

    config_dict = json.load(open("sampling_configconfig.json", "r"))
    slices_centers = [-30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    mu_means = [[[center, 0., 0.], [center, 0., 0.], [center, 0., 0.]] for center in slices_centers]
    configs = [config_dict.copy() for _ in range(len(slices_centers))]
    for i, mu_mean in enumerate(mu_means):
        configs[i]["mu_mean"] = mu_mean

    samplers = [GaussianParameterSampler.from_dict(config) for config in configs]

    patient_names = PATIENTS.copy()

    patient_loaders = [PatientDataLoader(patient_name) for patient_name in patient_names]
    patients = [patient_loader.create_patient_data() for patient_loader in patient_loaders]

    n_angles = 8
    angles = np.linspace(0, 180, n_angles)

    projector = Projector(angles=angles, slice_shape=(patients[0].slice_shape))
    _stacked_norm = projector.save_stacked_system_matrices(
        os.path.join(DATA_DIR, "system_matrices_norm.pt"), normalize=True)
    _stacked_sys = projector.save_stacked_system_matrices(
        os.path.join(DATA_DIR, "system_matrices.pt"), normalize=False)

    for patient_index, patient in enumerate(patients):
        print(f"Processing patient {patient_index + 1} of {len(patients)}")
        for sampler_index, sampler in enumerate(samplers):
            print(f"\tDeformation {sampler_index + 1} of {len(samplers)}")
            transformed_patient, vector_field = transform_ct(patient, sampler.sample(inversion="none"))

        projection_angles = projector.generate(transformed_patient)
        projection_angles = np.array([val for val in projection_angles.values()])

        chunk_size = 5
        offset = 8

        # Number of chunks
        n_chunks = (patient.n_slices - offset) // chunk_size

        for i in range(n_chunks):
            print(f"\t\tProcessing chunk {i + 1} of {n_chunks}")
            # Get the start and end index for the current chunk
            start = i * chunk_size + offset
            end = start + chunk_size

            # chunk the data
            ionct_chunk = patient.ion_ct[start:end, :, :]
            patient_ct = patient.ct[start:end, :, :]
            transformed_ionct_chunk = transformed_patient.ion_ct[start:end, :, :]
            mask_chunk = patient.mask[start:end, :, :]
            projection_angles_chunk = projection_angles[:, start:end, :]
            vector_field_chunk = vector_field[start:end]

            # check the shapes
            assert ionct_chunk.shape == (chunk_size, patient.slice_shape[0], patient.slice_shape[1])
            assert patient_ct.shape == (chunk_size, patient.slice_shape[0], patient.slice_shape[1])
            assert transformed_ionct_chunk.shape == (chunk_size, patient.slice_shape[0], patient.slice_shape[1])
            assert projection_angles_chunk.shape == (n_angles, chunk_size, patient.slice_shape[1])

            # Save the chunk data to a numpy file
            np.save(os.path.join(DATA_DIR, f"ionct_chunk_{patient.name}_{i}.npy"), ionct_chunk)
            np.save(os.path.join(DATA_DIR, f"ct_chunk_{patient.name}_{i}.npy"), patient_ct)
            np.save(os.path.join(DATA_DIR, f"transformed_ionct_chunk_{patient.name}_{i}.npy"), transformed_ionct_chunk)
            np.save(os.path.join(DATA_DIR, f"mask_chunk_{patient.name}_{i}.npy"), mask_chunk)
            np.save(os.path.join(DATA_DIR, f"angles_chunk_{patient.name}_{i}.npy"), projection_angles_chunk)
            np.save(os.path.join(DATA_DIR, f"vector_field_chunk_{patient.name}_{i}.npy"), vector_field_chunk)



if __name__ == '__main__':
    sample_data()


