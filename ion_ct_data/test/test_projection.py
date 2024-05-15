import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import scipy.sparse
from ion_ct_data.straight_projection import Projector
from ion_ct_data.patient_data import PatientDataLoader, PatientCTData, PATIENTS
from ion_ct_data.helpers.plotting import plot_comparison, plot_projections
import torch

def generate_system_matrix(shape, angles):
    system_matrices_angles = {}
    for i, theta in enumerate(angles):
        system_matrix = np.zeros(shape=(np.prod(shape), shape[0]))
        for row_index in range(shape[0]):
            image_zeros = np.zeros(shape)
            image_zeros[row_index, :] = 1
            image_rotated = rotate(image_zeros, theta, reshape=False, order=1)
            image_row = image_rotated.reshape(np.prod(shape), order='F')
            system_matrix[:, row_index] = image_row
        system_matrix = scipy.sparse.csc_matrix(system_matrix)
        system_matrices_angles[theta] = system_matrix
    return system_matrices_angles

def forward_projection(system_matrix, image):
    return system_matrix @ image.flatten(order='F')

def back_projection(system_matrix, sinogram):
    return (system_matrix.transpose() @ sinogram).reshape(system_matrix.shape[1], system_matrix.shape[1], order='F')


if __name__ == '__main__':

    patient_name = PATIENTS[0]
    patient_loader = PatientDataLoader(patient_name)
    patient = patient_loader.create_patient_data()

    angles = np.linspace(0, 180, 5, endpoint=False)

    projector = Projector(slice_shape=patient.slice_shape, angles=angles)
    projections = projector.generate(patient)

    #projections to numpy array
    projections_array = np.stack([val for val in projections.values()])