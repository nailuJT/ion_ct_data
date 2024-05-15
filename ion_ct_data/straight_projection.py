"""
This module contains the PatientCT class which is used to handle patient CT data.

The data should be structured in the following way:

/project
└── med2
    └── IONCT
        └── Data
            └── ML
                ├── DeepBackProj
                │   ├── 20231006_{patient_name}_1mm3mm1mm_test_slices.npy
                │   └── calibs
                │       └── pretraining_{patient_name}_calibs_acc_5mixed
                │           └── RSP_accurate_slice0.npy
                └── PrimalDual
                    └── ReferenceCTs
                        └── 20231011_analytical_{patient_name}_1mm3mm1mm.npy
                        └── 20231011_analytical_{patient_name}_1mm3mm1mm_mask.npy

Where {patient_name} should be replaced with the name of the patient.

The PatientCT class provides methods to load and process the CT data, including generating system matrices and projections.
"""

import scipy
import numpy as np
import os
from scipy.interpolate import interp1d
import torch
from scipy.ndimage import rotate
from tqdm.auto import tqdm


class Projector:
    """
    Class to generate projections.
    """

    def __init__(self, slice_shape, angles, voxel_size=(1.0, 1.0, 1.0)):
        self.angles = angles
        self.slice_shape = slice_shape
        self.system_matrices = self._calculate_system_matrix(angles, slice_shape)
        self.voxel_size = voxel_size

    def generate(self, patient, save_path=None, normalize=True):
        """
        Generates the projections for a given patient and a given set of system matrices.
        """

        projection_angles = {}

        for angle, system_matrix in tqdm(self.system_matrices.items()):

            projection_angle = np.zeros((patient.n_slices, patient.shape[1]))

            enumerated_cts = enumerate(zip(patient.ion_ct,
                                           patient.mask))

            for slice, (ion_ct_block, mask_image) in enumerated_cts:

                ion_ct_masked = ion_ct_block * mask_image
                system_matrix_masked = system_matrix.multiply(mask_image.flatten()[:, np.newaxis])

                if normalize:
                    system_matrix_masked = self.normalize_system_matrix(system_matrix_masked)

                system_matrix_masked = system_matrix_masked.tocoo()
                indices = np.vstack((system_matrix_masked.row, system_matrix_masked.col))

                indices_tensor = torch.LongTensor(indices)
                system_matrix_masked_tensor = torch.FloatTensor(system_matrix_masked.data)

                sys_coo_tensor = torch.sparse.FloatTensor(indices_tensor, system_matrix_masked_tensor,
                                                          torch.Size(system_matrix_masked.shape))

                projection_angle_slice = system_matrix_masked.transpose().dot(ion_ct_masked.flatten())

                if save_path is not None:
                    torch.save(sys_coo_tensor,
                               save_path + '/sysm_slice' + str(slice) + '_angle' + str(int(angle)) + '.pt')
                    np.save(save_path + '/proj_slice' + str(slice) + '_angle' + str(int(angle)) + '.npy',
                            projection_angle_slice)

                projection_angle[slice] = projection_angle_slice

            # projection of angles we project on
            projection_angles[angle] = projection_angle

        return projection_angles

    @staticmethod
    def _calculate_system_matrix(angles, slice_shape):
        """
        Generates a system matrix for a given shape and a given set of angles.

        :param shape: Shape of the image.
        :param angles: Angles for which the system matrix should be generated.
        :return: Dictionary of system matrices for each angle.
        """

        system_matrices_angles = {}

        for i, theta in tqdm(enumerate(angles)):

            system_matrix = np.zeros(shape=(np.prod(slice_shape), slice_shape[0]))

            for row_index in range(slice_shape[0]):
                image_zeros = np.zeros(slice_shape)
                image_zeros[row_index, :] = 1
                image_rotated = rotate(image_zeros, theta, reshape=False, order=1)

                image_row = image_rotated.reshape(np.prod(slice_shape))
                system_matrix[:, row_index] = image_row

            system_matrix = scipy.sparse.csc_matrix(system_matrix)
            system_matrices_angles[theta] = system_matrix

        return system_matrices_angles

    @staticmethod
    def _system_matrices_to_tensor(system_matrices):
        """
        Converts the system matrices to sparse tensors.
        """
        system_matrices_sparse = {}
        for angle, system_matrix in system_matrices.items():
            system_matrix = system_matrix.tocoo()
            indices = np.vstack((system_matrix.row, system_matrix.col))
            indices_tensor = torch.LongTensor(indices)
            system_matrix_tensor = torch.FloatTensor(system_matrix.data)
            system_matrices_sparse[angle] = torch.sparse.FloatTensor(indices_tensor, system_matrix_tensor,
                                                                     torch.Size(system_matrix.shape))

        return system_matrices_sparse

    @staticmethod
    def _stack_system_matrices_tensor(system_matrices_tensor, angles):
        """
        Stacks the system matrices.
        """
        return torch.stack([system_matrices_tensor[angle] for angle in angles])

    def save_stacked_system_matrices(self, path, normalize=False):
        """
        Saves the stacked system matrices to a given path.
        """
        if normalize:
            system_matrices = self.system_matrices
            for angle, system_matrix in system_matrices.items():
                system_matrices[angle] = self.normalize_system_matrix(system_matrix)
        else:
            system_matrices = self.system_matrices

        system_matrices_tensor = self._system_matrices_to_tensor(system_matrices)
        stacked_system_matrices = self._stack_system_matrices_tensor(system_matrices_tensor, self.angles)
        torch.save(stacked_system_matrices, path)
        return stacked_system_matrices

    @staticmethod
    def normalize_system_matrix(system_matrix):
        """
        Normalizes a system matrix.
        """
        normalization_sum = system_matrix.sum(0)
        indexes_zeros = np.where(normalization_sum == 0)[1]
        normalization_sum[0, indexes_zeros] = 1
        system_matrix = system_matrix.multiply(1. / normalization_sum)
        return system_matrix

