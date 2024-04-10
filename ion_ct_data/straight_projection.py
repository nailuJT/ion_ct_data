"""
This module contains the PatientCT class which is used to handle patient CT data.

The data should be structured in the following way:

/project
└── med2
    └── Ines.Butz
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

PATIENTS = ['male1', 'female1', 'male2','female2','male3', 'female3', 'male4','female4', 'male5', 'female5']

HU_ORIGINAL = np.array([-1400, -1000, -800, -600, -400, -200, 0, 200, 400, 600, 800, 1400])

class PatientCT:
    """
    Class to handle patient CT data.
    """

    BASE_PATH = '/project/med2/Ines.Butz/Data/ML'
    DEEP_BACK_PROJ_PATH = os.path.join(BASE_PATH, 'DeepBackProj')
    PRIMAL_DUAL_PATH = os.path.join(BASE_PATH, 'PrimalDual')
    REFERENCE_CTS_PATH = os.path.join(PRIMAL_DUAL_PATH, 'ReferenceCTs')
    RSP_ACCURATE_PATH = os.path.join(DEEP_BACK_PROJ_PATH, 'calibs')

    MAGNETIC_DEVIATIONS = {
        0.05: '5',
        0.2: '20'
    }

    def __init__(self, patient_name, n_slice_block=1, mode='train', magnetic_deviation=0.05, error='mixed'):

        self.mode = mode
        self.error = error
        self.magnetic_deviation = magnetic_deviation
        self.n_slice_block = n_slice_block

        self.name = patient_name
        self.ct = self._load_ct().transpose(1, 0, 2)
        self.mask = self._load_mask().transpose(1, 0, 2)
        self.slices = self._load_slices(mode=self.mode)
        self.rsp_accurate = self._load_rsp_accurate(deviation=self.magnetic_deviation,
                                                    error=self.error,)

    def _load_ct(self):
        ct_filename = f'20231011_analytical_{self.name}_1mm3mm1mm.npy'
        ct_path = os.path.join(self.REFERENCE_CTS_PATH, ct_filename)
        return np.load(ct_path)

    def _load_mask(self):
        mask_filename = f'20231011_analytical_{self.name}_1mm3mm1mm_mask.npy'
        mask_path = os.path.join(self.REFERENCE_CTS_PATH, mask_filename)
        return np.load(mask_path)

    def _load_slices(self, mode="train", offset=1):
        slice_filename = f'20231006_{self.name}_1mm3mm1mm_{mode}_slices.npy'
        slices_path = os.path.join(self.DEEP_BACK_PROJ_PATH, slice_filename)
        return np.load(slices_path) - offset

    def _load_rsp_accurate(self, deviation=0.05, error='mixed'):
        magn_deviation = self.MAGNETIC_DEVIATIONS[deviation]
        rsp_accurate_files = []

        for slice in range(self.n_slices):
            rsp_relative_path = f"pretraining_{self.name}_calibs_acc_{magn_deviation}{error}/" \
                                f"RSP_accurate_slice{slice+1}.npy"
            rsp_accurate_path = os.path.join(self.RSP_ACCURATE_PATH, rsp_relative_path)
            rsp_accurate_files.append(np.load(rsp_accurate_path))

        return np.array(rsp_accurate_files)

    def ct_blocks(self, block_size):
        num_blocks = self.ct.shape[0] // block_size -1
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            yield self.ct[start:end, :, :]

    def mask_blocks(self, block_size):
        num_blocks = self.mask.shape[0] // block_size -1
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            yield self.mask[start:end, :, :]

    def ion_ct_blocks(self, block_size):
        ion_ct = self.ion_ct
        num_blocks = ion_ct.shape[0] // block_size -1
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            yield ion_ct[ start:end, :, :]

    def save_ion_ct(self, path):
        np.save(path, self.ion_ct)

    def save_mask(self, path):
        np.save(path, self.mask)

    def save_ct(self, path):
        np.save(path, self.ct)

    @property
    def shape(self):
        return self.ct.shape

    @property
    def slice_shape(self):
        return self.ct.shape[1:]

    @property
    def n_slices(self):
        return self.ct.shape[0]

    @property
    def ion_ct(self, hu_original=HU_ORIGINAL):
        ion_ct = np.empty_like(self.ct)
        for i in range(self.n_slices):
            renormalization = interp1d(hu_original, self.rsp_accurate[i, :], kind='linear')
            ion_ct[i] = renormalization(self.ct[i])

        return ion_ct

    def apply_mask(self, image):
        """
        Applies the mask to an image.
        """
        return image * self.mask

class Projector:
    """
    Class to generate projections.
    """

    def __init__(self, slice_shape, angles, voxel_size=(1.0, 1.0, 1.0)):
        self.angles = angles
        self.slice_shape = slice_shape
        self.system_matrices = self.calculate_system_matrix()
        self._system_matrices_tensor = None
        self.voxel_size = voxel_size

    def calculate_system_matrix(self):
        """
        Generates a system matrix for a given shape and a given set of angles.

        :param shape: Shape of the image.
        :param angles: Angles for which the system matrix should be generated.
        :return: Dictionary of system matrices for each angle.
        """

        system_matrices_angles = {}

        for i, theta in tqdm(enumerate(self.angles)):

            system_matrix = np.zeros(shape=(np.prod(self.slice_shape), self.slice_shape[0]))

            for row_index in range(self.slice_shape[0]):

                image_zeros = np.zeros(self.slice_shape)
                image_zeros[row_index, :] = 1
                image_rotated = rotate(image_zeros, theta, reshape=False, order=1)

                image_row = image_rotated.reshape(np.prod(self.slice_shape), order='F')
                system_matrix[:, row_index] = image_row

            system_matrix = scipy.sparse.csc_matrix(system_matrix)
            system_matrices_angles[theta] = system_matrix

        return system_matrices_angles

    def system_matrices_to_tensor(self, normalize=False):
        """
        Converts the system matrices to sparse tensors.
        """
        if normalize:
            system_matrices = {angle: self.normalize_system_matrix(system_matrix) for angle, system_matrix in self.system_matrices.items()}
        else:
            system_matrices = self.system_matrices

        system_matrices_sparse = {}
        for angle, system_matrix in system_matrices.items():
            system_matrix = system_matrix.tocoo()
            indices = np.vstack((system_matrix.row, system_matrix.col))
            indices_tensor = torch.LongTensor(indices)
            system_matrix_tensor = torch.FloatTensor(system_matrix.data)
            system_matrices_sparse[angle] = torch.sparse.FloatTensor(indices_tensor, system_matrix_tensor,
                                                                     torch.Size(system_matrix.shape))

        return system_matrices_sparse

    def stack_system_matrices_tensor(self, system_matrices_tensor):
        """
        Stacks the system matrices.
        """
        return torch.stack([system_matrices_tensor[angle] for angle in self.angles])

    def save_stacked_system_matrices(self, path, normalize=False):
        """
        Saves the stacked system matrices to a given path.
        """
        system_matrices_tensor = self.system_matrices_to_tensor(normalize=normalize)
        stacked_system_matrices = self.stack_system_matrices_tensor(system_matrices_tensor)
        torch.save(stacked_system_matrices, path)

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
                system_matrix_masked = system_matrix.multiply(mask_image.flatten('F')[:, np.newaxis])

                if normalize:
                    system_matrix_masked = self.normalize_system_matrix(system_matrix_masked)

                system_matrix_masked = system_matrix_masked.tocoo()
                indices = np.vstack((system_matrix_masked.row, system_matrix_masked.col))

                indices_tensor = torch.LongTensor(indices)
                system_matrix_masked_tensor = torch.FloatTensor(system_matrix_masked.data)

                sys_coo_tensor = torch.sparse.FloatTensor(indices_tensor, system_matrix_masked_tensor,
                                                          torch.Size(system_matrix_masked.shape))

                projection_angle_slice = system_matrix_masked.transpose().dot(ion_ct_masked.flatten(order='F'))

                if save_path is not None:
                    torch.save(sys_coo_tensor,
                               save_path + '/sysm_slice' + str(slice) + '_angle' + str(int(angle)) + '.pt')
                    np.save(save_path + '/proj_slice' + str(slice) + '_angle' + str(int(angle)) + '.npy',
                            projection_angle_slice)

                projection_angle[slice] = projection_angle_slice

            projection_angles[angle] = projection_angle

        return projection_angles

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

def generate(system_matrices, ct_array, mask_array):
    """
    Generates the projections for a given patient and a given set of system matrices.
    """

    projection_angles = {}

    n_slices = ct_array.shape[0]

    for angle, system_matrix in system_matrices.items():

        projection_angle = np.zeros((n_slices, ct_array.shape[1]))

        enumerated_cts = enumerate(zip(ct_array,
                                       mask_array))

        for slice, (ion_ct_block, mask_image) in enumerated_cts:

            ion_ct_masked = ion_ct_block * mask_image
            system_matrix_masked = system_matrix.multiply(mask_image.flatten('F')[:, np.newaxis])

            system_matrix_masked = system_matrix_masked.tocoo()
            indices = np.vstack((system_matrix_masked.row, system_matrix_masked.col))

            indices_tensor = torch.LongTensor(indices)
            system_matrix_masked_tensor = torch.FloatTensor(system_matrix_masked.data)

            projection_angle_slice = system_matrix_masked.transpose().dot(ion_ct_masked.flatten(order='F'))

            projection_angle[slice] = projection_angle_slice

        projection_angles[angle] = projection_angle

    return projection_angles

