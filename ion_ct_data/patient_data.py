from abc import ABC, abstractmethod
import os
import numpy as np
from scipy.interpolate import interp1d


class AbstractDataLoader(ABC):
    @abstractmethod
    def load_ct(self):
        pass

    @abstractmethod
    def load_mask(self):
        pass

    @abstractmethod
    def load_rsp_accurate(self, deviation, error):
        pass

    @abstractmethod
    def create_patient_data(self) -> 'PatientCTData':
        pass


class PatientDataLoader(AbstractDataLoader):
    MAGNETIC_DEVIATION = {
        0.05: '5',
        0.2: '20'
    }
    HU_ORIGINAL = np.array([-1400, -1000, -800, -600, -400, -200, 0, 200, 400, 600, 800, 1400])

    def __init__(self, patient_name):
        self.patient_name = patient_name
        self.base_path = '/project/med2/IONCT/Data/ML'
        self.recerence_ct_path = os.path.join(self.base_path, 'PrimalDual', 'ReferenceCTs')
        self.rsp_accurate_path = os.path.join(self.base_path, 'DeepBackProj', 'calibs')

    def load_ct(self):
        ct_filename = f'20231011_analytical_{self.patient_name}_1mm3mm1mm.npy'
        ct_path = os.path.join(self.recerence_ct_path, ct_filename)
        return np.load(ct_path)

    def load_mask(self):
        mask_filename = f'20231011_analytical_{self.patient_name}_1mm3mm1mm_mask.npy'
        mask_path = os.path.join(self.rsp_accurate_path, mask_filename)
        return np.load(mask_path)

    def load_rsp_accurate(self, deviation=0.05, error='mixed'):
        magn_deviation = self.MAGNETIC_DEVIATIONS[deviation]
        rsp_accurate_files = []

        n_slices = self.get_n_slices()

        for slice in range(n_slices):
            rsp_relative_path = f"pretraining_{self.patient_name}_calibs_acc_{magn_deviation}{error}/" \
                                f"RSP_accurate_slice{slice + 1}.npy"
            rsp_accurate_path = os.path.join(self.rsp_accurate_path, rsp_relative_path)
            rsp_accurate_files.append(np.load(rsp_accurate_path))

        return np.array(rsp_accurate_files)

    def get_n_slices(self):
        ct = self.load_ct()
        return ct.shape[0]

    def create_patient_data(self) -> 'PatientCTData':
        ct = self.load_ct()
        mask = self.load_mask()
        rsp_accurate = self.load_rsp_accurate()
        return PatientCTData(
            patient_name=self.patient_name,
            ct=ct,
            mask=mask,
            rsp_accurate=rsp_accurate,
            n_slice_block=1,
            magnetic_deviation=0.05,
            error='mixed',
            hu_original=self.HU_ORIGINAL
        )


class PatientCTData:
    def __init__(self, patient_name, hu_original, ct, mask, rsp_accurate, n_slice_block=1, magnetic_deviation=0.05, error='mixed'):
        self.name = patient_name
        self.ct = ct.transpose(1, 0, 2)
        self.mask = mask.transpose(1, 0, 2)
        self.rsp_accurate = rsp_accurate
        self.n_slice_block = n_slice_block
        self.magnetic_deviation = magnetic_deviation
        self.error = error
        self.n_slices = self.ct.shape[0]
        self.hu_original = hu_original

    @staticmethod
    def _iterate_over_blocks(data, block_size):
        num_blocks = data.shape[0] // block_size - 1
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            yield data[start:end, :, :]

    def iterate_over_blocks(self, data_key, block_size):
        data = getattr(self, data_key)
        return self._iterate_over_blocks(data, block_size)

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
    def ion_ct(self):
        ion_ct = np.empty_like(self.ct)
        for i in range(self.n_slices):
            renormalization = interp1d(self.hu_original, self.rsp_accurate[i, :], kind='linear')
            ion_ct[i] = renormalization(self.ct[i])
        return ion_ct

    def apply_mask(self, image):
        return image * self.mask

