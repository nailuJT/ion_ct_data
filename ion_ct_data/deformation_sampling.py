import numpy as np
from data_transform import GaussianParameters, apply_gaussian_transform3d
import copy


class GaussianParameterSampler:
    def __init__(self, alpha_mean, alpha_std, mu_mean, mu_std, sigma_mean, sigma_std, rotation_mean, rotation_std, dimension=3):
        """
        Initialize the GaussianParameterSampler with mean and standard deviation parameters for
        alpha, mu, sigma, and rotation.

        :param alpha_mean: Mean values for strength of deformation.
        :param alpha_std: Standard deviations for strength of deformation.
        :param mu_mean: Mean values for deformation center.
        :param mu_std: Standard deviations for deformation center.
        :param sigma_mean: Mean values for deformation spread.
        :param sigma_std: Standard deviations for deformation spread.
        :param rotation_mean: Mean values for rotation directions.
        :param rotation_std: Stndard deviations for rotation directions.
        :param dimension: Dimensionality of CT images.
        """
        self.alpha_mean = np.array(alpha_mean)
        self.alpha_std = np.array(alpha_std)
        self.mu_mean = np.array(mu_mean)
        self.mu_std = np.array(mu_std)
        self.sigma_mean = np.array(sigma_mean)
        self.sigma_std = np.array(sigma_std)
        self.rotation_mean = np.array(rotation_mean)
        self.rotation_std = np.array(rotation_std)
        self.dimension = np.array(dimension)

        self.correlation_deformation = 0.1
        self.correlation_directions = 0.8

    @staticmethod
    def _construct_covariance_matrix(dimension, correlation_deformation, correlation_directions, standard_deviations):
        """
        Construct a covariance matrix given the dimension, correlations, and standard deviations.

        :param dimension: Dimensionality of the data.
        :param correlation_deformation: Correlation value for deformation.
        :param correlation_directions: Correlation value for directions.
        :param standard_deviations: Standard deviations for normalization.
        :return: Covariance matrix.
        """
        if not np.isscalar(standard_deviations):
            standard_deviations = np.array(standard_deviations).flatten()

        correlation_matrix_deformation = np.full((dimension, dimension), correlation_deformation)
        np.fill_diagonal(correlation_matrix_deformation, 1)
        correlation_matrix_deformation_full = np.kron(np.eye(dimension), correlation_matrix_deformation)

        correlation_matrix_directions = np.eye(dimension) * correlation_directions
        correlation_matrix_directions_full = np.kron(np.ones(dimension) - np.eye(dimension), correlation_matrix_directions)

        covariance_matrix = correlation_matrix_deformation_full + correlation_matrix_directions_full
        normalization = np.outer(standard_deviations, standard_deviations)
        covariance_matrix *= normalization

        return covariance_matrix

    def covariance_matrix(self, standard_deviations=1.0):
        """
        :param standard_deviations: Standard deviations for normalization.
        :return: Covariance matrix.
        """
        return self._construct_covariance_matrix(self.dimension, self.correlation_deformation,
                                                 self.correlation_directions, standard_deviations)

    def sample(self, inversion="all", inversion_chance=0.5):
        """
        Sample Gaussian parameters for data augmentation.

        :param inversion: Inversion setting ('none', 'all', 'indep').
        :param inversion_chance: Probability of inversion.
        :return: Dictionary with sampled parameters.
        """
        if inversion == "none":
            inversion_vector = np.zeros(self.dimension)
        elif inversion == "all":
            inversion_vector = np.ones(self.dimension) * np.random.choice([-1, 1], p=[1 - inversion_chance, inversion_chance])
        elif inversion == "indep":
            inversion_vector = np.random.choice([-1, 1], size=self.dimension, p=[1 - inversion_chance, inversion_chance])
        else:
            raise ValueError("Invalid inversion setting. Choose from 'none', 'all', 'indep'.")

        alpha_directions = np.random.normal(self.alpha_mean.flatten(), self.alpha_std, self.dimension)
        alpha_directions *= inversion_vector

        mu_directions = np.random.multivariate_normal(self.mu_mean.flatten(), self.covariance_matrix(self.mu_std))
        mu_directions = np.reshape(mu_directions, (self.dimension, self.dimension))

        sigma_directions = np.random.multivariate_normal(self.sigma_mean.flatten(), self.covariance_matrix(self.sigma_std))
        sigma_directions = np.reshape(sigma_directions, (self.dimension, self.dimension))

        rotation_directions = np.random.normal(self.rotation_mean.flatten(), self.rotation_std, (self.dimension, self.dimension))

        return {"alpha_dirs": alpha_directions, "mu_dirs": mu_directions, "sigma_dirs": sigma_directions, "rotation_dirs": rotation_directions}

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()

    @classmethod
    def from_config(cls, config):
        """
        :param config: Configuration dictionary.
        :return: GaussianParameterSampler instance.
        """
        return cls(config['alpha_mean'], config['alpha_std'], config['mu_mean'], config['mu_std'],
                   config['sigma_mean'], config['sigma_std'], config['rotation_mean'], config['rotation_std'])

    @classmethod
    def from_dict(cls, config_dict):
        """
        :param config_dict: Configuration dictionary.
        :return: GaussianParameterSampler instance.
        """
        return cls(config_dict['alpha_mean'], config_dict['alpha_std'], config_dict['mu_mean'], config_dict['mu_std'],
                   config_dict['sigma_mean'], config_dict['sigma_std'], config_dict['rotation_mean'], config_dict['rotation_std'])


def transform_ct(patient, gaussian_parameters, normalize=True):
    """
    Samples a Gaussian transform and applies it to a projection.

    :param patient: Patient object with CT data.
    :param gaussian_parameters: Gaussian parameters for transformation.
    :param normalize: Flag to normalize data (default is True).
    :return: Transformed patient object and vector field.
    """
    patient = copy.deepcopy(patient)
    ct_original = patient.ct
    mask_original = patient.mask

    if normalize:
        # TODO: implement normalization based on voxel size and image size
        pass

    ct_transformed, vector_field = apply_gaussian_transform3d(ct_original, **gaussian_parameters)
    mask_transformed, _ = apply_gaussian_transform3d(mask_original, **gaussian_parameters)
    patient.ct = ct_transformed
    patient.mask = mask_transformed

    return patient, vector_field


def test_sampler():
    """
    Test function for GaussianParameterSampler.
    """
    config_dict = {
        "alpha_mean": [300, 1500, 1500],
        "alpha_std": [400, 2000, 2000],
        "mu_mean": [[0.0, 0.0, 12.0], [10.0, 0.0, 12.0], [0.0, 0.0, 12.0]],
        "mu_std": [[40.0, 40.0, 5.0], [40.0, 40.0, 5.0], [40.0, 40.0, 5.0]],
        "sigma_mean": [[25, 25, 12], [25, 25, 12], [25, 25, 12]],
        "sigma_std": [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
        "rotation_mean": [0.0, 0.0, 0.0],
        "rotation_std": [40.0, 40.0, 40.0]
    }

    sampler = GaussianParameterSampler.from_dict(config_dict)
    print(sampler.sample())

if __name__ == '__main__':
    test_sampler()
