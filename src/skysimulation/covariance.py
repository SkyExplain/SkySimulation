import numpy as np
from scipy.stats import skewnorm

def covariance_asymmetric_errors(Dl, err_neg, err_pos, num_samples):
    """
    Simulate samples from asymmetric errors using a skew-normal distribution.
    """
    mean = Dl
    std_dev = (err_pos + err_neg) / 2
    skew_param = (err_pos - err_neg) / (err_pos + err_neg) * 10
    return skewnorm.rvs(a=skew_param, loc=mean, scale=std_dev, size=num_samples)

def add_noise_spectrum(spectrum, covariance_matrix, seed0):
    """
    Adds noise to a given power spectrum using a multivariate normal distribution.
    """
    rng = np.random.default_rng(seed0)
    noisy = rng.multivariate_normal(spectrum[: len(covariance_matrix)], covariance_matrix, 1)
    return noisy[0]
