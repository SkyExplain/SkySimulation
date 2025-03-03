from scipy.stats import skewnorm
import numpy as np
import camb
import csv
import os

def covariance_asymmetric_errors(Dl, err_neg, err_pos, num_samples):
    """Example function for the simulation of a covariance matrix from asymmetric errors."""
    mean = Dl
    std_dev = (err_pos + err_neg) / 2  #std deviation
    skew_param = (err_pos - err_neg) / (err_pos + err_neg) * 10  #Skewness factor

    samples = skewnorm.rvs(a=skew_param, loc=mean, scale=std_dev, size=num_samples)
    return samples


def generate_camb_power_spectra(H0, ombh2, omch2, mnu, omk, tau,
                                As, ns, lmax, halofit_version):
    """
    Generates CMB power spectra using CAMB.

    Parameters:
        H0 (float): Hubble constant.
        ombh2 (float): Physical baryon density.
        omch2 (float): Physical cold dark matter density.
        mnu (float): Sum of neutrino masses.
        omk (float): Curvature density parameter.
        tau (float): Optical depth to reionization.
        As (float): Amplitude of scalar perturbations.
        ns (float): Scalar spectral index.
        lmax (int): Maximum multipole moment.
        halofit_version (str): Halofit version for nonlinear power spectra.

    Returns:
        tuple: (ell values, TT power spectrum, TE power spectrum, EE power spectrum)
    """

    params = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau,
                             As=As, ns=ns, lmax=lmax, halofit_version=halofit_version)

    results = camb.get_results(params)
    powers = results.get_cmb_power_spectra(params, CMB_unit='muK')
    unlensedCL = powers['unlensed_scalar']

    # Extract power spectra
    ell = np.arange(len(unlensedCL))
    tt = unlensedCL[:, 0]
    ee = unlensedCL[:, 1]
    te = unlensedCL[:, 3]

    return ell, tt, te, ee

def add_noise_spectrum(ell, spectrum, covariance_matrix, seed):
    """
    Adds noise to a given power spectrum using a multivariate normal distribution.

    Parameters:
        ell (array): Multipole values.
        spectrum (array): Power spectrum values.
        covariance_matrix (array): Covariance matrix for noise.
        seed (int): Random seed for reproducibility.

    Returns:
        array: Noisy power spectrum.
    """
    np.random.seed(seed)
    noisy_spectrum = np.random.multivariate_normal(spectrum[:len(covariance_matrix)], covariance_matrix, 1)
    return noisy_spectrum[0]

def save_power_spectrum(file_path, ell, noisy_spectrum):
    """
    Saves the noisy power spectrum to a CSV file.

    Parameters:
        file_path (str): Path to save the file.
        ell (array): Multipole values.
        noisy_spectrum (array): Noisy power spectrum.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as x:
        writer = csv.writer(x, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(ell)
        writer.writerow(noisy_spectrum)

