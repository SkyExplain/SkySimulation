from scipy.stats import skewnorm
from collections import namedtuple
import numpy as np
import healpy as hp
import time
import camb
import csv
import os

def covariance_asymmetric_errors(Dl, err_neg, err_pos, num_samples):
    """
    Example function for the simulation of a covariance matrix from asymmetric errors.
    """
    mean = Dl
    std_dev = (err_pos + err_neg) / 2  #std deviation
    skew_param = (err_pos - err_neg) / (err_pos + err_neg) * 10  #Skewness factor

    samples = skewnorm.rvs(a=skew_param, loc=mean, scale=std_dev, size=num_samples)
    return samples

def generate_camb_power_spectra(H0, ombh2, omch2, mnu, omk, tau,
                                As, ns, lmax, halofit_version='mead'):
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
    #Create a tuple to store the data
    CMBPowerSpectra = namedtuple("CMBPowerSpectra", ["ell", "tt", "te", "ee"])
    
    params = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau,
                             As=As, ns=ns, lmax=lmax, halofit_version=halofit_version)

    results = camb.get_results(params)
    powers = results.get_cmb_power_spectra(params, CMB_unit='muK')
    unlensedCL = powers['unlensed_scalar']

    #Extract power spectra
    ell = np.arange(len(unlensedCL))
    Cl_TT = unlensedCL[:, 0]
    Cl_EE = unlensedCL[:, 1]
    Cl_TE = unlensedCL[:, 3]

    print(type(CMBPowerSpectra(ell, Cl_TT, Cl_TE, Cl_EE)))

    return CMBPowerSpectra(ell, Cl_TT, Cl_TE, Cl_EE)

def add_noise_spectrum(spectrum, covariance_matrix, seed):
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

def generate_cmb_map(cmb_cls, nside=2048, output_dir="./", file_prefix="cmb_map"):
    """
    Generates a simulated CMB map using Healpy and saves it to a .fits file with a unique name.

    Parameters:
        cmb_cls (array): The CMB power spectrum (e.g., from Planck).
        nside (int): The resolution of the map (default is 2048).
        output_dir (str): Directory where the output file will be saved (default is current directory).
        file_prefix (str): Prefix for the output file name (default is "cmb_map").
    """
    #Generate CMB map using Healpy
    cmb_map = hp.synfast(cmb_cls, nside=nside, new=True)
    
    #Visualize
    hp.mollview(cmb_map, title="Simulated CMB Map")
    hp.graticule()

    return cmb_map

def generate_and_save_cmb_map(cmb_cls, nside=2048, output_dir="./", file_prefix="cmb_map"):
    """
    Generates a simulated CMB map using Healpy and saves it to a .fits file with a unique name.

    Parameters:
        cmb_cls (array): The CMB power spectrum (e.g., from Planck).
        nside (int): The resolution of the map (default is 2048).
        output_dir (str): Directory where the output file will be saved (default is current directory).
        file_prefix (str): Prefix for the output file name (default is "cmb_map").
    """
    #Generate CMB map using Healpy's synfast
    cmb_map = generate_cmb_map(cmb_cls[0], nside=nside, new=True)
    
    #Generate a unique file name by appending a timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}{file_prefix}_{timestamp}.fits"
    
    #Save the generated map to a FITS file
    hp.write_map(output_file, cmb_map)
    
    #Visualize the map
    hp.mollview(cmb_map, title="Simulated CMB Map")
    hp.graticule()

    print(f"CMB map saved as {output_file}")

def PK(k, As, ns, amp, freq, wid, centre, phase):
    """
    Function for the feature in the primordial Power spectrum (here power law with one wavepacket)
    """
    Pk = As*(k/0.05)**(ns-1)*(1+ np.sin(phase+k*freq)*amp*np.exp(-(k-centre)**2/wid**2))
    
    return Pk
