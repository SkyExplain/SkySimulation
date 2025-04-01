from scipy.stats import skewnorm
from collections import namedtuple
import numpy as np
import healpy as hp
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


def PK(k, As, ns, amp, freq, wid, centre, phase):
    """
    Function for the feature in the primordial Power spectrum (here power law with one wavepacket)
    """
    Pk = As*(k/0.05)**(ns-1)*(1+ np.sin(phase+k*freq)*amp*np.exp(-(k-centre)**2/wid**2))
    
    return Pk


def generate_camb_power_spectra(H0, ombh2, omch2, mnu, omk, tau,
                                As, ns, lmax, halofit_version='mead', custom_PK=False, amp=0, freq=0, wid=1, centre=0.05, phase=0):
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
        custom_PK (bool): If True, uses a modified primordial power spectrum.
        amp, freq, wid, centre, phase: Parameters for the custom PK function.


    Returns:
        tuple: (ell values, TT power spectrum, TE power spectrum, EE power spectrum)
    """
    #Create a tuple to store the data
    CMBPowerSpectra = namedtuple("CMBPowerSpectra", ["ell", "tt", "te", "ee"])
    
    params = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau,
                             As=As, ns=ns, lmax=lmax, halofit_version=halofit_version)

    #Customizedinitial power spectrum function
    if custom_PK:
        params.set_initial_power_function(PK, args=(As, ns, amp, freq, wid, centre, phase),
                                          effective_ns_for_nonlinear=ns)
    else:
        params.InitPower.set_params(As=As, ns=ns)

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


def generate_cmb_temperature_map(cmb_cls, nside=2048, output_dir="./", file_prefix="cmb_map"):
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
    hp.mollview(cmb_map, title="Simulated CMB Map", unit="K")

    return cmb_map


def save_cmb_temperature_map(cmb_cls, nside, n_map, output_dir="./", file_prefix="cmb_map", custom_Pk=False):
    """
    Generates a simulated CMB map using Healpy and saves it to a .fits file with a unique name.

    Parameters:
        cmb_cls (array): The CMB power spectrum (e.g., from Planck).
        nside (int): The resolution of the map (default is 2048).
        output_dir (str): Directory where the output file will be saved (default is current directory).
        file_prefix (str): Prefix for the output file name (default is "cmb_map").
    """
    #Generate CMB map using Healpy's synfast
    cmb_temp_map = hp.synfast(cmb_cls, nside=nside, new=True)
    
    if custom_Pk:
        output_file = f"{output_dir}{file_prefix}_feature_{n_map}.fits"
    else:
        output_file = f"{output_dir}{file_prefix}_{n_map}.fits"
    
    #Save the generated map to a FITS file
    hp.write_map(output_file, cmb_temp_map)

    print(f"CMB temperature map saved as {output_file}")


def generate_cmb_polarization_maps(cl_tt, cl_ee, cl_bb, cl_te, nside, output_dir="./", file_prefix="cmb_pol_map", custom_smooth=False, fwhm=0.0):
    """
    Generates a simulated CMB polarization map using Healpy and saves it to a .fits file with a unique name.

    Parameters:
        cl_tt, cl_ee, cl_bb, cl_te (arrays): The CMB power spectrum.
        nside (int): The resolution of the map (default is 2048).
        output_dir (str): Directory where the output file will be saved (default is current directory).
        file_prefix (str): Prefix for the output file name (default is "cmb_map").
        custom_smooth (bool): If True, uses a 5-degree smoothing function for the polarization maps.
        (To viasualize them like: 
        https://www.kicc.cam.ac.uk/research/cosmic-microwave-background-and-the-early-universe/Planck-Component-Separation)
        fwhm (float): Full width at half maximum for the Gaussian beam smoothing.
    """

    pol_maps = hp.synfast([cl_tt, cl_ee, cl_bb, cl_te], nside=nside, new=True, pol=True)
    #Extract the temprature polarization maps
    T_map, Q_map, U_map = pol_maps  #T, Q, and U components

    #If custom_smooth is True, apply a custom smoothing function
    if custom_smooth:
        fwhm = np.radians(fwhm)  #5-degree smoothing
        Q_smooth = hp.smoothing(Q_map, fwhm=fwhm)
        U_smooth = hp.smoothing(U_map, fwhm=fwhm)
    else:
        Q_smooth = Q_map
        U_smooth = U_map

    #Visualize the maps
    hp.mollview(Q_smooth, title="CMB Polarization (Q component)", cmap="RdBu", unit="μK")
    hp.mollview(U_smooth, title="CMB Polarization (U component)", cmap="RdBu", unit="μK")

    return Q_smooth, U_smooth


def save_cmb_polarization_maps(cl_tt, cl_ee, cl_bb, cl_te, nside, n_map, output_dir="./", file_prefix="cmb_pol_map", custom_smooth=False, fwhm=0.0, custom_Pk=False):
    """
    Saves simulated CMB polarization map using Healpy and saves it to a .fits file with a unique name.

    Parameters:
        cl_tt, cl_ee, cl_bb, cl_te (arrays): The CMB power spectrum.
        nside (int): The resolution of the map (default is 2048).
        output_dir (str): Directory where the output file will be saved (default is current directory).
        file_prefix (str): Prefix for the output file name (default is "cmb_map").
        custom_smooth (bool): If True, uses a 5-degree smoothing function for the polarization maps.
        (To viasualize them like: 
        https://www.kicc.cam.ac.uk/research/cosmic-microwave-background-and-the-early-universe/Planck-Component-Separation)
        fwhm (float): Full width at half maximum for the Gaussian beam smoothing.
        custom_Pk (bool): If True, uses a modified primordial power spectrum, nd adds the word "feature" to the generated file map.
    """

    #Generate CMB map using Healpy's synfast
    pol_maps = hp.synfast([cl_tt, cl_ee, cl_bb, cl_te], nside=nside, new=True, pol=True)
    #Extract the temprature polarization maps
    T_map, Q_map, U_map = pol_maps  #T, Q, and U components

    #If custom_smooth is True, apply a custom smoothing function
    if custom_smooth:
        fwhm = np.radians(fwhm)  #5-degree smoothing
        Q_smooth = hp.smoothing(Q_map, fwhm=fwhm)
        U_smooth = hp.smoothing(U_map, fwhm=fwhm)
    else:
        Q_smooth = Q_map
        U_smooth = U_map

    if custom_Pk:
        output_file1 = f"{output_dir}{file_prefix}_Q_feature_{n_map}.fits"
        output_file2 = f"{output_dir}{file_prefix}_U_feature_{n_map}.fits"
    else:
        output_file1 = f"{output_dir}{file_prefix}_Q_{n_map}.fits"
        output_file2 = f"{output_dir}{file_prefix}_U_{n_map}.fits"

    #Save the generated map to a FITS file
    hp.write_map(output_file1, Q_smooth)
    hp.write_map(output_file2, U_smooth)

    print(f"CMB polarization maps saved as {output_file1} and {output_file2}")