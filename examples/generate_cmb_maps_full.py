import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import math
import numpy as np
import camb
import csv
import time
from camb import model, initialpower
print('Using CAMB %s'%(camb.__version__))

N_theta =500 #Number of samples

data_directory = "/mnt/netapp1/Store_CSIC/home/csic/eoy/ioj/SkySimulation/data/"
path_down_mask = data_directory + "simulated_data/down_mask/galactic_mask_256.fits"
os.chdir(data_directory)
print("Current working directory:", os.getcwd())

start = time.time()

#Random seed
seed0 = 314100

#nside: maps (resolution)
nside = 256

#Load the Planck data
#Planck TT data
data_Planck = np.loadtxt('./experimental_data/COM_PowerSpect_CMB-TT-full_R3.01.txt')
ls_Planck_TT = data_Planck[:, 0]
dl_Planck_TT = data_Planck[:, 1]
sdl_Planck_TT1 = data_Planck[:, 2]
sdl_Planck_TT2 = data_Planck[:, 3]
round_ls_Pl_TT = np.round(ls_Planck_TT)

#Planck TE data
data_Planck = np.loadtxt('./experimental_data/COM_PowerSpect_CMB-TE-full_R3.01.txt')
ls_Planck_TE = data_Planck[:, 0]
dl_Planck_TE = data_Planck[:, 1]
sdl_Planck_TE1 = data_Planck[:, 2]
sdl_Planck_TE2 = data_Planck[:, 3]
round_ls_Pl_TE = np.round(ls_Planck_TE)

#Planck EE data
data_Planck = np.loadtxt('./experimental_data/COM_PowerSpect_CMB-EE-full_R3.01.txt')
ls_Planck_EE = data_Planck[:, 0]
dl_Planck_EE = data_Planck[:, 1]
sdl_Planck_EE1 = data_Planck[:, 2]
sdl_Planck_EE2 = data_Planck[:, 3]
round_ls_Pl_EE = np.round(ls_Planck_EE)

#Conerting Dls to the c_ls^TT (in μK^2)
def Cls(l,DlTT):
    ClTT = [(2*math.pi)/(l[i]*(l[i]+1))*DlTT[i] for i in range(len(l))]
    return ClTT

#Read Covariance matrices
base_pathCV = "./simulated_data/simulated_cov_matrices/"
cov_matx_dltt_mcmc = np.loadtxt(base_pathCV +'/dlstt_cov_matx(mcmc).csv', delimiter=",")
cov_matx_dlte_mcmc = np.loadtxt(base_pathCV +'/dlste_cov_matx(mcmc).csv', delimiter=",")
cov_matx_dlee_mcmc = np.loadtxt(base_pathCV +'/dlsee_cov_matx(mcmc).csv', delimiter=",")

######################################################################################
#                        Generate the power spectra & maps
######################################################################################
from skysimulation import PK
from skysimulation import generate_camb_power_spectra
from skysimulation import add_noise_spectrum
from skysimulation import save_power_spectrum
from skysimulation import save_cmb_temperature_map
from skysimulation import save_cmb_polarization_maps
from skysimulation import apply_galactic_mask

rng = np.random.default_rng(0)
#Priors p(theta): simple uniform boxes
omega_cdm_samples = rng.uniform(0.10, 0.15, size=N_theta)
omega_b_samples   = rng.uniform(0.021, 0.023, size=N_theta)   #omega_b = Omega_b h^2
As_samples        = rng.uniform(1.8e-9, 2.4e-9, size=N_theta)
ns_samples        = rng.uniform(0.94, 0.99, size=N_theta)

#Feature-only priors p(phi):  A_lin
A_lin_samples     = rng.uniform(0.01, 0.06, size=N_theta)
freq_samples = rng.integers(low=100, high=500, size=N_theta) 

flag = 0

for theta_idx in range(N_theta):

    omega_cdm = omega_cdm_samples[theta_idx]
    omega_b   = omega_b_samples[theta_idx]
    As        = As_samples[theta_idx]
    ns        = ns_samples[theta_idx]

    #For reproducibility + avoid accidental correlations:
    #different seeds per (theta_idx, class)
    seed_lcdm    = seed0
    seed_feature = seed0

    ##-------------------------------------------------------
    #      ΛCDM case (standard primordial power spectrum)
    ##-------------------------------------------------------
    Power_spectra = generate_camb_power_spectra(
        67.4, omega_b, omega_cdm, 0.06, 0,
        tau=0.0544, As=As, ns=ns,
        halofit_version='mead', lmax=2507,
        custom_PK=False
    )

    dlstt_noisy = add_noise_spectrum(Power_spectra.tt, cov_matx_dltt_mcmc, seed_lcdm)
    dlste_noisy = add_noise_spectrum(Power_spectra.te, cov_matx_dlte_mcmc, seed_lcdm)
    dlsee_noisy = add_noise_spectrum(Power_spectra.ee, cov_matx_dlee_mcmc, seed_lcdm)
    #save_power_spectrum(f"{output_spct_lcdm}dlstt_lcdm_{flag_lcdm}.csv", round_ls_Pl_TT, dlstt_noisy_lcdm)
    #save_power_spectrum(f"{output_spct_lcdm}dlste_lcdm_{flag_lcdm}.csv", round_ls_Pl_TE, dlste_noisy_lcdm)
    #save_power_spectrum(f"{output_spct_lcdm}dlsee_lcdm_{flag_lcdm}.csv", round_ls_Pl_EE, dlsee_noisy_lcdm)

    cl_tt = Cls(round_ls_Pl_TT, dlstt_noisy)
    cl_ee = Cls(round_ls_Pl_EE, dlsee_noisy)
    cl_bb = np.zeros_like(cl_ee)
    cl_te = Cls(round_ls_Pl_TE, dlste_noisy)
    cl_eb = np.zeros_like(cl_ee)
    cl_tb = np.zeros_like(cl_ee)

    output_lcdm = "./simulated_data/simulated_maps/simulated_masked_maps/"
    save_cmb_polarization_maps(
        cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb,
        seed0=seed_lcdm, nside=nside,
        n_map=theta_idx,                #use theta_idx as ID
        output_dir=output_lcdm,
        custom_smooth=True, fwhm_arcmin=5.0,
        custom_Pk=False, apply_mask=True, mask_path=path_down_mask  
    )

    ##-------------------------------------------------------
    #              Non-standar feature case
    ##-------------------------------------------------------
    A_lin = A_lin_samples[theta_idx]
    freq_i = int(freq_samples[theta_idx])

    #generate_camb_power_spectra(H0, ombh2, omch2, mnu, omk, tau, As, ns, lmax, halofit_version="mead", custom_PK=False,
    #amp=0, freq=0, wid=1, centre=0.05, phase=0)
    Power_spectra = generate_camb_power_spectra(
        67.4, omega_b, omega_cdm, 0.06, 0,
        tau=0.0544, As=As, ns=ns,
        halofit_version='mead', lmax=2507,
        custom_PK=True,
        amp=A_lin, freq=freq_i, wid=0.04, centre=0.06, phase=0
    )

    dlstt_noisy = add_noise_spectrum(Power_spectra.tt, cov_matx_dltt_mcmc, seed_feature)
    dlste_noisy = add_noise_spectrum(Power_spectra.te, cov_matx_dlte_mcmc, seed_feature)
    dlsee_noisy = add_noise_spectrum(Power_spectra.ee, cov_matx_dlee_mcmc, seed_feature)
    #save_power_spectrum(f"{output_spct_f}dlstt_feature_{flag_feature}.csv", round_ls_Pl_TT, dlstt_noisy_feature)
    #save_power_spectrum(f"{output_spct_f}dlste_feature_{flag_feature}.csv", round_ls_Pl_TE, dlste_noisy_feature)
    #save_power_spectrum(f"{output_spct_f}dlsee_feature_{flag_feature}.csv", round_ls_Pl_EE, dlsee_noisy_feature)

    cl_tt = Cls(round_ls_Pl_TT, dlstt_noisy)
    cl_ee = Cls(round_ls_Pl_EE, dlsee_noisy)
    cl_bb = np.zeros_like(cl_ee)
    cl_te = Cls(round_ls_Pl_TE, dlste_noisy)
    cl_eb = np.zeros_like(cl_ee)
    cl_tb = np.zeros_like(cl_ee)

    output_feature = "./simulated_data/simulated_maps/simulated_masked_maps/"
    save_cmb_polarization_maps(
        cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb,
        seed0=seed_feature, nside=nside,
        n_map=theta_idx,                #same theta_idx
        output_dir=output_feature,
        custom_smooth=True, fwhm_arcmin=5.0,
        custom_Pk=True, apply_mask=True, mask_path=path_down_mask  
    )

    flag += 1

end = time.time()

print(f"Elapsed time: {end - start:.2f} seconds")