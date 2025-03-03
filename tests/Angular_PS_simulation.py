import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import math
import numpy as np
import camb
import csv
from camb import model, initialpower
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

#path:
base_path = "/mnt/lustre/scratch/nlsas/home/csic/eoy/ioj/CMBFeatureNet/simulated_maps/"

#Random seed
seed0 = 314100

#Planck TT data
data_Planck = np.loadtxt(base_path + './experimental_data/COM_PowerSpect_CMB-TT-full_R3.01.txt')
ls_Planck_TT = data_Planck[:, 0]
dl_Planck_TT = data_Planck[:, 1]
sdl_Planck_TT1 = data_Planck[:, 2]
sdl_Planck_TT2 = data_Planck[:, 3]
round_ls_Pl_TT = np.round(ls_Planck_TT)

#Planck TE data
data_Planck = np.loadtxt(base_path + './experimental_data/COM_PowerSpect_CMB-TE-full_R3.01.txt')
ls_Planck_TE = data_Planck[:, 0]
dl_Planck_TE = data_Planck[:, 1]
sdl_Planck_TE1 = data_Planck[:, 2]
sdl_Planck_TE2 = data_Planck[:, 3]
round_ls_Pl_TE = np.round(ls_Planck_TE)

#Planck EE data
data_Planck = np.loadtxt(base_path + './experimental_data/COM_PowerSpect_CMB-EE-full_R3.01.txt')
ls_Planck_EE = data_Planck[:, 0]
dl_Planck_EE = data_Planck[:, 1]
sdl_Planck_EE1 = data_Planck[:, 2]
sdl_Planck_EE2 = data_Planck[:, 3]
round_ls_Pl_EE = np.round(ls_Planck_EE)

from CMBFeatureNet import generate_camb_power_spectra
Power_spectra = generate_camb_power_spectra(67.4, 0.02237, 0.1200, 0.06, 0, tau=0.0544,  
                    As=2.1e-9, ns=0.9649, halofit_version='mead', lmax=2507)

from CMBFeatureNet import add_noise_spectrum
#Read Covariance matrices
cov_matx_dltt_mcmc = np.loadtxt(base_path + './simulated_data/dlstt_cov_matx(mcmc).csv', delimiter=",")
cov_matx_dlte_mcmc = np.loadtxt(base_path + './simulated_data/dlste_cov_matx(mcmc).csv', delimiter=",")
cov_matx_dlee_mcmc = np.loadtxt(base_path + './simulated_data/dlsee_cov_matx(mcmc).csv', delimiter=",")

dlstt_mcmc = add_noise_spectrum(Power_spectra.tt, cov_matx_dltt_mcmc, seed0)
dlste_mcmc = add_noise_spectrum(Power_spectra.te, cov_matx_dlte_mcmc, seed0)
dlsee_mcmc = add_noise_spectrum(Power_spectra.ee, cov_matx_dlee_mcmc, seed0)

from CMBFeatureNet import save_power_spectrum
save_power_spectrum(base_path + './simulated_data/dltt_noisy_assym.csv', round_ls_Pl_TT, dlstt_mcmc)
save_power_spectrum(base_path + './simulated_data/dlte_noisy_assym.csv', round_ls_Pl_TE, dlste_mcmc)
save_power_spectrum(base_path + './simulated_data/dlee_noisy_assym.csv', round_ls_Pl_EE, dlsee_mcmc)