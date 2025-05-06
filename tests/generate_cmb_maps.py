import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import math
import numpy as np
import camb
import csv
from camb import model, initialpower
print('Using CAMB %s'%(camb.__version__))

#Set the path to the data directory
data_directory = "/mnt/lustre/scratch/nlsas/home/csic/eoy/ioj/CMBFeatureNet/data/"
os.chdir(data_directory)
print("Current working directory:", os.getcwd())

#Set random seed
seed0 = 314100

#nside: maps (resolution)
nside = 64

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
#                           Generate the power spectra
######################################################################################
from CMBFeatureNet import PK
from CMBFeatureNet import generate_camb_power_spectra
from CMBFeatureNet import save_power_spectrum

#-------------------------------------------------------------------------------------
#                        ΛCDM case (standard primordial power spectrum)
#-------------------------------------------------------------------------------------
omega_cdms = np.linspace(0.1, 0.15, 4) #Planck: omega_cdm = 0.12011
nss = np.linspace(0.94, 1.02, 3) #Planck: ns = 0.9649

flag_lcdm = 0

for omega_cdm in omega_cdms:
    for ns in nss:
        Power_spectra = generate_camb_power_spectra(67.4, 0.02237, omega_cdm, 0.06, 0, tau=0.0544,  
                        As=2.1e-9, ns=ns, halofit_version='mead', lmax=2507, custom_PK=False)

        output_spct_lcdm = "./simulated_data/simulated_ang_power_spectra/"
        from CMBFeatureNet import add_noise_spectrum
        dlstt_noisy_lcdm = add_noise_spectrum(Power_spectra.tt, cov_matx_dltt_mcmc, seed0)
        dlste_noisy_lcdm = add_noise_spectrum(Power_spectra.te, cov_matx_dlte_mcmc, seed0)
        dlsee_noisy_lcdm = add_noise_spectrum(Power_spectra.ee, cov_matx_dlee_mcmc, seed0)
        save_power_spectrum(f"{output_spct_lcdm}dlstt_lcdm_{flag_lcdm}.csv", round_ls_Pl_TT, dlstt_noisy_lcdm)
        save_power_spectrum(f"{output_spct_lcdm}dlste_lcdm_{flag_lcdm}.csv", round_ls_Pl_TE, dlste_noisy_lcdm)
        save_power_spectrum(f"{output_spct_lcdm}dlsee_lcdm_{flag_lcdm}.csv", round_ls_Pl_EE, dlsee_noisy_lcdm)

        #Convert Dls to Cls
        cl_tt = Cls(round_ls_Pl_TT,dlstt_noisy_lcdm)  #TT
        cl_ee = Cls(round_ls_Pl_EE,dlsee_noisy_lcdm)  #EE
        cl_bb = np.zeros_like(cl_ee)  #BB (set to zero if not considering B-modes)
        cl_te = Cls(round_ls_Pl_TE,dlste_noisy_lcdm)  #TE
        cl_eb = np.zeros_like(cl_ee) 
        cl_tb = np.zeros_like(cl_ee) 

        output_lcdm = "./simulated_maps/lcdm/"
        #Generate and save the temperature map
        from CMBFeatureNet import save_cmb_temperature_map
        save_cmb_temperature_map(cl_tt, nside=nside, n_map=flag_lcdm, output_dir=output_lcdm, custom_Pk=False)

        #The polarization maps
        from CMBFeatureNet import save_cmb_polarization_maps
        save_cmb_polarization_maps(cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb, nside=nside, n_map=flag_lcdm, output_dir=output_lcdm, custom_smooth=False, custom_Pk=False)
        flag_lcdm += 1

#-------------------------------------------------------------------------------------
#                                Non-standar feature case
#-------------------------------------------------------------------------------------
freq = 1000 #Frequency
ks = np.linspace(0.02,1,1000) #wavenumber
omega_cdms = np.linspace(0.1, 0.15, 4)
A_lins = np.linspace(0.01, 0.06, 3)

flag_feature = 0
for omega_cdm in omega_cdms:
    for A_lin in A_lins:
        Power_spectra = generate_camb_power_spectra(67.4, 0.02237, omega_cdm, 0.06, 0, tau=0.0544,  
                            As=2.1e-9, ns=0.9649, halofit_version='mead', lmax=2507, custom_PK=True, amp=A_lin, freq=freq, wid=0.08, centre=0.2, phase=0)
        
        output_spct_f = "./simulated_data/simulated_ang_power_spectra/"
        dlstt_noisy_feature = add_noise_spectrum(Power_spectra.tt, cov_matx_dltt_mcmc, seed0)
        dlste_noisy_feature = add_noise_spectrum(Power_spectra.te, cov_matx_dlte_mcmc, seed0)
        dlsee_noisy_feature = add_noise_spectrum(Power_spectra.ee, cov_matx_dlee_mcmc, seed0)
        save_power_spectrum(f"{output_spct_f}dlstt_feature_{flag_feature}.csv", round_ls_Pl_TT, dlstt_noisy_feature)
        save_power_spectrum(f"{output_spct_f}dlste_feature_{flag_feature}.csv", round_ls_Pl_TE, dlste_noisy_feature)
        save_power_spectrum(f"{output_spct_f}dlsee_feature_{flag_feature}.csv", round_ls_Pl_EE, dlsee_noisy_feature)

        #Convert Dls to Cls
        cl_tt = Cls(round_ls_Pl_TT,dlstt_noisy_feature)  #TT
        cl_ee = Cls(round_ls_Pl_EE,dlsee_noisy_feature)  #EE
        cl_bb = np.zeros_like(cl_ee)  #BB (set to zero if not considering B-modes)
        cl_te = Cls(round_ls_Pl_TE,dlste_noisy_feature)  #TE

        output_feature = "./simulated_maps/feature/"
        #Generate and save the temperature map
        save_cmb_temperature_map(cl_tt, nside=nside, n_map=flag_feature, output_dir=output_feature, custom_Pk=True)

        #The polarization maps
        save_cmb_polarization_maps(cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb, nside=nside, n_map=flag_feature, output_dir=output_feature, custom_smooth=False, custom_Pk=True)
        flag_feature += 1