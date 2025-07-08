#Import packages
import pandas as pd
import numpy as np
import math
import os
from astropy.io import fits
import healpy as hp
from matplotlib import pyplot as plt

seed0 = 314100
data_directory = "/cosmodata/iocampo/SkySimulation/data/"
os.chdir(data_directory)
print("Current working directory:", os.getcwd())

#Planck TT data
data_Planck = np.loadtxt('./experimental_data/COM_PowerSpect_CMB-TT-full_R3.01.txt')
ls_Planck_TT = data_Planck[:, 0]
dl_Planck_TT = data_Planck[:, 1]
sdl_Planck_TT1 = data_Planck[:, 2]
sdl_Planck_TT2 = data_Planck[:, 3]
round_ls_Pl_TT = np.round(ls_Planck_TT)

#Conerting Dls^TT to the c_ls^TT
def Cls(l,DlTT):
    ClTT = [(2*math.pi)/(l[i]*(l[i]+1))*DlTT[i] for i in range(len(l))]
    return ClTT

def Dls(l, ClTT):
    DlTT = [l[i]*(l[i]+1)/(2*np.pi)*ClTT[i] for i in range(len(l))]
    return DlTT

#Generate the power spectra
from SkySimulation import generate_camb_power_spectra
base_pathCV = "./simulated_data/simulated_cov_matrices/"
cov_matx_dltt_mcmc = np.loadtxt(base_pathCV +'/dlstt_cov_matx(mcmc).csv', delimiter=",")

#Function to generate the differences plot (feature - lcdm)
def cmb_cls_plot(nside):   
    seed0 = 314100 
    fsize = 16 
    #Convert in map
    cmb_map_feature = generate_cmb_temperature_map(cl_tt_feature, nside=nside, seed0=seed0)
    cmb_map_lcdm = generate_cmb_temperature_map(cl_tt_lcdm, nside=nside, seed0=seed0)

    #Upper plot (Power Spectrum)
    np.random.seed(seed0)
    cl_tt_map_feature = hp.anafast(cmb_map_feature, pol=True)
    np.random.seed(seed0)
    cl_tt_map_lcdm = hp.anafast(cmb_map_lcdm, pol=True)

    ell = np.arange(len(Power_spectra_lcdm[0]))
    #lmax = len(cl_tt_map)
    lmax = len(cl_tt_map_feature)

    fig = plt.figure(figsize=(10, 6))

    #Upper plot (Power Spectrum)
    frame1 = fig.add_axes((.1, .4, .8, .6))
    frame1.plot(ell[:lmax], Dls(ell[:lmax], cl_tt_map_lcdm)[:lmax], 
                label='$\Lambda CDM_{maps}$', alpha=0.6, color='blue')
    frame1.plot(ell[:lmax], Dls(ell[:lmax], cl_tt_map_feature)[:lmax], 
                label='$Feature_{maps}, A_{lin}=0.06$', alpha=0.8, color='orange', linestyle='--')
    #frame1.plot(ell, Power_spectra_normal[1], 
                #label='$\Lambda CDM$', color='gray')
    frame1.set_xlabel(r'$\ell$', fontsize=fsize)
    frame1.set_ylabel(r'$D_\ell^{TT_{map}}=C_\ell \, \ell(\ell+1)/2\pi$', fontsize=fsize)
    frame1.set_xlim(-5, len(Power_spectra_feature[0]))
    frame1.tick_params(axis='x', labelsize=0.1)
    frame1.grid(True, which='both', linestyle='--', linewidth=0.5)
    frame1.legend(fontsize=fsize)
    
    frame2 = fig.add_axes((.1, .19, .8, .22), sharex=frame1)
    #diff1 = (Dls(ell[:lmax], cl_tt_map_feature) - Power_spectra_normal[1][:lmax])
    #diff2 = (Dls(ell[:lmax], cl_tt_map_lcdm) - Power_spectra_normal[1][:lmax])
    diff1 = np.array(Dls(ell[:lmax], cl_tt_map_feature)) - np.array(Dls(ell[:lmax], cl_tt_map_lcdm))
    #frame2.plot(ell[:lmax], diff2, color='blue', alpha=0.5)
    frame2.plot(ell[:lmax], diff1, alpha=0.8, color='orange', linestyle='--')
    frame2.axhline(y=0, alpha=0.6, color='blue')
    frame2.set_xlabel(r'$\ell$', fontsize=fsize)
    frame2.set_ylabel(r'$(D_\ell^{TT_{\Lambda CDM}} - D_\ell^{TT_{feature}})$', fontsize=fsize)
    frame2.tick_params(axis='x', labelsize=10)
    frame2.grid(True, which='both', linestyle='--', linewidth=0.5)

    return plt

#Convert in map and save the feature map
from SkySimulation import generate_cmb_temperature_map

#Function to generate the CMB map, to loop over nside:
def cmb_map_plot_feature(nside):
    np.random.seed(seed0)
    cmb_map_feature = generate_cmb_temperature_map(cl_tt_feature, nside=nside, seed0=seed0)
    plt.savefig(f"/cosmodata/iocampo/SkySimulation/plots/cmb_map_{nside}.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    plt.clf()

    return plt

#------------------------------------------------------------------------------

#Generate the power spectra
Power_spectra_lcdm = generate_camb_power_spectra(67.4, 0.02237, 0.1200, 0.06, 0, tau=0.0544,  
                    As=2.1e-9, ns=0.9649, halofit_version='mead', lmax=2507, custom_PK=False)

Power_spectra_feature = generate_camb_power_spectra(67.4, 0.02237, 0.1200, 0.06, 0, tau=0.0544,  
                    As=2.1e-9, ns=0.9649, halofit_version='mead', lmax=2507, custom_PK=True, 
                    amp=0.0599, freq = 1000, wid=0.04, centre=0.06, phase=0)

#Add noise
from SkySimulation import add_noise_spectrum
dlstt_noisy_feature = add_noise_spectrum(Power_spectra_lcdm.tt[:len(cov_matx_dltt_mcmc)], cov_matx_dltt_mcmc, seed0=seed0)
cl_tt_feature = Cls(round_ls_Pl_TT, dlstt_noisy_feature)
dlstt_noisy_lcdm = add_noise_spectrum(Power_spectra_feature.tt[:len(cov_matx_dltt_mcmc)], cov_matx_dltt_mcmc, seed0=seed0)
cl_tt_lcdm = Cls(round_ls_Pl_TT, dlstt_noisy_lcdm)

#Nside values
nside_list = [64, 128, 256, 512, 1024]


for nside in nside_list:
    from SkySimulation import read_map
    plt_cmb = cmb_map_plot_feature(nside)
    plt_obj = cmb_cls_plot(nside)
    plot_path = os.path.join('/cosmodata/iocampo/SkySimulation/plots/', f'feature_lcdm_nside_{nside}.png')
    plt_obj.savefig(plot_path)
    plt_obj.close()
