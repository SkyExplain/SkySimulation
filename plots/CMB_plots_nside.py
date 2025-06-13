#Import packages
import pandas as pd
import numpy as np
import math
import os
from astropy.io import fits
import healpy as hp
from matplotlib import pyplot as plt

data_directory = "/mnt/lustre/scratch/nlsas/home/csic/eoy/ioj/CMBFeatureNet/data/"
os.chdir(data_directory)
print("Current working directory:", os.getcwd())

#Conerting Dls to the c_ls^TT
def Cls(l,DlTT):
    ClTT = [(2*math.pi)/(l[i]*(l[i]+1))*DlTT[i] for i in range(len(l))]
    return ClTT

def Dls(l, ClTT):
    DlTT = [l[i]*(l[i]+1)/(2*np.pi)*ClTT[i] for i in range(len(l))]
    return DlTT

#Read the cmb angular power spectra
simulated_dlstt = np.loadtxt('./simulated_data/simulated_ang_power_spectra/dlstt_feature_0_NSIDE128.csv', delimiter=',')

# Example Nside values
nside_list = [64, 128, 512, 1024, 2048]

def cmb_map_plot(nside):
    nside = hp.npix2nside(len(map_temp_data_feature))
    print(f"NSIDE: {nside}")

    hp.mollview(map_temp_data_feature, title=f"Temperature Map, Nside={nside}", unit="Intensity")
    plt.savefig(f'/mnt/lustre/scratch/nlsas/home/csic/eoy/ioj/CMBFeatureNet/plots/cmb_map_feature_nside_{nside}.png')
    plt.clf()

    return plt

def generate_plot(nside):
    # Compute the power spectra
    cl_tt_map_feature = hp.anafast(map_temp_data_feature, pol=True)
    cl_tt_map_lcdm = hp.anafast(map_temp_data_lcdm, pol=True)

    ell = np.arange(len(simulated_dlstt[1]))
    #lmax = len(cl_tt_map)
    lmax = len(cl_tt_map_lcdm)

    fig = plt.figure(figsize=(10, 6))

    #Upper plot (Power Spectrum)
    frame1 = fig.add_axes((.1, .4, .8, .6))
    frame1.plot(ell[:lmax], Dls(ell[:lmax], cl_tt_map_feature)[:lmax], 
                label='from maps (feature)', linewidth=0.5, alpha=0.8, color='red')
    frame1.plot(ell[:lmax], Dls(ell[:lmax], cl_tt_map_lcdm)[:lmax], 
                label='from maps (lcdm)', linewidth=0.5, alpha=0.5, color='blue')
    frame1.plot(ell, simulated_dlstt[1], 
                label='simulated', linewidth=0.5, alpha=0.3)
    frame1.set_xlabel(r'Multipole moment $\ell$')
    frame1.set_ylabel(r'$D_\ell^{TT_{map}}=C_\ell \, \ell(\ell+1)/2\pi$')
    frame1.set_xlim(-5, len(simulated_dlstt[1]))
    frame1.tick_params(axis='x', labelsize=0.1)
    frame1.grid(True, which='both', linestyle='--', linewidth=0.5)
    frame1.legend()
    
    frame2 = fig.add_axes((.1, .19, .8, .22), sharex=frame1)
    diff1 = (Dls(ell[:lmax], cl_tt_map_feature) - simulated_dlstt[1][:lmax])
    diff2 = (Dls(ell[:lmax], cl_tt_map_lcdm) - simulated_dlstt[1][:lmax])
    frame2.plot(ell[:lmax], diff1, label='difference feature, $A_{lin}=0.1$', linewidth=0.5, color='red', alpha=0.8)
    frame2.plot(ell[:lmax], diff2, label='difference lcdm-like', linewidth=0.5, color='blue', alpha=0.5)
    frame2.set_xlabel(r'Multipole moment $\ell$')
    frame2.set_ylabel(r'$(D_\ell^{TT_{map}} - D_\ell^{sim})$')
    frame2.tick_params(axis='x', labelsize=10)
    frame2.grid(True, which='both', linestyle='--', linewidth=0.5)
    frame2.legend()

    return plt



for nside in nside_list:
    from CMBFeatureNet import read_map
    test_path = "./simulated_maps/"
    map_temp_data_feature = read_map(test_path + f'tests_NSIDE{nside}/cmb_map_feature_0.fits')
    map_temp_data_lcdm = read_map(test_path + f'tests_NSIDE{nside}/cmb_map_0.fits')
    plt_cmb = cmb_map_plot(nside)
    plt_obj = generate_plot(nside)
    plot_path = os.path.join('/mnt/lustre/scratch/nlsas/home/csic/eoy/ioj/CMBFeatureNet/plots/', f'plot_nside_{nside}.png')
    plt_obj.savefig(plot_path)
    plt_obj.close()
