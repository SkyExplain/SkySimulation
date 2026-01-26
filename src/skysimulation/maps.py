import numpy as np
import healpy as hp

def generate_cmb_temperature_map(cmb_cls, nside, seed0):
    rng = np.random.default_rng(seed0)
    # healpy uses global RNG internally in some places; setting seed via np.random.seed is sometimes needed.
    np.random.seed(seed0)
    cmb_map = hp.synfast(cmb_cls, nside=nside, new=True, sigma=None, pol=False)
    hp.mollview(cmb_map, title=f"CMB Map, Nside={nside}", unit="K")
    return cmb_map

def save_cmb_temperature_map(cmb_cls, nside, n_map, seed0, output_dir="./", file_prefix="cmb_map", custom_Pk=False):
    np.random.seed(seed0)
    cmb_temp_map = hp.synfast(cmb_cls, nside=nside, new=True, pol=False, sigma=None)
    suffix = "_feature" if custom_Pk else ""
    output_file = f"{output_dir}{file_prefix}{suffix}_{n_map}.fits"
    hp.write_map(output_file, cmb_temp_map)

def generate_cmb_polarization_maps(
    cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb,
    nside, seed0,
    custom_smooth=False, fwhm_arcmin=5.0,
):
    np.random.seed(seed0)
    T_map, Q_map, U_map = hp.synfast([cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb], nside=nside, new=True, pol=True)

    if custom_smooth:
        fwhm_rad = np.radians(fwhm_arcmin / 60)
        Q_smooth = hp.smoothing(Q_map, fwhm=fwhm_rad)
        U_smooth = hp.smoothing(U_map, fwhm=fwhm_rad)
    else:
        Q_smooth, U_smooth = Q_map, U_map

    hp.mollview(Q_smooth, title="CMB Polarization (Q component)", cmap="RdBu", unit="μK")
    hp.mollview(U_smooth, title="CMB Polarization (U component)", cmap="RdBu", unit="μK")
    return Q_smooth, U_smooth

def save_cmb_polarization_maps(
    cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb,
    nside, n_map, seed0,
    output_dir="./", file_prefix="cmb_pol_map",
    custom_smooth=False, fwhm_arcmin=0.0,
    custom_Pk=False,
):
    np.random.seed(seed0)
    T_map, Q_map, U_map = hp.synfast([cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb], nside=nside, new=True, pol=True)

    if custom_smooth:
        fwhm_rad = np.radians(fwhm_arcmin / 60)
        T_map = hp.smoothing(T_map, fwhm=fwhm_rad)
        Q_map = hp.smoothing(Q_map, fwhm=fwhm_rad)
        U_map = hp.smoothing(U_map, fwhm=fwhm_rad)

    suffix = "_feature" if custom_Pk else ""
    output_file0 = f"{output_dir}{file_prefix}_T{suffix}_{n_map}.fits"
    output_file1 = f"{output_dir}{file_prefix}_Q{suffix}_{n_map}.fits"
    output_file2 = f"{output_dir}{file_prefix}_U{suffix}_{n_map}.fits"

    hp.write_map(output_file0, T_map)
    hp.write_map(output_file1, Q_map)
    hp.write_map(output_file2, U_map)

def deconvolve_gaussian_beam(ells, Cl_smooth_map, fwhm_arcmin):
    fwhm_rad = np.radians(fwhm_arcmin / 60.0)
    sigma = fwhm_rad / np.sqrt(8.0 * np.log(2.0))
    B_ell = np.exp(-0.5 * ells * (ells + 1) * sigma**2)
    return Cl_smooth_map / (B_ell**2)
