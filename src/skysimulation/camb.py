from collections import namedtuple
import numpy as np
import camb

from .pk import PK

def generate_camb_power_spectra(
    H0, ombh2, omch2, mnu, omk, tau,
    As, ns, lmax,
    halofit_version="mead",
    custom_PK=False,
    amp=0, freq=0, wid=1, centre=0.05, phase=0,
):
    """
    Generates CMB power spectra using CAMB.

    Returns:
        namedtuple("CMBPowerSpectra", ["ell", "tt", "te", "ee"])
    """
    CMBPowerSpectra = namedtuple("CMBPowerSpectra", ["ell", "tt", "te", "ee"])

    params = camb.set_params(
        H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau,
        As=As, ns=ns, lmax=lmax, halofit_version=halofit_version
    )

    if custom_PK:
        params.set_initial_power_function(
            PK,
            args=(As, ns, amp, freq, wid, centre, phase),
            effective_ns_for_nonlinear=ns,
        )
    else:
        params.InitPower.set_params(As=As, ns=ns)

    results = camb.get_results(params)
    powers = results.get_cmb_power_spectra(params, CMB_unit="muK")
    unlensedCL = powers["unlensed_scalar"]

    ell = np.arange(len(unlensedCL))
    Cl_TT = unlensedCL[:, 0]
    Cl_EE = unlensedCL[:, 1]
    Cl_TE = unlensedCL[:, 3]

    return CMBPowerSpectra(ell, Cl_TT, Cl_TE, Cl_EE)
