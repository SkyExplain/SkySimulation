import csv
import os
import numpy as np
from astropy.io import fits

def save_power_spectrum(file_path, ell, noisy_spectrum):
    """
    Saves the noisy power spectrum to a CSV file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(ell)
        writer.writerow(noisy_spectrum)

def read_map(file_path: str) -> np.ndarray:
    """
    Reads a Healpy map from a FITS file and flattens the data.
    """
    with fits.open(file_path) as hdul:
        #If you want the verbose FITS info, uncomment:
        #hdul.info()
        if len(hdul) > 1 and hasattr(hdul[1], "columns"):
             print(hdul[1].columns)
        data = np.concatenate(hdul[1].data["T"])
        
    return data
