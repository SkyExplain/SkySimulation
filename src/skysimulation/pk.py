import numpy as np

def PK(k, As, ns, amp, freq, wid, centre, phase):
    """
    Feature template for the primordial power spectrum:
    power-law times (1 + wavepacket).
    """
    return As * (k / 0.05) ** (ns - 1) * (
        1 + np.sin(phase + k * freq) * amp * np.exp(-((k - centre) ** 2) / (wid ** 2))
    )
