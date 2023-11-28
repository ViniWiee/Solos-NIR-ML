import numpy as np

def MeanCenter(nir_spectrum_2d):

    nir_mean = np.mean(nir_spectrum_2d, axis=1)

    newNirMean = nir_mean.reshape(-1, 1)
    nir_spectrum_2d_mc = nir_spectrum_2d - newNirMean
    return nir_spectrum_2d_mc


