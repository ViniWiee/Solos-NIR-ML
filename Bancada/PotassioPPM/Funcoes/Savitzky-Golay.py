from scipy.signal import savgol_filter

w = 3
p = 2

SGSpectrum = savgol_filter(Spectrum, w, polyorder=p, deriv=2)