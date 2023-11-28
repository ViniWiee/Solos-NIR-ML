from sklearn.ensemble import IsolationForest


def detect(spectrum):

    iso = IsolationForest(contamination=0.085)
    yhat = iso.fit_predict(spectrum)
    mask = yhat != -1

    OutlierDetectedSpectrum = spectrum[mask, :]

    return OutlierDetectedSpectrum
