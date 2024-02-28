from sys import stdout
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pyod.models.qmcd import QMCD
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict
import astartes as at

## lendo e preparando o dataset

Dados = pd.read_excel('Dados.xlsx',sheet_name='Portatil')
Dados.info()


y = np.asarray(Dados.iloc[:,258])*100
y = np.asarray(y,dtype='int')
spectrum = np.asarray(Dados.iloc[:,2:258])

"""
A seguir, iremos utilizar o método Savitzky-Golay com segunda derivada. 
A Função foi encontrada em  https://nirpyresearch.com/savitzky-golay-smoothing-method/

"""

w = 17
p = 2

savitskyGolaySpectrum= savgol_filter(spectrum, w, polyorder=p, deriv=2)


## O código para este modelo de seleção de variáveis foi encontrado em:
## https://nirpyresearch.com/variable-selection-method-pls-python/

# Initial Data

def pls_variable_selection(X, y, max_comp):
    # Define MSE array to be populated
    mse = np.zeros((max_comp, X.shape[1]))

    # Loop over the number of PLS components
    for i in range(max_comp):

        # Regression with specified number of components, using full spectrum
        pls1 = PLSRegression(n_components=i + 1)
        pls1.fit(X, y)

        # Indices of sort spectra according to ascending absolute value of PLS coefficients
        sorted_ind = np.argsort(np.abs(pls1.coef_[:, 0]))

        # Sort spectra accordingly
        Xc = X[:, sorted_ind]

        # Discard one wavelength at a time of the sorted spectra,
        # regress, and calculate the MSE cross-validation
        for j in range(Xc.shape[1] - (i + 1)):
            pls2 = PLSRegression(n_components=i + 1)
            pls2.fit(Xc[:, j:], y)

            y_cv = cross_val_predict(pls2, Xc[:, j:], y, cv=5)

            mse[i, j] = mean_squared_error(y, y_cv)

        comp = 100 * (i + 1) / (max_comp)
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")

    # # Calculate and print the position of minimum in MSE
    mseminx, mseminy = np.where(mse == np.min(mse[np.nonzero(mse)]))

    print("Optimised number of PLS components: ", mseminx[0] + 1)
    print("Wavelengths to be discarded ", mseminy[0])
    print('Optimised MSE ', mse[mseminx, mseminy][0])
    stdout.write("\n")
    # plt.imshow(mse, interpolation=None)
    # plt.show()

    # Calculate PLS with optimal components and export values
    pls = PLSRegression(n_components=mseminx[0] + 1)
    pls.fit(X, y)

    sorted_ind = np.argsort(np.abs(pls.coef_[:, 0]))

    Xc = X[:, sorted_ind]

    return (Xc[:, mseminy[0]:], mseminx[0] + 1, mseminy[0], sorted_ind)

def find_Best_Spectrum(X,y):
    menor = len(X[0])
    optimalPLSComponents = 1
    optimalSpectrum = X
    for i in range(1,15):
        opt_Xc, ncomp, wav, sorted_ind = pls_variable_selection(X, y, i)
        if wav < menor:
            menor = wav
            optimalPLSComponents = ncomp
            optimalSpectrum = opt_Xc

    return (optimalSpectrum,optimalPLSComponents)

selectedSpectrum, optimalPLSComponents = find_Best_Spectrum(savitskyGolaySpectrum,y)


"""Example of using Quasi-Monte Carlo Discrepancy (QMCD) for
outlier detection
"""
# Author: D Kulik
# License: BSD 2 clause

def optimalNumbersForKennardStoneMethod(X_train,y_train,X_test,y_test):
    maior = 0
    val_l = 0
    for i in range(1,len(selectedSpectrum[0])):
        overfit, score = kennardStonePLS(X_train, y_train, X_test, y_test, i)
        if score > maior:
           maior = score
           val_l = i
    return val_l


def kennardStonePLS(X_train, y_train,X_test, y_test, n_components):
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)

    y_calib = pls.predict(X_train)
    y_pred = pls.predict(X_test)

    # metrics in prediction
    rmse_p, mae_p, score_p = np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)
    rmse_calib, mae_p, score_calib = np.sqrt(mean_squared_error(y_train, y_calib)), mean_absolute_error(y_train, y_calib), r2_score(
        y_train, y_calib)

    return (abs(score_calib - score_p), score_p)

def findBestContamination(X, y):
    score = 0
    bestContamination = 0
    bestXSpectrum = X
    bestYSpectrum = y
    optimal = 0
    for i in range(30):
        clf = QMCD(contamination=0.01 + (i / 100))
        print("Contamination:", 0.01 + (i / 100))

        clf.fit(X)
        outliers = clf.labels_

        indexes = []

        for k in range(len(outliers)):
            if (outliers[k] == 1):
                indexes.append(k)

        X1 = np.delete(X, indexes, 0)
        y1 = np.delete(y, indexes)

        X_train, X_test, y_train, y_test = at.train_test_split(X1, y1, test_size=0.30)

        opt = optimalNumbersForKennardStoneMethod(X_train, y_train, X_test, y_test)

        overfit, newScore = kennardStonePLS(X_train, y_train, X_test, y_test, opt)
        if (newScore > score and overfit < 0.03):
            score = newScore
            bestContamination = 1 + i
            bestXSpectrum = X1
            bestYSpectrum = y1
            optimal = opt

    return (bestXSpectrum,bestYSpectrum, bestContamination,optimal)

spectrumWithoutOutliers, y ,c, opt = findBestContamination(selectedSpectrum,y)


def printKennardStonePLS(X_train, y_train,X_test, y_test, n_components):
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)

    y_calib = pls.predict(X_train)
    y_pred = pls.predict(X_test)

    # metrics in prediction
    rmse_p, mae_p, score_p = np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)
    rmse_calib, mae_p, score_calib ,sd = np.sqrt(mean_squared_error(y_train, y_calib)), mean_absolute_error(y_train, y_calib), r2_score(
        y_train, y_calib), np.std(y_test)
    rpd = sd / rmse_p

    print('R2 Calib: %5.3f' % score_calib)
    print('R2 CV: %5.3f' % score_p)
    print('MSE Calib: %5.3f' % rmse_calib)
    print('MSE CV: %5.3f' % rmse_p)
    print("RPD : %5.3f" % rpd)


    z = np.polyfit(y_test, y_pred, 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_pred, y_test, c='red', edgecolors='k')
        ax.plot(z[1] + z[0] * y_test, y_test, c='blue', linewidth=1)
        ax.plot(y_test, y_test, color='green', linewidth=1)
        plt.title('$R^{2}$ : ' + str(score_p))
        plt.xlabel('Proteina Predita (porcentagem)')
        plt.ylabel('Proteina Medida(porcentagem)')

        plt.show()


X_train, X_test, y_train, y_test = at.train_test_split(spectrumWithoutOutliers, y, test_size = 0.30)

pd.DataFrame(X_train).to_csv("CSV/X_train.csv", header=None, index=None)
pd.DataFrame(y_train).to_csv("CSV/y_train.csv", header=None, index=None)
pd.DataFrame(X_test).to_csv("CSV/X_test.csv", header=None, index=None)
pd.DataFrame(y_test).to_csv("CSV/y_test.csv", header=None, index=None)

with plt.style.context(('seaborn')):
    f, axs = plt.subplots(1, 1, figsize=(8, 10))
    n, bins, patches = axs.hist(y_train, bins=20, alpha=0.75, label="Training set")
    axs.hist(y_test, bins=bins, alpha=0.75, label="Test set")
    axs.legend()
    axs.set_xlabel("TOC content")
    axs.set_ylabel("Number of samples")

    train_legend = np.repeat("_Training set", X_train.shape[0])
    train_legend = "Training set"
    test_legend = np.repeat("_Test set", X_test.shape[0])
    test_legend = "Test set"

plt.tight_layout()
plt.show()

printKennardStonePLS(X_train, y_train, X_test, y_test, opt)

pd.DataFrame(spectrumWithoutOutliers).to_csv("CSV/spectrumWithoutOutliers.csv", header=None, index=None)
pd.DataFrame(y).to_csv("CSV/y.csv", header=None, index=None)








