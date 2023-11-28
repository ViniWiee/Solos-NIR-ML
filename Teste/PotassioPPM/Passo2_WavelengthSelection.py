import pandas as pd
from numpy import genfromtxt
from sys import stdout
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


## O código para este modelo de seleção de variáveis foi encontrado em:
## https://nirpyresearch.com/variable-selection-method-pls-python/

# Initial Data

X = genfromtxt('CSV/savitskygolayspectrum.csv', delimiter=',')
y = genfromtxt('CSV/y.csv', delimiter=',')


def mc(input_data):
    # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i, :] -= input_data[i, :].mean()
    return input_data


def snv(input_data):
    # Define a new array and populate it with the corrected data
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        output_data[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])

    return output_data

def simple_pls_cv(X, y, n_comp):
    # Run PLS with suggested number of components
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X, y)
    y_c = pls.predict(X)

    # Cross-validation
    y_cv = cross_val_predict(pls, X, y, cv=10)

    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)

    # Calculate mean square error for calibration and cross validation
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)

    print('R2 calib: %5.3f' % score_c)
    print('R2 CV: %5.3f' % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    print('MSE CV: %5.3f' % mse_cv)

    # Plot regression

    z = np.polyfit(y, y_cv, 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_cv, y, c='red', edgecolors='k')
        ax.plot(z[1] + z[0] * y, y, c='blue', linewidth=1)
        ax.plot(y, y, color='green', linewidth=1)
        plt.title('$R^{2}$ (CV): ' + str(score_cv))
        plt.xlabel('Predicted $^{\circ}$Brix')
        plt.ylabel('Measured $^{\circ}$Brix')

        plt.show()

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
    for i in range(1,11):
        opt_Xc, ncomp, wav, sorted_ind = pls_variable_selection(X, y, i)
        if wav < menor:
            menor = wav
            optimalPLSComponents = ncomp
            optimalSpectrum = opt_Xc

    return (optimalSpectrum,optimalPLSComponents)

selectedSpectrum, optimalPLSComponents = find_Best_Spectrum(X,y)


fig, axs = plt.subplots(2, 1)  # initialize a figure with one axis
axs[0].plot(X)  # plot the data
axs[0].set_xlabel('wavenumber [$\mathregular{cm^{-1}]}$')  # matplotlib can interpret this LateX expression
axs[0].set_ylabel('Absorbancia')
axs[0].grid(True)  # show a grid to help guide the viewer's eye
axs[0].set_title("Espectro savitskygolay")  # set a title from the filename of the input data
axs[1].plot(selectedSpectrum)
axs[1].set_xlabel('wavenumber [$\mathregular{cm^{-1}]}$')
axs[1].set_title("Espectro pós Selec")
axs[1].grid(True)
fig.tight_layout()  # a standard option for matplotlib
plt.show()



# Exporting the spectrum into csv file.
pd.DataFrame(selectedSpectrum).to_csv("CSV/selectedSpectrum.csv", header=None, index=None)

