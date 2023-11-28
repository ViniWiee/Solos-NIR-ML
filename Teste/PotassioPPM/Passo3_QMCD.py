"""Example of using Quasi-Monte Carlo Discrepancy (QMCD) for
outlier detection
"""
import sys

# Author: D Kulik
# License: BSD 2 clause



import astartes as at
from sys import stdout

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import genfromtxt
from pyod.models.qmcd import QMCD
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_predict


def optimalNumbersForKennardStoneMethod(X_train,y_train,X_test,y_test):
    maior = 0
    val_l = 0
    for i in range(1,len(X[0])):
        overfit, score = base_pls(X_train, y_train, X_test, y_test, i)
        if score > maior:
           maior = score
           val_l = i
    return val_l


def base_pls(X_train, y_train,X_test, y_test, n_components):
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)

    y_calib = pls.predict(X_train)
    y_pred = pls.predict(X_test)

    # metrics in prediction
    rmse_p, mae_p, score_p = np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)
    rmse_calib, mae_p, score_calib = np.sqrt(mean_squared_error(y_train, y_calib)), mean_absolute_error(y_train, y_calib), r2_score(
        y_train, y_calib)

    print('R2 Calib: %5.3f' % score_calib)
    print('R2 CV: %5.3f' % score_p)
    print('MSE Calib: %5.3f' % rmse_calib)
    print('MSE CV: %5.3f' % rmse_p)

    return (abs(score_calib -score_p),score_p)

def findBestContamination(X, y):
    score = 0
    bestContamination = 0
    bestSpectrum = X
    bestYSpectrum = y
    optimal = 0
    for i in range(30):
        clf = QMCD(contamination=0.01+(i/100))
        print("Contamination:" ,0.01+(i/100))


        clf.fit(X)
        outliers = clf.labels_

        indexes = []

        for k in range(len(outliers)):
            if (outliers[k] == 1):
                indexes.append(k)

        X1 = np.delete(X,indexes,0)
        y1 = np.delete(y,indexes)

        X_train, X_test, y_train, y_test = at.train_test_split(X1, y1, test_size=0.30)

        opt  = optimalNumbersForKennardStoneMethod(X_train,y_train,X_test,y_test)

        overfit, newScore = base_pls(X_train,y_train,X_test,y_test,opt)
        if( newScore > score and overfit < 0.05):
            score = newScore
            bestContamination = 1+i
            bestXSpectrum = X1
            bestYSpectrum = y1
            optimal = opt

    return (bestXSpectrum,bestYSpectrum, bestContamination,optimal)



X = genfromtxt('CSV/selectedSpectrum.csv', delimiter=',')
y = genfromtxt('CSV/y.csv', delimiter=',')

spectrumWithoutOutliers, y ,c, opt = findBestContamination(X,y)

print(opt)

pd.DataFrame(spectrumWithoutOutliers).to_csv("CSV/spectrumWithoutOutliers.csv", header=None, index=None)
pd.DataFrame(y).to_csv("CSV/y.csv", header=None, index=None)






