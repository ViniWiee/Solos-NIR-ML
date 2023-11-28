from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import kennard_stone as ks
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sys import stdout

from sklearn.model_selection import KFold, ShuffleSplit, train_test_split, cross_val_predict
from sklearn.utils import resample, check_random_state
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import astartes as at
#  https://nirpyresearch.com/kennard-stone-algorithm/


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

    z = np.polyfit(y_test, y_pred, 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_pred, y_test, c='red', edgecolors='k')
        ax.plot(z[1] + z[0] * y_test, y_test, c='blue', linewidth=1)
        ax.plot(y_test, y_test, color='green', linewidth=1)
        plt.title('$R^{2}$ (CV): ' + str(score_p))
        plt.xlabel('Potassio Predito (ppm)')
        plt.ylabel('Potassio Medido (ppm)')

        plt.show()



X = genfromtxt('CSV/spectrumWithoutOutliers.csv', delimiter=',')
y = genfromtxt('CSV/y.csv', delimiter=',')


X_train, X_test, y_train, y_test = at.train_test_split(X, y, test_size = 0.30)

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

base_pls(X_train, y_train, X_test, y_test, 3)

pd.DataFrame(X_train).to_csv("CSV/TrainingX.csv", header=None, index=None)
pd.DataFrame(y_train).to_csv("CSV/TrainingY.csv", header=None, index=None)
pd.DataFrame(X_test).to_csv("CSV/ValidationX.csv", header=None, index=None)
pd.DataFrame(y_test).to_csv("CSV/ValidationY.csv", header=None, index=None)