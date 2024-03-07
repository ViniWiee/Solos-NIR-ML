import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

y_test = genfromtxt('CSV/y_test.csv', delimiter=',')
y_train = genfromtxt('CSV/y_train.csv', delimiter=',')
X_train = genfromtxt('CSV/X_train.csv', delimiter=',')
X_test = genfromtxt('CSV/X_test.csv', delimiter=',')


pipe = Pipeline([
        ('svr', SVR())])

parameters = {
    'svr__kernel' : ['linear','poly','rbf'],
    'svr__degree' : list(range(1,10)),
    'svr__gamma': [1.0,10.0,100.0,1000.0,10000.0,100000.0,1000000.0],
    'svr__C': [1.0,10.0,100.0,1000.0,10000.0,100000.0,1000000.0]
}

model = GridSearchCV(pipe, param_grid = parameters, cv=10)


model.fit(X_train, y_train)

## TEST SET METRICS
y_val = model.predict(X_test)
validation_r2 = model.score(X_test,y_test)
validation_rmse = np.sqrt(mean_squared_error(y_test, y_val))
validation_mae = mean_absolute_error(y_test, y_val)
validation_sd = np.std(y_test)
validation_rpd = validation_sd/validation_rmse

print(validation_r2)
print(validation_rmse)
print(validation_mae)
print(validation_rpd)

## TRAIN SET METRICS
y_cal = model.predict(X_train)
calibration_r2 = model.score(X_train,y_train)
calibration_rmse = np.sqrt(mean_squared_error(y_train, y_cal))
calibration_mae = mean_absolute_error(y_train, y_cal)
calibration_sd = np.std(y_train)
calibration_rpd = calibration_sd/calibration_rmse

print(calibration_r2)
print(calibration_rmse)
print(calibration_mae)
print(calibration_rpd)


Vmetrics = [validation_r2,validation_rmse,validation_mae,validation_rpd]
Cmetrics = [calibration_r2,calibration_rmse,calibration_mae,calibration_rpd]

##print(model.cv_results_)
cvlist = [model.cv_results_['split0_test_score'], model.cv_results_['split1_test_score'], model.cv_results_['split2_test_score'], model.cv_results_['split3_test_score'],
          model.cv_results_['split4_test_score'],model.cv_results_['split5_test_score'],model.cv_results_['split6_test_score'],
            model.cv_results_['split7_test_score'],model.cv_results_['split8_test_score'],model.cv_results_['split9_test_score']]

pd.DataFrame(cvlist).to_csv("CSV/cvresultsFosforoPorcentagem.csv", header=None, index=None)
pd.DataFrame(Vmetrics).to_csv("CSV/Vmetrics.csv", header=None, index=None)
pd.DataFrame(Cmetrics).to_csv("CSV/Cmetrics.csv", header=None, index=None)

print(model.best_params_)

