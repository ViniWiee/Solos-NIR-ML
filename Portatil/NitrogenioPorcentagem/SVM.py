from numpy import genfromtxt
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
    'svr__gamma': [1.0,10.0,100.0,1000.0,10000.0,1000000.0,10000000.0,10000000.0],
    'svr__C': [1.0,10.0,100.0,1000.0,10000.0,100000.0,1000000.0,100000000.0]
}

model = GridSearchCV(pipe, param_grid = parameters, cv=10)


model.fit(X_train, y_train)

print(model.score(X_train,y_train))

print(model.best_params_)

