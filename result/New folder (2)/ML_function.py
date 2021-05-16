import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.svm import SVR,SVC
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt

def rolling_windows(data ,train_len ,validation_len ,test_len ,forward_len ):

    total_len = train_len + validation_len + test_len
    forward_times = (data.shape[0] - total_len) // forward_len
    train = []
    validation = []
    test = []

    for i in range(forward_times+1):
        train.append(data.iloc[forward_len * i: forward_len * i + train_len,:])
        validation.append(data.iloc[forward_len * i + train_len : forward_len * i + train_len + validation_len,:])
        test.append(data.iloc[forward_len * i + train_len + validation_len : forward_len * i + total_len, :])

    return train,validation,test

# Support vector cross validation
def SV_CV(train,type):

    best_parameter = []
    # Objective Parameters
    param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 5, 10],
                  'gamma': [0.001, 0.01, 0.1, 1, 5, 10]}

    for i in range(len(train)):

        # construct x_train, y_train
        train_data = train[i]
        x_train = train_data.iloc[:, 2:]  # if dataframe changes, remember to correct
        y_train = train_data.iloc[:, 1]  # if dataframe changes, remember to correct

        if type == 'regression':
            svr_grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid,
                                       cv=KFold(n_splits=5), verbose=2, n_jobs=-1)

            svr_grid_search.fit(x_train, y_train)

            best_parameter.append(svr_grid_search.best_params_)

        elif type == 'classification':
            svr_grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid,
                                       cv=KFold(n_splits=5), verbose=2, n_jobs=-1)

            svr_grid_search.fit(x_train, y_train)

            best_parameter.append(svr_grid_search.best_params_)

    return best_parameter

# Random forest cross validation
def RF_CV(train,type):

    best_parameter = []
    # Objective Parameters
    param_grid = {'n_estimators': np.arange(100, 1001, 100),
                  'max_depth': np.arange(1, 16, 1),
                  'max_features': np.arange(1, 8, 2)}

    for i in range(len(train)):

        # construct x_train, y_train
        train_data = train[i]
        x_train = train_data.iloc[:, 2:]  # if dataframe changes, remember to correct
        y_train = train_data.iloc[:, 1]  # if dataframe changes, remember to correct

        if type == 'regression':
            rf_grid_search = GridSearchCV(RandomForestRegressor(), param_grid,
                                      cv=KFold(n_splits=5), verbose=2, n_jobs=-1)

            rf_grid_search.fit(x_train,y_train)

            best_parameter.append(rf_grid_search.best_params_)

        elif type == 'classification':
            rf_grid_search = GridSearchCV(RandomForestClassifier(), param_grid,
                                      cv=KFold(n_splits=5), verbose=2, n_jobs=-1)

            rf_grid_search.fit(x_train, y_train)

            best_parameter.append(rf_grid_search.best_params_)

    return best_parameter


def SV_Prediction(best_paramter,train,test,type):

    y_pred = np.array([])
    y_true = np.array([])
    date = np.array([])

    for i in range(len(train)):

        # construct x_train, y_train and x_test
        test_data = test[i]
        train_data = train[i]
        x_train = train_data.iloc[:, 2:]  # if dataframe changes, remember to correct
        y_train = train_data.iloc[:, 1]  # if dataframe changes, remember to correct
        x_test = test_data.iloc[:, 2:]  # if dataframe changes, remember to correct
        y_test = test_data.iloc[:, 1]  # if dataframe changes, remember to correct

        # Date
        date = np.append(date,test_data.iloc[:, 0])

        C = best_paramter[i]['C']
        gamma = best_paramter[i]['gamma']

        if type == 'regression':

            # fit the model and save the result
            model = SVR(C=C,gamma=gamma,kernel='rbf')
            model.fit(x_train, y_train)
            y_pred = np.append(y_pred, model.predict(x_test))
            y_true = np.append(y_true, y_test.values)


        if type == 'classification':

            # fit the model and save the result
            model = SVC(C=C, gamma=gamma, kernel='rbf')
            model.fit(x_train, y_train)
            y_pred = np.append(y_pred, model.predict(x_test))
            y_true = np.append(y_true, y_test.values)

    result = pd.DataFrame(data={'date':date,'y_pred':y_pred,'y_true':y_true})

    return result


def RF_Prediction(best_paramter,train,test,type):
    y_pred = np.array([])
    y_true = np.array([])
    date = np.array([])

    for i in range(len(train)):

        # construct x_train, y_train and x_test
        test_data = test[i]
        train_data = train[i]
        x_train = train_data.iloc[:, 2:-1]  # if dataframe changes, remember to correct
        y_train = train_data.iloc[:, 1]
        x_test = test_data.iloc[:, 2:-1]
        y_test = test_data.iloc[:, 1]

        # Date
        date = np.append(date, test_data.iloc[:, 0])

        # loading the best parameters
        n_estimators = best_paramter[i]['n_estimators']
        max_depth = best_paramter[i]['max_depth']
        max_features = best_paramter[i]['max_features']

        if type == 'regression':

            model = RandomForestRegressor(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        max_features=max_features
                                          )
            model.fit(x_train, y_train)
            y_pred = np.append(y_pred, model.predict(x_test))
            y_true = np.append(y_true, y_test.values)

        elif type == 'classification':
            model = RandomForestClassifier(n_estimators=n_estimators,
                                          max_depth=max_depth,
                                          max_features=max_features
                                          )
            model.fit(x_train, y_train)
            y_pred = np.append(y_pred, model.predict(x_test))
            y_true = np.append(y_true, y_test.values)

    result = pd.DataFrame(data={'date':date,'y_pred': y_pred, 'y_true' : y_true})
    return result
