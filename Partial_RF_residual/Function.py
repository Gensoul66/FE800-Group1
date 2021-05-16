# %%

import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV,KFold
from xgboost import XGBRFRegressor


def transform_data(x, y,train_len):

    # initilize scaler
    x_mean = np.mean(x[0:,:])
    x_std = np.std(x[0:,:])
    y_mean = np.mean(y[0:,:])
    y_std = np.std(y[0:,:])


    # transform
    y_transform = (y - y_mean) / y_std
    x_transform = (x - x_mean) / x_std


    y_scaler = [y_mean, y_std]

    return x_transform, y_transform, y_scaler


def split_sequences1(x_transform, y_transform, date, timesteps) :
    # Initialize x,y
    x = []
    y = []
    date_ = []

    # split the data
    for i in range(timesteps, len(date) + 1) :
        seq_x, seq_y, seq_date = x_transform[i - timesteps:i, :], y_transform[i-1,:], date[i - 1]
        x.append(seq_x)
        y.append(seq_y)
        date_.append(seq_date)

    return np.array(x).astype('float64'), np.array(y).astype('float64'), np.array(date_)

# For ANN model
def split_sequences2(x_transform, y_transform, date, timesteps):
    x = []
    y = []
    date_ = []

    # construct feature per timesteps for a group
    for i in range(timesteps, len(date) + 1):
        y.append(y_transform[i-1,:])
        x.append(x_transform[i - timesteps :i])  # reshape
        date_.append(date[i - 1])

    # convert the into array
    y = np.asarray(y)
    x = np.asarray(x).reshape(len(y), -1)
    date = np.asarray(date_)

    return x, y, date

def rolling_windows_Model(x, y, date, train_len, validation_len, test_len, forward_len):
    total_len = train_len + validation_len + test_len
    forward_times = (x.shape[0] - total_len) // forward_len

    # x
    x_train, x_val, x_test = [], [], []
    # y
    y_train, y_val, y_test = [], [], []
    # date
    date_train, date_val, date_test = [], [], []

    # rolling window
    for i in range(forward_times + 1):
        # train
        x_train.append(x[forward_len * i : forward_len * i + train_len]) # TODO
        y_train.append(y[forward_len * i : forward_len * i + train_len,:])
        date_train.append(date[forward_len * i : forward_len * i + train_len])

        # val
        x_val.append(x[forward_len * i + train_len : forward_len * i + train_len + validation_len]) # TODO
        y_val.append(y[forward_len * i + train_len : forward_len * i + train_len + validation_len,:])
        date_val.append(date[forward_len * i + train_len : forward_len * i + train_len + validation_len])

        # test
        x_test.append(x[forward_len * i + train_len + validation_len : forward_len * i + total_len]) # TODO
        y_test.append(y[forward_len * i + train_len + validation_len : forward_len * i + total_len,:])
        date_test.append(date[forward_len * i + train_len + validation_len : forward_len * i + total_len])

    return x_train, y_train, x_val, y_val, x_test, y_test, date_train, date_val, date_test

def data_generator_lstm(name, timesteps, train_len, val_len, test_len, forward_len) :

    # read the current dir
    current = os.getcwd() + '/processed_data/Weekly_Regression/'

    # load the data
    data = pd.read_csv(current + 'rw_' + name + '.csv')

    # split data into x,y and date
    x = data.iloc[:, 2 :].values
    y = data.iloc[:, 1].values.reshape(-1,1)
    date = data.iloc[:,0].values

    # transform data
    x_transform, y_transform, y_scaler = transform_data(x,y,train_len)

    # split data LSTM
    x_transform, y_transform, date = split_sequences1(x_transform, y_transform, date, timesteps)

    # construct train, validation, test dataset
    x_train, y_train, x_val, y_val, x_test, y_test, date_train, date_val, date_test = rolling_windows_Model(x_transform,
                                                                                                            y_transform,
                                                                                                            date,
                                                                                                            train_len,
                                                                                                            val_len,
                                                                                                            test_len,
                                                                                                            forward_len)

    return x_train, y_train, x_val, y_val, x_test, y_test, date_test, y_scaler


def data_generator_ann(name, features, train_len, val_len, test_len, forward_len) :
    # read the current dir
    current = os.getcwd() + '/processed_data/Weekly_Regression/'

    # load the data
    data = pd.read_csv(current + 'rw_' + name + '.csv')

    # split data into x,y and date
    x = data.iloc[:, 2:].values
    y = data.iloc[:, 1].values.reshape(-1,1)
    date = data.iloc[:,0].values

    # transform data
    x_transform, y_transform, y_scaler = transform_data(x, y,train_len)

    # split data ANN
    x_transform, y_transform, date = split_sequences2(x_transform, y_transform, date, features)

    # construct train, validation, test dataset
    x_train, y_train, x_val, y_val, x_test, y_test, date_train, date_val, date_test = rolling_windows_Model(x_transform,
                                                                                                            y_transform,
                                                                                                            date,
                                                                                                            train_len,
                                                                                                            val_len,
                                                                                                            test_len,
                                                                                                            forward_len)

    return x_train, y_train, x_val, y_val, x_test, y_test, date_test, y_scaler


def residual_generator_lstm(name, timesteps, train_len, val_len, test_len, forward_len):

    # read the current dir
    current = os.getcwd() + '/residual/feature/'

    # load the data
    data = pd.read_csv(current + name + '_Reg.csv')

    # split data into x,y and date
    x = data.iloc[:, 2 :].values
    y = data.iloc[:, 1].values.reshape(-1,1)
    date = data.iloc[:,0].values


    # split data LSTM
    x_transform, y_transform, date = split_sequences1(x, y, date, timesteps)

    # construct train, validation, test dataset
    x_train, y_train, x_val, y_val, x_test, y_test, date_train, date_val, date_test = rolling_windows_Model(x_transform,
                                                                                                            y_transform,
                                                                                                            date,
                                                                                                            train_len,
                                                                                                            val_len,
                                                                                                            test_len,
                                                                                                            forward_len)

    return x_train, y_train, x_val, y_val, x_test, y_test, date_test


def residual_generator_ann(name, features, train_len, val_len, test_len, forward_len):

    # read the current dir
    current = os.getcwd() + '/residual/feature/'

    # load the data
    data = pd.read_csv(current + name + '_Reg.csv')

    # split data into x,y and date
    x = data.iloc[:, 2 :].values
    y = data.iloc[:, 1].values.reshape(-1,1)
    date = data.iloc[:,0].values

    # split data ANN
    x_transform, y_transform, date = split_sequences2(x, y, date, features)

    # construct train, validation, test dataset
    x_train, y_train, x_val, y_val, x_test, y_test, date_train, date_val, date_test = rolling_windows_Model(x_transform,
                                                                                                            y_transform,
                                                                                                            date,
                                                                                                            train_len,
                                                                                                            val_len,
                                                                                                            test_len,
                                                                                                            forward_len)

    return x_train, y_train, x_val, y_val, x_test, y_test, date_test



def generator_price(x_train, y_train, x_val, y_val, x_test, y_test, date_test, y_scaler,
              model, callback, epochs, batch_size, thred) :
    # initialize
    acc = 100
    count = 0
    y_true_transform, y_pred_transform = 0,0

    while acc >= thred:
        y_true = np.array([])
        y_pred = np.array([])
        date = np.array([])

        for i in range(len(y_train)):
            model.fit(x_train[i], y_train[i], validation_data=(x_val[i], y_val[i]),
                      epochs=epochs, batch_size=batch_size, callbacks=[callback], verbose=0)

            predition = model.predict(x_test[i])

            y_true = np.append(y_true, y_test[i])
            y_pred = np.append(y_pred, predition)
            date = np.append(date, date_test[i])

        y_true_transform = y_true * y_scaler[1] + y_scaler[0]
        y_pred_transform = y_pred * y_scaler[1] + y_scaler[0]

        acc = mean_absolute_error(y_true_transform, y_pred_transform)
        count = count + 1
        print(count, acc)

    # plot the graph
    plt.plot(y_true_transform, label='y_true')
    plt.plot(y_pred_transform, label='y_pred')
    plt.legend()

    # save as result
    result = pd.DataFrame(data={'Date': date,
                                'y_true' : y_true_transform,
                                'y_pred' : y_pred_transform})

    return result

def generator_residual(x_train, y_train, x_val, y_val, x_test, y_test, date_test,
              model, epochs, batch_size, thred):
    # initialize
    acc = 100
    count = 0
    y_true = 0
    y_pred = 0

    while acc >= thred:
        y_true = np.array([])
        y_pred = np.array([])
        date = np.array([])

        for i in range(len(y_train)):
            model.fit(x_train[i], y_train[i], validation_data=(x_val[i], y_val[i]),
                      epochs=epochs, batch_size=batch_size,verbose=0)

            predition = model.predict(x_test[i])

            y_true = np.append(y_true, y_test[i])
            y_pred = np.append(y_pred, predition)
            date = np.append(date, date_test[i])

        acc = mean_absolute_error(y_true, y_pred)
        count = count + 1
        print(count, acc)

    # plot the graph
    plt.plot(y_true, label='y_true')
    plt.plot(y_pred, label='y_pred')
    plt.legend()

    # save as result
    result = pd.DataFrame(data={'Date': date,
                                'y_true_residual': y_true,
                                'y_pred_residual': y_pred})

    return result

# Support vector cross validation
def SV_CV(x_train,y_train):

    best_parameter = []
    # Objective Parameters
    param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 5, 10],
                  'gamma': [0.001, 0.01, 0.1, 1, 5, 10]}

    for i in range(len(x_train)):

        x_train_ = x_train[i]  # if dataframe changes, remember to correct
        y_train_ = y_train[i]  # if dataframe changes, remember to correct

        svr_grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid,
                                       cv=KFold(n_splits=5),n_jobs=-1)

        svr_grid_search.fit(x_train_, y_train_)

        best_parameter.append(svr_grid_search.best_params_)

    return best_parameter

# Random forest cross validation
def RF_CV(x_train, y_train) :
    best_parameter = []
    # Objective Parameters
    param_grid = {'n_estimators' : np.arange(100, 250, 50),
                  'max_depth' : np.arange(1, 5, 1),
                  'subsample' : np.arange(0.0, 1.1, 0.1)}

    for i in range(len(x_train)) :
        x_train_ = x_train[i]
        y_train_ = y_train[i]

        rf_grid_search = GridSearchCV(XGBRFRegressor(),
                                      param_grid,
                                      scoring='neg_mean_absolute_error',
                                      cv=KFold(n_splits=5),
                                      n_jobs=-1)

        rf_grid_search.fit(x_train_, y_train_)

        best_parameter.append(rf_grid_search.best_params_)

    return best_parameter

def RF_Prediction_Price(best_paramter, x_train, y_train, x_test, y_test, date_test, y_scaler) :
    y_pred = np.array([])
    y_true = np.array([])
    date = np.array([])

    for i in range(len(x_train)) :
        x_train_ = x_train[i]
        y_train_ = y_train[i]
        x_test_ = x_test[i]
        y_test_ = y_test[i]

        # Date
        date = np.append(date, date_test[i])

        # loading the best parameters
        n_estimators = best_paramter[i]['n_estimators']
        max_depth = best_paramter[i]['max_depth']
        subsample = best_paramter[i]['subsample']

        model = XGBRFRegressor(n_estimators=n_estimators,
                               max_depth=max_depth,
                               subsample=subsample)

        model.fit(x_train_, y_train_)
        y_pred = np.append(y_pred, model.predict(x_test_))
        y_true = np.append(y_true, y_test_)

    y_true_transform = y_true * y_scaler[1] + y_scaler[0]
    y_pred_transform = y_pred * y_scaler[1] + y_scaler[0]
    result = pd.DataFrame(data={'Date' : date.tolist(),
                                'y_true' : y_true_transform.tolist(),
                                'y_pred' : y_pred_transform.tolist()})

    return result


def SV_Prediction_Price(best_paramter,x_train,y_train,x_test,y_test,date_test,y_scaler):

    y_pred = np.array([])
    y_true = np.array([])
    date = np.array([])

    for i in range(len(x_train)):

        x_train_ = x_train[i]
        y_train_ = y_train[i]
        x_test_ = x_test[i]
        y_test_ = y_test[i]

        # Date
        date = np.append(date,date_test[i])

        C = best_paramter[i]['C']
        gamma = best_paramter[i]['gamma']


        # fit the model and save the result
        model = SVR(C=C,gamma=gamma,kernel='rbf')
        model.fit(x_train_, y_train_)
        y_pred = np.append(y_pred, model.predict(x_test_))
        y_true = np.append(y_true, y_test_)

    y_true_transform = y_true * y_scaler[1] + y_scaler[0]
    y_pred_transform = y_pred * y_scaler[1] + y_scaler[0]
    result = pd.DataFrame(data={'Date':date.tolist(),
                                'y_true':y_true_transform.tolist(),
                                'y_pred':y_pred_transform.tolist()})

    return result


def SV_Prediction_Residual(best_paramter,x_train,y_train,x_test,y_test,date_test):

    y_pred = np.array([])
    y_true = np.array([])
    date = np.array([])

    for i in range(len(x_train)):

        x_train_ = x_train[i]
        y_train_ = y_train[i]
        x_test_ = x_test[i]
        y_test_ = y_test[i]

        # Date
        date = np.append(date,date_test[i])

        C = best_paramter[i]['C']
        gamma = best_paramter[i]['gamma']


        # fit the model and save the result
        model = SVR(C=C,gamma=gamma,kernel='rbf')
        model.fit(x_train_, y_train_)
        y_pred = np.append(y_pred, model.predict(x_test_))
        y_true = np.append(y_true, y_test_)

    result = pd.DataFrame(data={'Date':date.tolist(),
                                'y_true':y_true.tolist(),
                                'y_pred':y_pred.tolist()})

    return result

def RF_Prediction_Residual(best_paramter, x_train, y_train, x_test, y_test, date_test, y_scaler) :
    y_pred = np.array([])
    y_true = np.array([])
    date = np.array([])

    for i in range(len(x_train)):
        x_train_ = x_train[i]
        y_train_ = y_train[i]
        x_test_ = x_test[i]
        y_test_ = y_test[i]

        # Date
        date = np.append(date, date_test[i])

        # loading the best parameters
        n_estimators = best_paramter[i]['n_estimators']
        max_depth = best_paramter[i]['max_depth']
        subsample = best_paramter[i]['subsample']

        model = XGBRFRegressor(n_estimators=n_estimators,
                               max_depth=max_depth,
                               subsample=subsample)

        model.fit(x_train_, y_train_)
        y_pred = np.append(y_pred, model.predict(x_test_))
        y_true = np.append(y_true, y_test_)

    result = pd.DataFrame(data={'Date' : date.tolist(),
                                'y_true' : y_true.tolist(),
                                'y_pred' : y_pred.tolist()})

    return result

