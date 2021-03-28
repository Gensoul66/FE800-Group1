
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input,LSTM,Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import BayesianOptimization,RandomSearch
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt

n_features = 8

# rolling data function
def rolling_windows(data, train_len, validation_len, test_len, forward_len) :
    total_len = train_len + validation_len + test_len
    forward_times = (data.shape[0] - total_len) // forward_len
    train = []
    validation = []
    test = []

    for i in range(forward_times+1):
        train.append(data.iloc[forward_len * i : forward_len * i + train_len, :])
        validation.append(data.iloc[forward_len * i + train_len : forward_len * i + train_len + validation_len, :])
        test.append(data.iloc[forward_len * i + train_len + validation_len : forward_len * i + total_len, :])

    return train, validation, test


# ANN model
def ANN_model(unit1, dp1, nl1, unit2, dp2, nl2, lr, type):
    ann = Sequential()
    ann.add(Input(shape=(n_features,)))  # 5

    # first layer
    for i in range(nl1) :
        ann.add(Dense(unit1, activation='relu'))
        ann.add(Dropout(dp1))

    # second layer
    for i in range(nl2):
        ann.add(Dense(unit2, activation='relu'))
        ann.add(Dropout(dp2))

    # output dense
    if type == 'regression':
        ann.add(Dense(1))
        ann.compile(
            optimizer=Adam(learning_rate=lr), loss="mse"
        )
    elif type == 'classification':
        ann.add(Dense(2), 'sigmoid')
        ann.compile(
            optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=['accuracy']
        )

    return ann


# ANN tuner
def ANN_Regression_tuner(hp):
    # parameter for layer1
    number_neurons1 = hp.Int('Number_of_neurons1', min_value=64, max_value=128, step=8)
    number_dropout_rate1 = hp.Float("Number_of_dropout_rate1", min_value=0.2, max_value=0.5, step=0.1)
    number_layers1 = hp.Int('Number_of_layer1', min_value=1, max_value=5, step=1)

    # paraemeter for layer2
    number_neurons2 = hp.Int('Number_of_neurons2', min_value=8, max_value=64, step=8)
    number_dropout_rate2 = hp.Float("Number_of_dropout_rate2", min_value=0.2, max_value=0.5, step=0.1)
    number_layers2 = hp.Int('Number_of_layer2', min_value=1, max_value=5, step=1)

    # input layer
    model = Sequential()
    model.add(Input(shape=(n_features,)))  # feature 改变，dimesion也要改 # 5

    # layer 1
    for i in range(number_layers1) :
        model.add(Dense(number_neurons1, activation='relu'))
        model.add(Dropout(number_dropout_rate1))

    # layer 2
    for i in range(number_layers2) :
        model.add(Dense(number_neurons2, activation='relu'))
        model.add(Dropout(number_dropout_rate2))

    # output layer
    model.add(Dense(1))

    lr = hp.Choice("lr", values=[0.1, 0.015, 0.01, 0.0015, 0.001])

    model.compile(
        optimizer=Adam(learning_rate=lr), loss="mse"

    )
    return model

def ANN_Classification_tuner(hp):
    # parameter for layer1
    number_neurons1 = hp.Int('Number_of_neurons1', min_value=64, max_value=128, step=8)
    number_dropout_rate1 = hp.Float("Number_of_dropout_rate1", min_value=0.2, max_value=0.5, step=0.1)
    number_layers1 = hp.Int('Number_of_layer1', min_value=1, max_value=5, step=1)

    # paraemeter for layer2
    number_neurons2 = hp.Int('Number_of_neurons2', min_value=8, max_value=64, step=8)
    number_dropout_rate2 = hp.Float("Number_of_dropout_rate2", min_value=0.2, max_value=0.5, step=0.1)
    number_layers2 = hp.Int('Number_of_layer2', min_value=1, max_value=5, step=1)

    # input layer
    model = Sequential()
    model.add(Input(shape=(n_features,)))  # feature 改变，dimesion也要改 # 5

    # layer 1
    for i in range(number_layers1):
        model.add(Dense(number_neurons1, activation='relu'))
        model.add(Dropout(number_dropout_rate1))

    # layer 2
    for i in range(number_layers2):
        model.add(Dense(number_neurons2, activation='relu'))
        model.add(Dropout(number_dropout_rate2))

    # output layer
    model.add(Dense(1,activation='sigmoid'))

    lr = hp.Choice("lr", values=[0.1, 0.015, 0.01, 0.0015, 0.001])


    model.compile(
        optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=['accuracy']
    )

    return model

def ANN_Regression_CV_result(train, validation,file_name):
    result = []
    best_model = []

    for i in range(len(train)) :
        # loading test, validation and training dataset
        val_data = validation[i]
        train_data = train[i]

        # Split the dataset into x,y
        x_train = train_data.iloc[:, 2:]  # if dataframe changes, remember to correct
        y_train = train_data.iloc[:, 1]

        x_val = val_data.iloc[:, 2:]
        y_val = val_data.iloc[:, 1]

        filenames = "tune_ANN" + str(i) + file_name
        tuner_ANN = BayesianOptimization(
            ANN_Regression_tuner,
            objective='val_loss',
            max_trials=10,
            executions_per_trial=3,
            project_name=filenames,
            overwrite=False
        )

        tuner_ANN.search(
            x=x_train,
            y=y_train,
            verbose=1,
            epochs=100,
            batch_size=30,
            validation_data=(x_val, y_val)
        )

        result.append(tuner_ANN.get_best_hyperparameters()[0].values)
        best_model.append(tuner_ANN.get_best_models(num_models=1)[0])

    return result,best_model

def ANN_Classification_CV_result(train, validation,file_name):
    result = []
    best_model = []

    for i in range(len(train)):
        # loading test, validation and training dataset
        val_data = validation[i]
        train_data = train[i]

        # Split the dataset into x,y
        x_train = train_data.iloc[:, 2:]  # if dataframe changes, remember to correct
        y_train = train_data.iloc[:, 1]

        x_val = val_data.iloc[:, 2:]
        y_val = val_data.iloc[:, 1]

        filenames = "tune_ANN" + str(i) + file_name
        tuner_ANN =BayesianOptimization(
            ANN_Classification_tuner,
            objective='val_accuracy',
            max_trials=10,
            executions_per_trial=3,
            project_name=filenames,
            overwrite=False
        )

        tuner_ANN.search(
            x=x_train,
            y=y_train,
            verbose=1,
            epochs=100,
            batch_size=30,
            validation_data=(x_val, y_val)
        )

        result.append(tuner_ANN.get_best_hyperparameters()[0].values)
        best_model.append(tuner_ANN.get_best_models(num_models=1)[0])

    return result,best_model


def ANN_Prediction(best_model,test,type):
    y_true = np.array([])
    y_pred = np.array([])
    date_test = np.array([])

    for i in range(len(test)):

        # loading test, validation and training dataset
        test_data = test[i]

        # Split the dataset into x,y

        x_test = test_data.iloc[:, 2:]
        y_test = test_data.iloc[:, 1]

        model = best_model[i]
        if type == 'regression':
            y_pred = np.append(y_pred, model.predict(x_test))
            y_true = np.append(y_true, y_test)
            date_test = np.append(date_test, test_data.iloc[:, 0])

        elif type == 'classification':
            y_pred = np.append(y_pred, model.predict_classes(x_test))
            y_true = np.append(y_true, y_test)
            date_test = np.append(date_test, test_data.iloc[:, 0])

    result = pd.DataFrame(data={'Date': date_test, 'y_true': y_true, 'y_pred': y_pred})
    return result