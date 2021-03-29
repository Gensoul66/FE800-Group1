import pandas_ta as ta
import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.preprocessing import MinMaxScaler
import os
import copy

# Download ETF stock price

def data_download(stock_ticker,file_names,start_date,end_date,frequence): # 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    stock_daily_price = []
    for i in stock_ticker:
        temp = yf.download(i,start=start_date,end=end_date,interval=frequence)
        temp = temp.dropna()
        temp.index = pd.DatetimeIndex(temp.index).to_period('D')
        stock_daily_price.append(temp)

    # Save the the file
    current_adress = os.getcwd()
    for i in range(len(stock_daily_price)):
        stock_daily_price[i].to_csv(current_adress + "/Price/" + file_names[i],index=True)

    return stock_daily_price

def stock_return(stock_price):

    # Calculate the stock return
    stock_return = []
    for i in stock_price:
        stock_return.append(i['Close'].pct_change())

    return stock_return


def stock_movement(stock_return):

    movement = []
    stock_return_copy = copy.deepcopy(stock_return)
    for i in stock_return_copy:
        move = np.append(np.NaN,np.where(i[1:]>=0,1,0))
        i[0:] = move
        movement.append(i)
    return movement

# smoothing data
def ExpSmoothing(vec):
  fit = SimpleExpSmoothing(vec,initialization_method="estimated").fit()
  return fit.fittedvalues


# pre-processing data
def data_smoothing(stock_price):

    pre_stock_price = copy.deepcopy(stock_price)
    for stock in pre_stock_price:
        for j in range(5):
            stock.iloc[:, j] = ExpSmoothing(stock.iloc[:, j])

    return pre_stock_price


# feature engineering
def feature_engineering(pre_stock_price):
    feature = []
    for i in pre_stock_price :
        # feature engineering

        # new features
        sma5 = i.ta.sma(length=5)
        sma10 = i.ta.sma(length=10)
        sma21 = i.ta.sma(length=21)

        # old feature new features
        macd_signal = i.ta.macd(fast=12, slow=26, signal=9)['MACDs_12_26_9']
        macd = i.ta.macd(fast=12, slow=26,signal=9)['MACD_12_26_9']
        rsi_14 = i.ta.rsi(length=14)
        willr_14 = i.ta.willr(length=14)
        proc = i.ta.roc(length=14)
        aobv = i.ta.aobv()['OBV']
        s_return = i['close'].pct_change()
        s_vol = s_return.rolling(14).std()

        df = pd.concat([sma5,sma10,sma21,
                        rsi_14, willr_14, macd_signal, macd, proc, aobv, s_return, s_vol], axis=1)
        df.columns = ['SMA5','SMA10','SMA21',
                      'RSI_14', 'WILLR_14', 'MACD_SIGNAL','MACD', 'ROC_14', 'Aobv', 'R(t-1)', 'Vol(t-1)']
        feature.append(df)

    return feature

# help(pre_stock_dailyprice[0].ta)

# Min-Max normalization
def data_normalization(feature,save_scaler):
    scaler = MinMaxScaler()

    # save the feature and scaler
    norm_feature = []
    my_scaler = []

    # copy data
    feature_temp = copy.deepcopy(feature)

    if save_scaler == False:
        for i in feature_temp:
            i.iloc[:,0:] = scaler.fit_transform(i.iloc[:,0:])
            norm_feature.append(i.iloc[:,0:])

        return norm_feature

    elif save_scaler == True:
        for i in feature_temp:

            min_max = [np.max(i.iloc[:,0]),np.min(i.iloc[:,0])]
            i.iloc[:,0:] = scaler.fit_transform(i.iloc[:,0:])
            norm_feature.append(i.iloc[:,0:])

            my_scaler.append(min_max)

        return norm_feature,my_scaler


# Combine Y and X and dropna
def data_combination(y, x, type):
    data = []
    if type == 'regression':
        for i in range(len(y)):
            df = pd.concat([y[i]['Close'], x[i]], axis=1)
            df.columns = ['Price',                                          # Close price
                          'SMA5','SMA10','SMA21',                           # new feature
                          'RSI_14', 'WILLR_14', 'MACD_SIGNAL',
                          'MACD', 'ROC_14', 'Aobv', 'R(t-1)', 'Vol(t-1)']
            df = df.dropna()
            data.append(df)

    elif type == 'classification':
        for i in range(len(y)):
            df = pd.concat([y[i], x[i]],axis=1)
            df.columns = ['Label',
                          'SMA5', 'SMA10', 'SMA21',
                          'RSI_14', 'WILLR_14', 'MACD_SIGNAL',
                          'MACD', 'ROC_14', 'Aobv', 'R(t-1)', 'Vol(t-1)']
            df = df.dropna()
            data.append(df)

    return data

def data_shift(data) :
    shift_data = copy.deepcopy(data)
    new_data = []

    for i in shift_data :
        i.iloc[:, 1 :] = i.iloc[:, 1 :].shift(1)
        new_data.append(i.dropna())

    return new_data


# write the file
def data_writer(data, document_name, file_name):
    current_adress = os.getcwd()
    for i in range(len(data)) :
        data[i].to_csv(current_adress + "/" + document_name + "/" + file_name[i] + ".csv", index=True)