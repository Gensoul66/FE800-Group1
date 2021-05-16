import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
def VARloop(train,test):
    exog = train[["MACD"]]
    mod = sm.tsa.VARMAX(train[["Price","SMA5","SMA10","SMA21","RSI_14","WILLR_14"]], order=(5,0), trend='n', exog=exog)
    res = mod.fit(maxiter=1000, disp=False)
    x = np.arange(0,400) 
    y = test[["Price"]]
    plt.title("Matplotlib demo") 
    plt.xlabel("time series") 
    plt.ylabel("Price")
    plt.plot(x,y,"x-",label = "true")  
    plt.plot(x,res.predict(start=1, end=400).Price,"+-",label = "predict")  
    plt.show()
f = pd.read_csv('ARIMA/rd_AGG.csv')
train = f[["Price","SMA5","SMA10","SMA21","RSI_14","WILLR_14","MACD_SIGNAL","MACD"]][0:1099]
test = f[["Price","SMA5","SMA10","SMA21","RSI_14","WILLR_14","MACD_SIGNAL","MACD"]][1100:1500]
VARloop(train,test)
