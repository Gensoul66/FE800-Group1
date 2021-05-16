import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
def VARloop(train,test):
    exog = train[["ROC_14","Aobv"]]
    mod = sm.tsa.VARMAX(train[["Price","SMA5","SMA10","SMA21","RSI_14","WILLR_14","MACD_SIGNAL","MACD"]], order=(5,0), trend='n', exog=exog)
    res = mod.fit(maxiter=1000, disp=False)
    x = np.arange(0,1500) 
    y = test[["Price"]]
    plt.title("Matplotlib demo") 
    plt.xlabel("time series") 
    plt.ylabel("Price")
    plt.plot(x,y,label = "true")  
    plt.plot(x,res.predict(start=1, end=1500).Price,label = "predict")  
    plt.show()
f = pd.read_csv('ARIMA/rd_AGG.csv')
train = f[["Price","SMA5","SMA10","SMA21","RSI_14","WILLR_14","MACD_SIGNAL","MACD","ROC_14","Aobv","R(t-1)","Vol(t-1)"]][0:1559]
test = f[["Price","SMA5","SMA10","SMA21","RSI_14","WILLR_14","MACD_SIGNAL","MACD","ROC_14","Aobv","R(t-1)","Vol(t-1)"]][0:1500]
VARloop(train,test)
