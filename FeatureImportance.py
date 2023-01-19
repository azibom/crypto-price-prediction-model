import matplotlib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Get Bitcoin data
data = yf.download(tickers='BTC-USD',start="2018-01-01", interval = '1d')
data = data.reset_index()

data = data.drop(['Date', 'Adj Close'], axis = 1)
data['SMA5'] = data['Close'].rolling(5).mean()
data['SMA9'] = data['Close'].rolling(9).mean()
data['SMA30'] = data['Close'].rolling(30).mean()
data['SMA100'] = data['Close'].rolling(100).mean()
data['SMA200'] = data['Close'].rolling(200).mean()
data.dropna(inplace=True)

plt.figure(figsize=(14,5))
plt.plot(data['Close'], color = 'red', label = 'Close Price')
plt.plot(data['SMA5'], color = 'green', label = 'SMA5 Price')
plt.plot(data['SMA9'], color = 'blue', label = 'SMA9 Price')
plt.plot(data['SMA30'], color = 'yellow', label = 'SMA30 Price')
plt.plot(data['SMA100'], color = 'orange', label = 'SMA100 Price')
plt.plot(data['SMA200'], color = 'gold', label = 'SMA200 Price')
plt.title('Bitcoin Price Prediction using RNN-LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig("FeatureImportance.png")

# Feature selection by variance threshold method
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)
selector = VarianceThreshold(1)
selector.fit(data)
print(data.columns[selector.get_support()])
# Index(['Close', 'Volume', 'SMA30', 'SMA200'], dtype='object')


# Univariate feature selection by SelectKBest
from sklearn.feature_selection import SelectKBest, mutual_info_regression
scaler = MinMaxScaler()
Y_train = data['Close']
X_train = data.drop(['Close'], axis = 1)
selector = SelectKBest(mutual_info_regression, k = 4)
selector.fit(X_train, Y_train)
print(X_train.columns[selector.get_support()])
# Index(['Open', 'High', 'Low', 'SMA5'], dtype='object')


# Recursive Feature Removal (RFE)
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X_train, Y_train)
print(X_train.columns[selector.get_support()])
# Index(['Open', 'High', 'Low', 'SMA5', 'SMA9'], dtype='object')
