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

data_test = data[data['Date'] > '2021-12-01'].copy()
data_training = data[data['Date'] < '2021-12-01'].copy()
training_data = data_training.drop(['Date', 'Adj Close'], axis = 1)
scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)

X_train = [] 
Y_train = []
for i in range(60, training_data.shape[0]):
    X_train.append(training_data[i-60:i])
    Y_train.append(training_data[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

#Initialize the RNN
model = Sequential()
model.add(LSTM(5,return_sequences=True,input_shape=(X_train.shape[1], 5)))
model.add(LSTM(10,return_sequences=True))
model.add(LSTM(20))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs = 3, batch_size =32)

part_60_days = data_training.tail(60)
df= part_60_days.append(data_test, ignore_index = True)
df = df.drop(['Date', 'Adj Close'], axis = 1)

inputs = scaler.transform(df)

X_test = []
Y_test = []
for i in range (60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    Y_test.append(inputs[i, 0])
X_test, Y_test = np.array(X_test), np.array(Y_test)
Y_pred = model.predict(X_test)

print("accuracy  = ", 100 - mean_absolute_error(Y_test, Y_pred) * 100)

# Chart
max_val = data_training['Open'].max()
min_val = data_training['Open'].min()

scale = (max_val - min_val)
Y_test = Y_test*scale + min_val
Y_pred = Y_pred*scale + min_val

plt.figure(figsize=(14,5))
plt.plot(Y_test, color = 'red', label = 'Real Bitcoin Price')
plt.plot(Y_pred, color = 'green', label = 'Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction using RNN-LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig("FirstPic.png")

# accuracy  =  93.55376062112887