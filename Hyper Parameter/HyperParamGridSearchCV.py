import matplotlib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

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
def build_model(var_units_1=10, var_units_2=10, var_units_3=10, var_optimizer='adam'):
  model = Sequential()
  model.add(LSTM(var_units_1,return_sequences=True,input_shape=(X_train.shape[1], 5)))
  model.add(LSTM(var_units_2,return_sequences=True))
  model.add(LSTM(var_units_3))
  model.add(Dense(1))
  
  model.compile(optimizer=var_optimizer, loss = 'mean_squared_error', metrics=['accuracy'])

  return model

_units_1 = [10, 20, 30]
_units_2 = [10, 20, 30]
_units_3 = [10, 20, 30]
_optimizers=['sgd','adam']
_batch_size=[16,32,64]
params=dict(var_units_1=_units_1,
            var_units_2=_units_2,
            var_units_3=_units_3,
            var_optimizer=_optimizers,
            batch_size=_batch_size)
print(params)

model = KerasClassifier(build_fn=build_model,epochs=3,batch_size=16)
rscv = GridSearchCV(model, param_grid=params)
rscv_results = rscv.fit(X_train, Y_train)
print('Best is when using {}'.format(rscv_results.best_params_))

# Best is when using {'var_units_3': 20, 'var_units_2': 20, 'var_units_1': 20, 'var_optimizer': 'adam', 'batch_size': 32}