# crypto-price-prediction-model
crypto price prediction model

In this project we try to make a better model with a focus on validation
That is my first model befor make it better with validation
![alt text](https://github.com/azibom/crypto-price-prediction-model/blob/master/First%20Model/FirstPic.png)

And that is my second model after validation
![alt text](https://github.com/azibom/crypto-price-prediction-model/blob/master/Final%20Model/FinalPic.png)

The core of my model is sth like that
```
#Initialize the RNN
model = Sequential()
model.add(LSTM(20,return_sequences=True,input_shape=(X_train.shape[1], 4)))
model.add(LSTM(20,return_sequences=True))
model.add(LSTM(20))
model.add(Dense(1))
```

For more details look at the sorce codes ...
