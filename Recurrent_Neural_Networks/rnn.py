# Recurrent Neural Network

# Part 1 - Data Preprocessing
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the training data
dataset_train = pd.read_csv('D:\Coding_wo_cp\Deep_Learning_A_Z\Volume 1 - Supervised Deep Learning\Part 3 - Recurrent Neural Networks (RNN)\Section 12 - Building a RNN\Recurrent_Neural_Networks\Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# creating a data stucture with 60 time steps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1) )

# Part 2 - Building the RNN 
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

regressor = Sequential()
# 1st layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1) ))
regressor.add(Dropout(0.2))# some fraction of neurons will be dropped out at each iteration of training

# 2nd layer
regressor.add(LSTM(units = 50, return_sequences = True)) # return_sequences = True is done when we have
# to add another LSTM layer after te current layer
regressor.add(Dropout(0.2))

# 3rd layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# 4th layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# output layer
regressor.add(Dense(units = 1))

# compilation
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Part - 3 Making predictions

# Getting real stock price 
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values 

# Predicting stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60 :].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

# creating a data stucture with 60 time steps and 1 output
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real google stock price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted google stock price')
plt.title("Google Stock Price Prediction")
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
























