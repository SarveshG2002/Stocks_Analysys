import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error

def download_data(ticker, start_date, end_date, file_name):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(file_name)

def stock_price_prediction(file_name, target_column, sequence_length):
    # Load the data
    data = pd.read_csv(file_name)
    
    # Prepare the data
    data = data[[target_column]]
    data = data.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    
    # Split the data into training and testing sets
    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size
    train = data[0:train_size, :]
    test = data[train_size:len(data), :]
    
    # Prepare the training data
    def create_dataset(dataset, sequence_length=sequence_length):
        X = []
        y = []
        for i in range(len(dataset) - sequence_length):
            a = dataset[i:i + sequence_length, 0]
            X.append(a)
            y.append(dataset[i + sequence_length, 0])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_dataset(train, sequence_length)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)
    
    # Prepare the test data
    X_test, y_test = create_dataset(test, sequence_length)
    X_test = np


download_data()
