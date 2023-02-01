import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import yfinance as yf
import time
from datetime import datetime
import random
import os
import io,base64



def predict(ticker,time,epoch,batch,user):
    msft = yf.Ticker(ticker)
    df = msft.history(period=time)
    print(df.shape)

    df = df['Open'].values
    df = df.reshape(-1, 1)

    dataset_train = np.array(df[:int(df.shape[0]*0.8)])
    dataset_test = np.array(df[int(df.shape[0]*0.8):])

    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_train = scaler.fit_transform(dataset_train)
    dataset_test = scaler.transform(dataset_test)

    def create_dataset(df):
        x = []
        y = []
        for i in range(50, df.shape[0]):
            x.append(df[i-50:i, 0])
            y.append(df[i, 0])
        x = np.array(x)
        y = np.array(y)
        return x,y


    x_train, y_train = create_dataset(dataset_train)
    x_test, y_test = create_dataset(dataset_test)

    print(len(x_train))
    print(len(x_test))

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    model = Sequential()
    model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))


    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(x_train, y_train, epochs=epoch, batch_size=batch)
    model.save('stock_prediction.h5')
    model = load_model('stock_prediction.h5')

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    print(predictions[0][0])
    print(y_test_scaled[0][0])
    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_facecolor('#000041')
    ax.plot(y_test_scaled, color='red', label='Original price')
    plt.plot(predictions, color='cyan', label='Predicted price')
    plt.legend()
    #plt.show()
    date=user
    #num=datetime.now()
    #num=num.strftime("%H:%M:%S")
    num=str(random.random())
    num="123"
    #C:\Users\karan mhatre\OneDrive\Desktop\Stocks
    #C:\Users\DELL\Desktop\Stocks\Stocks\static\cache\Data
    path="C:/Users/DELL/Desktop/Stocks/Stocks/static/cache/"+date
    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
    my_base64_jpgData=str(my_base64_jpgData)
    ret=[my_base64_jpgData[2:-1],float(predictions[0][0]),float(y_test_scaled[0][0])]
    """
    if os.path.exists(path):
        if os.path.exists(path+"/"+num):
            plt.savefig(path+"/"+num+"/"+ticker+".png")
        else:
            os.mkdir(path+"/"+num)
            plt.savefig(path+"/"+num+"/"+ticker+".png")
    else:
        os.mkdir(path)
        os.mkdir(path+"/"+num)
        plt.savefig(path+"/"+num+"/"+ticker+".png")

    #cached_img = open(path+"/"+num+"/"+ticker+".png")
    #cached_img_b64 = base64.b64encode(cached_img.read())
    return "cache/"+date+"/"+num+"/"+ticker+".png"
    """
    return ret

#print(predict("GOOGL","10y",1,128,"sarvesh"))


"""
msft = yf.Ticker("GOOGL")
df = msft.history(period="15y")
print(df.shape)

df = df['Open'].values
df = df.reshape(-1, 1)

dataset_train = np.array(df[:int(df.shape[0]*0.8)])
dataset_test = np.array(df[int(df.shape[0]*0.8):])

scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.transform(dataset_test)

def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y


x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)

print(len(x_train))
print(len(x_test))

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


model = Sequential()
model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=10, batch_size=64)
model.save('stock_prediction.h5')
model = load_model('stock_prediction.h5')

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
print(predictions[0][0])
print(y_test_scaled[0][0])
fig, ax = plt.subplots(figsize=(16,8))
ax.set_facecolor('#000041')
ax.plot(y_test_scaled, color='red', label='Original price')
plt.plot(predictions, color='cyan', label='Predicted price')
plt.legend()
plt.show()
"""
"""
import requests
 
# Making a GET request
r = requests.get('https://www.tipranks.com/stocks/aapl/forecast')
 
# check status code for response received
# success code - 200
print(r)
 
# print content of request
print(r.content)

"""





































