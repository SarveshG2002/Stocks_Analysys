#To import data
import yfinance as yf
import ss
#To do Files related work
import os
import prdict
from send_mail import send_email_otp
#To organize data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import numpy as np
from random import randint

# Web activity
from flask import Flask, render_template, request, jsonify, url_for, redirect,make_response,Response
from socket import gethostname,gethostbyname

#For database
import mysql.connector
#import io

#image incryption
import base64

import time
from datetime import datetime
import random

import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.linear_model import LinearRegression
from nsetools import Nse
from pprint import pprint
from csv import writer
import io



app=Flask(__name__)
session_otp_id=[]




def vada_model(ticker,span,user):
    msft = yf.Ticker(ticker)
    msft_hist = msft.history(period="max")
    DATA_PATH = ticker+".json"
    
    if os.path.exists(DATA_PATH):
        # Read from file if we've already downloaded the data.
        with open(DATA_PATH) as f:
            msft_hist = pd.read_json(DATA_PATH)
    else:
        msft = yf.Ticker(ticker)
        msft_hist = msft.history(period="max")
        # Save file to json in case we need it later.  This prevents us from having to re-download it every time.
        msft_hist.to_json(DATA_PATH)


    print(msft_hist.head(5))

    # Visualize microsoft stock prices
    msft_hist.plot.line(y="Close", use_index=True)
    data = msft_hist[["Close"]]
    data = data.rename(columns = {'Close':'Actual_Close'})

    # Setup our target.  This identifies if the price went up or down
    data["Target"] = msft_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

    msft_prev = msft_hist.copy()
    msft_prev = msft_prev.shift(1)

    # Create our training data
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    data = data.join(msft_prev[predictors]).iloc[1:]
    model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)


    i = 1000
    step = 750
    
    #print(predictions[0].head())

    def backtest(data, model, predictors, start=1000, step=750):
        predictions = []
        # Loop over the dataset in increments
        for i in range(start, data.shape[0], step):
            # Split into train and test sets
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()

            # Fit the random forest model
            model.fit(train[predictors], train["Target"])

            # Make predictions
            preds = model.predict_proba(test[predictors])[:,1]
            preds = pd.Series(preds, index=test.index)
            preds[preds > .5] = 1
            preds[preds<=.5] = 0

            # Combine predictions and test values
            combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)

            predictions.append(combined)

        return pd.concat(predictions)

    weekly_mean = data.rolling(7).mean()["Close"]
    quarterly_mean = data.rolling(90).mean()["Close"]
    annual_mean = data.rolling(365).mean()["Close"]

    weekly_trend = data.shift(1).rolling(7).sum()["Target"]

    data["weekly_mean"] = weekly_mean / data["Close"]
    data["quarterly_mean"] = quarterly_mean / data["Close"]
    data["annual_mean"] = annual_mean / data["Close"]

    data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
    data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]


    data["weekly_trend"] = weekly_trend


    data["open_close_ratio"] = data["Open"] / data["Close"]
    data["high_close_ratio"] = data["High"] / data["Close"]
    data["low_close_ratio"] = data["Low"] / data["Close"]

    full_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "high_close_ratio","low_close_ratio"]

    predictions = backtest(data.iloc[365:], model, full_predictors)
    print(predictions)
    precision_score(predictions["Target"], predictions["Predictions"])

    predictions["Predictions"].value_counts()
    print(predictions["Predictions"].value_counts())
    
    predictions.iloc[-(int(span)):].plot()
    #plt.show()
    #C:\Users\karan mhatre\OneDrive\Desktop\Stocks/New_project/Stocks
    date=user
    #num=datetime.now()
    #num=num.strftime("%H:%M:%S")
    num=str(random.random())
    #C:\Users\karan mhatre\OneDrive\Desktop\Stocks
    #C:\Users\DELL\Desktop\Stocks\Stocks\static\cache\Data
    path="C:/Users/DELL/Desktop/Stocks/Stocks/static/cache/"+date
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


def getLivestocks():
    #stock_info = yf.Ticker('GOOGL').info
    # stock_info.keys() for other properties you can explore
    s=["AAPL","GOOGL","MSFT","TSLA","SBIN.NS"]
    d={}
    for i in s:
        d[i]=yf.Ticker(i).history(period="max")["Open"][len(yf.Ticker(i).history(period="max")["Open"])-1]
        #d.add(i,yf.Ticker(i).history(period="max")["Open"][len(yf.Ticker(i).history(period="max")["Open"])-1])
    print(d)
    #print()
    #print()
    #market_price = stock_info['regularMarketPrice']
    #previous_close_price = stock_info['regularMarketPreviousClose']
    #print('market price ', market_price)
    #print('previous close price ', previous_close_price)
    return d




class dataBase:
    __mydb=mysql.connector.connect(host="localhost",user="root",password="1234",database="stocks")
    __mycursor = __mydb.cursor()
    
    def save_new_user(self,name,mailid,upass):
        print("save user")
        query="insert into user values (%s, %s, %s)"
        value=(name,mailid,upass)
        self.__mycursor.execute(query,value)
        self.__mydb.commit()
        return "done"

    def save_stocks(self,ticker,user):
        query="select * from save where user= %s && ticker=%s"
        value=(user,ticker,)
        self.__mycursor.execute(query,value)
        result=self.__mycursor.fetchall()
        print(result)
        if result==[]:
            #print("save user")
            query="insert into save values (%s, %s)"
            value=(ticker,user)
            self.__mycursor.execute(query,value)
            self.__mydb.commit()
            return "done"
        else:
            return "not"

    def get_recent(self,user):
        query="select ticker from recent where user= %s"
        value=(user,)
        self.__mycursor.execute(query,value)
        result=self.__mycursor.fetchall()
        if result==[]:
            return ""
        return result[0][0]
        
        

    def buy_stocks(self,ticker,ori,user,qua,image):
        query="insert into buy values (%s, %s, %s, %s, %s)"
        value=(ticker,ori,qua,user,image)
        print(ticker)
        self.__mycursor.execute(query,value)
        self.__mydb.commit()
        return "done"
    
    def ver_id_exists(self,mail,pas):
        query="select * from user where mail= %s && pass=%s"
        value=(mail,pas,)
        self.__mycursor.execute(query,value)
        result=self.__mycursor.fetchall()
        print(result)
        if result==[]:
            return "false"
        else:
            return result
    def save_recent(self,user,ticker):
        query="insert into recent values (%s, %s)"
        value=(user,ticker)
        self.__mycursor.execute(query,value)
        self.__mydb.commit()
        return "done"
    def getsavestocks(self,user):
        query="select * from save where user= %s"
        value=(user,)
        self.__mycursor.execute(query,value)
        result=self.__mycursor.fetchall()
        print(result)
        if result==[]:
            return "false"
        else:
            return result
        return 0

    def update_recent(self,user,ticker):
        query="update recent set  ticker= '"+ticker+"' where user=%s"
        value=(user,)
        self.__mycursor.execute(query,value)
        self.__mydb.commit()
        return "done"
    

    def getbuystocks(self,user):
        query="select * from buy where user= %s"
        value=(user,)
        self.__mycursor.execute(query,value)
        result=self.__mycursor.fetchall()
        if result==[]:
            return "false"
        else:
            return result
        return 0


    def delstocks(self,user,tick,price,quant):        
        query="delete from buy where ticker=%s && price=%s && quantity=%s && user=%s"
        value=(tick,price,quant,user,)
        self.__mycursor.execute(query,value)
        self.__mydb.commit()
        print("stick selled")
        print(self.__mycursor.rowcount, "record(s) deleted")
        return "done"

    
    def drop_stock(self,user,tick):        
        query="delete from save where ticker=%s && user=%s"
        value=(tick,user,)
        self.__mycursor.execute(query,value)
        self.__mydb.commit()
        print(self.__mycursor.rowcount, "record(s) deleted")
        return "done"

    
    def sellstocks(self,user,tick,price,quant):
        query="update buy set  quantity= '"+str(quant)+"' where user=%s && ticker=%s"
        value=(user,tick,)
        self.__mycursor.execute(query,value)
        self.__mydb.commit()
        return "done"


@app.route("/")
def start():
    return render_template("main.html")

@app.route("/index")
def index():
    return render_template("index2.html")

@app.route("/getStock/",methods=["POST"])
def getStock():
    req = request.get_json()
    print("Getting....")
    #ret =vada_model(req["ticker"],req["span"],req["user"])
    ret = predict(req["ticker"])
    i=req["ticker"]
    i=get_current_price(i)
    print(i)
    db=dataBase()
    db.update_recent(req["user"],req["ticker"])
    ret.append(i)
    return {"ret":ret}

@app.route("/getlive/",methods=["POST"])
def getlive():
    print("Getting live....")
    ret=getLivestocks()
    return ret

@app.route("/login")
def login():
    return render_template("login.html")
@app.route("/buy_page")
def buy_page():
    return render_template("buy_page.html")

@app.route("/buySell_page")
def buySell_page():
    return render_template("buySell_page.html")
@app.route("/signup")
def signup():
    return render_template("signup.html")

@app.route("/ckeckout")
def co():
    return render_template("payment_submit.html")

@app.route("/withdraw")
def cw():
    return render_template("withdraw.html")


@app.route("/add_user/",methods=["POST"])
def add_user():
    req = request.get_json()
    name=req["name"]
    mail=req["mail"]
    pas=req["pass"]
    db=dataBase()
    ret=db.save_new_user(name,mail,pas)
    ret =db.save_recent(req["mail"],"")
    return ret

@app.route("/ver_user/",methods=["POST"])
def ver_user():
    req = request.get_json()
    mail=req["mail"]
    pas=req["pass"]
    db=dataBase()
    ret=db.ver_id_exists(mail,pas)
    ret={"ret":ret}
    return ret

@app.route("/get_recent/",methods=["POST"])
def get_recent():
    req = request.get_json()
    user=req["user"]
    print("recent",user)
    db=dataBase()
    ret=db.get_recent(user)
    return ret


@app.route("/save_stocks/",methods=["POST"])
def save_stocks():
    req = request.get_json()
    ticker=req["ticker"]
    user=req["user"]
    db=dataBase()
    ret=db.save_stocks(ticker,user)
    return ret


@app.route("/buy_stocks/",methods=["POST"])
def buy_stocks():
    req = request.get_json()
    ticker=req["ticker"]
    ori=req["ori"]
    user=req["user"]
    qua=req["qua"]
    image=req["image"]
    db=dataBase()
    ret=db.buy_stocks(ticker,ori,user,qua,image)
    return ret


@app.route("/getsavestocks/",methods=["POST"])
def getsavestocks():
    req = request.get_json()
    db=dataBase()
    user=req["user"]
    print(user)
    ret=db.getsavestocks(user)
    return {"ret":ret}


@app.route("/drop_stock/",methods=["POST"])
def drop_stock():
    req = request.get_json()
    db=dataBase()
    user=req["user"]
    ticker=req["ticker"]
    print(user)
    ret=db.drop_stock(user,ticker)
    return {"ret":ret}


@app.route("/getbuystocks/",methods=["POST"])
def getbuystocks():
    req = request.get_json()
    db=dataBase()
    user=req["user"]
    ret=db.getbuystocks(user)
    return {"ret":ret}

@app.route("/send_otp/",methods=["POST"])
def controll_car():
    global session_otp_id
    req=request.get_json()
    session_id=randint(1111111,99999999999999)
    otp=send_email_otp(req["email"])
    if (otp=="False"):
        return "False"
    session_otp_id.append({"email":req["email"],"session_id":session_id,"otp":otp})
    return {"otp":str(otp),"session_id":session_id}

@app.route("/check_otp/",methods=["POST"])
def check_otp():
    global session_otp_id
    print("cheking otp")
    req=request.get_json()
    print("printing req: ",req)
    con=False
    for i in session_otp_id:
        print({"email":req["email"],"session_id":int(req["session_id"]),"otp":req["otp"]})
        if(i=={"email":req["email"],"session_id":int(req["session_id"]),"otp":req["otp"]}):
            print("found session")
            print("below line commented")
            # result=db.save_new_user(req["email"],req["name"])
            print("above line commented")
            session_otp_id.remove(i)
            return "true"
        else:
            print("not found")
            return "not found"

"""
@app.route("/sellstocks/",methods=["POST"])
def sellstocks():
    req = request.get_json()
    db=dataBase()
    user=req["user"]
    tick=req["tick"]
    price=req["price"]
    quant=req["quant"]
    print(user)
    try:
        pp=yf.Ticker(tick).history(period="max")["Open"][len(yf.Ticker(tick).history(period="max")["Open"])-1]
    except:
        pp=yf.Ticker(tick).history(period="max")["Open"][len(yf.Ticker(tick).history(period="max")["Open"])-2]
    profit=pp-float(price)
    ret=db.sellstocks(user,tick,price,quant)
    return {"ret":profit}
"""

@app.route("/sellstocks/",methods=["POST"])
def sellstocks():
    req = request.get_json()
    db=dataBase()
    user=req["user"]
    tick=req["tick"]
    price=req["price"]
    quant=int(req["quant"])
    bquant=int(req["bquant"])
    diff=bquant-quant
    if(diff==0):
        ret =db.delstocks(user,tick,price,quant)
        return "0"
    else:
        ret=db.sellstocks(user,tick,price,diff)
        return "1"
    
#gainers()
@app.route("/getgainers/",methods=["POST"])
def getgainers():
    return prdict.gainers()
    
@app.route("/getLiveStockPrice/",methods=["POST"])
def getLiveStockPrice():
    req = request.get_json()
    #tick=req["tick"]
    #tick=tick.split(".")
    #tick=tick[0]
    #tick.upper()
    #print(tick)
    #pp=yf.Ticker(tick).history(period="max")["Open"][len(yf.Ticker(tick).history(period="max")["Open"])-1]
    return str(get_current_price(req["tick"]))

def predict(ticker):
    
    #URL of the API endpoint to fetch stock market data
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol="+ticker+"&apikey=9KYWPHAM078USX9R"

    # Send a GET request to the API endpoint
    response = requests.get(url)

    # Check if the response was successful
    if response.status_code == 200:
        # Parse the response data as a JSON object
        data = response.json()
    
        # Extract the required information from the JSON object
        latest_data = list(data["Time Series (Daily)"].items())[0]
        latest_date = latest_data[0]
        latest_close = float(latest_data[1]["4. close"])
        latest_low = float(latest_data[1]["3. low"])
        latest_high = float(latest_data[1]["2. high"])
        latest_open = float(latest_data[1]["1. open"])
    
        # Print the extracted information
        #print("Latest date:", latest_date)
        #print("Latest close:", latest_close)
    else:
        # If the response was not successful, print the error code
        print("Failed to fetch data from the API. Error code:", response.status_code)

    # Load the stock market data into a pandas DataFrame
    df = pd.DataFrame(data["Time Series (Daily)"].items(), columns=["Date", "Data"])
    df["Close"] = df["Data"].apply(lambda x: float(x["4. close"]))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    # Split the data into training and testing sets
    train = df.iloc[:-30]
    test = df.iloc[-30:]

    # Convert the index to the number of days since January 1st, 1970
    train["Index_Num"] = (train.index - pd.Timestamp("1970-01-01")) / pd.Timedelta(1, 'D')

    # Train a linear regression model on the training data
    model = LinearRegression()
    model.fit(train["Index_Num"].values.reshape(-1, 1), train["Close"].values)



    # Convert the index to the number of days since January 1st, 1970
    test["Index_Num"] = (test.index - pd.Timestamp("1970-01-01")) / pd.Timedelta(1, 'D')

    # Use the trained model to make predictions on the test data
    predictions = model.predict(test["Index_Num"].values.reshape(-1, 1))
    #print("prediction test: ",predictions)

    # Calculate the number of days since January 1st, 1970 for tomorrow's date
    tomorrow = (pd.Timestamp.now() + pd.Timedelta(1, 'D'))
    index_num = (tomorrow - pd.Timestamp("1970-01-01")) / pd.Timedelta(1, 'D')

    # Use the trained model to make a prediction for tomorrow's stock price
    tomorrow_prediction = model.predict(np.array([index_num]).reshape(-1, 1))[0]
    print("Tommorow: ",tomorrow_prediction)

    mse = mean_squared_error(test["Close"], predictions)
    print(mse)
    percent=(100 * np.sqrt(mse) / np.mean(test["Close"]))
    print("accuracy: ",percent)
    eue=percent * (tomorrow_prediction/100)
    print("eue: ",eue)
    #tomorrow_prediction=tomorrow_prediction+eue


    # Use the trained model to make predictions on the test data
    #predictions = model.predict(test.index.to_numpy().reshape(-1, 1))



    # Evaluate the performance of the model by comparing the predictions to the actual prices
    test["Predictions"] = predictions
    error = (test["Close"] - test["Predictions"]).mean()
    #print("Prediction error:", error)
    
    
    plt.plot(train.index, train["Close"], label="Training Data")
    
    # Plot the test data
    plt.plot(test.index, test["Close"], label="Test Data")
    
    # Plot the predicted price for tomorrow
    tomorrow = (pd.Timestamp.now() + pd.Timedelta(1, 'D'))
    plt.plot([tomorrow], [tomorrow_prediction], 'ro', label="Tomorrow's Prediction")
    
    # Add labels and a title to the plot
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Prediction")
    plt.legend()
    
    # Show the plot
    #plt.show()
    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
    my_base64_jpgData=str(my_base64_jpgData)
    plt.close("all")
    return [my_base64_jpgData[2:-1],tomorrow_prediction,latest_close,latest_low,latest_high,latest_open]


def get_current_price(symbol):
    print(symbol)
    api_key = "9KYWPHAM078USX9R"
    function = "GLOBAL_QUOTE"
    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    print(data)
    return data["Global Quote"]["05. price"]




app.run(host=gethostbyname(gethostname()),port="5010")

























