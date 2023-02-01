#To import data
import yfinance as yf
import ss
#To do Files related work
import os

#To organize data
import pandas as pd

#To make prediction
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import numpy as np

# Web activity
from flask import Flask, render_template, request, jsonify, url_for, redirect,make_response,Response
from socket import gethostname,gethostbyname

#For database
import mysql.connector
import io

#image incryption
import base64

import time
from datetime import datetime
import random

app=Flask(__name__)





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
    s=["AAPL","GOOGL","MSFT","TSLA","TTM"]
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

    def save_stocks(self,ticker,span,epoch,user):
        print("save user")
        query="insert into save values (%s, %s, %s, %s)"
        value=(ticker,span,epoch,user)
        self.__mycursor.execute(query,value)
        self.__mydb.commit()
        return "done"

    def buy_stocks(self,ticker,ori,user,qua):
        print("save user")
        query="insert into buy values (%s, %s, %s, %s)"
        value=(ticker,ori,qua,user)
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
    def save_recent(self,mail,name,folder,date):
        return 0
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

    def getbuystocks(self,user):
        query="select * from buy where user= %s"
        value=(user,)
        self.__mycursor.execute(query,value)
        result=self.__mycursor.fetchall()
        print(result)
        if result==[]:
            return "false"
        else:
            return result
        return 0
    def sellstocks(self,user,tick,price,quant):
        query="delete from buy where ticker=%s && price=%s && quantity=%s && user=%s"
        value=(tick,price,quant,user,)
        self.__mycursor.execute(query,value)
        self.__mydb.commit()
        print("stick selled")
        print(self.__mycursor.rowcount, "record(s) deleted")
        return "done"


@app.route("/")
def start():
    return render_template("main.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/getStock/",methods=["POST"])
def getStock():
    req = request.get_json()
    print("Getting....")
    #ret =vada_model(req["ticker"],req["span"],req["user"])
    ret = ss.predict(req["ticker"],req["span"]+"y",int(req["epoch"]),64,req["user"])
    return {"ret":ret}

@app.route("/getlive/",methods=["POST"])
def getlive():
    print("Getting live....")
    ret=getLivestocks()
    return ret

@app.route("/login")
def login():
    return render_template("login.html")
@app.route("/signup")
def signup():
    return render_template("signup.html")


@app.route("/add_user/",methods=["POST"])
def add_user():
    req = request.get_json()
    name=req["name"]
    mail=req["mail"]
    pas=req["pass"]
    db=dataBase()
    ret=db.save_new_user(name,mail,pas)
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


@app.route("/save_stocks/",methods=["POST"])
def save_stocks():
    req = request.get_json()
    ticker=req["ticker"]
    span=req["pre"]
    epoch=req["ori"]
    user=req["user"]
    db=dataBase()
    ret=db.save_stocks(ticker,span,epoch,user)
    return ret


@app.route("/buy_stocks/",methods=["POST"])
def buy_stocks():
    req = request.get_json()
    ticker=req["ticker"]
    ori=req["ori"]
    user=req["user"]
    qua=req["qua"]
    db=dataBase()
    ret=db.buy_stocks(ticker,ori,user,qua)
    return ret


@app.route("/getsavestocks/",methods=["POST"])
def getsavestocks():
    req = request.get_json()
    db=dataBase()
    user=req["user"]
    print(user)
    ret=db.getsavestocks(user)
    return {"ret":ret}

@app.route("/getbuystocks/",methods=["POST"])
def getbuystocks():
    req = request.get_json()
    db=dataBase()
    user=req["user"]
    print(user)
    ret=db.getbuystocks(user)
    return {"ret":ret}

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


app.run(host=gethostbyname(gethostname()))

























