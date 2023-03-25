import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.linear_model import LinearRegression
from nsetools import Nse
from pprint import pprint
from csv import writer
import io,base64

def gainers():
    #function to insert things into csv
    def append_list_as_row(file_name, list_of_elem):
        with open(file_name, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(list_of_elem)
    nse= Nse()

    #this list just stores names of the required stocks so that
    #you can use this directly into your code and use accordingly
    gainers_list = []

    #top gainers
    top_gainers = nse.get_top_gainers()
    #pprint(top_gainers)
    
    #remove comment of the next line to see all fields available
    #pprint(top_gainers[0])

    #insert column titles
    row=['Stock name','Open price','High price','Low price','Previous price','Volume','Ltp']
    
    #inserts the columns names into the csv file
    #append_list_as_row('topGainers.csv',row)

    
    #change value of range to receive your required numbers of stocks max-10
    gainers_list.append(row)
    print(top_gainers)
    for i in range(4):
        stock_name=top_gainers[i]['symbol']
        open_price=top_gainers[i]['openPrice']
        previous_price=top_gainers[i]['previousPrice']
        volume=top_gainers[i]['tradedQuantity']
        high_price=top_gainers[i]['highPrice']
        low_price=top_gainers[i]['lowPrice']
        ltp=top_gainers[i]['ltp']
        row=[stock_name,open_price,high_price,low_price,previous_price,volume,ltp]
    
        #file is named topGainers
        #append_list_as_row('topGainers.csv',row)
    
        #adds the names of gainer stocks to the list
        gainers_list.append(row)
    #print(gainers_list)

    
    #top losers
    top_losers = nse.get_top_losers()

    #remove comment of the next line to see all fields available
    #pprint(top_losers[0])

    #insert column titles
    row=['Stock name','Open price','High price','Low price','Previous price','Volume','Ltp']

    #inserts the columns names into the csv file
    #append_list_as_row('topLosers.csv',row)
    
    losers_list=[row]
    #change value of range to receive your required numbers of stocks max-10
    for i in range(5):
        print(top_losers[i])
        stock_name=top_losers[i]['symbol']
        open_price=top_losers[i]['openPrice']
        previous_price=top_losers[i]['previousPrice']
        volume=top_losers[i]['tradedQuantity']
        high_price=top_losers[i]['highPrice']
        low_price=top_losers[i]['lowPrice']
        ltp=top_losers[i]['ltp']
        row=[stock_name,open_price,high_price,low_price,previous_price,volume,ltp]
    
        #file is named topLosers.csv
        #append_list_as_row('topLosers.csv',row)
    
        #adds the name of the loser stocks
        losers_list.append(row)
    
    #remove the next line's comment to see the stock names or use the list in your code directly
    #print()
    #print(loosers_list)
    return {"gainers":gainers_list,"loosers":losers_list}
    

#9KYWPHAM078USX9R

ticker="msft"
#predict(ticker)
pprint(gainers())
