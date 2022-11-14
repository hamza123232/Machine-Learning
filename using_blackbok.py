#how to get stock data in python?   
import datetime as dt
import pandas_datareader.data as web

ticker = '^VIX'

start = dt.datetime(1990, 1, 1)
end   = dt.datetime(2022, 5, 1)  # almost today

data = web.DataReader(ticker, 'yahoo', start, end)

close_prices = data['Close']
print(close_prices)

last_price = close_prices[-1]                    
print('last:', last_price)                       

returns = data.pct_change()
print(returns)

close_returns = close_prices.pct_change()
print(close_returns)


import yfinance as yf

data = yf.download('^VIX', period="8d", interval='1h')

print(data)


