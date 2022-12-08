import pandas as pd
import yfinance as yf 
import os
from binance.client import Client
import pandas as pd
import datetime, time

def get_daily_data(ticker:str, period: str, verbose=False): 
    df = yf.download(tickers=ticker, period=period, interval='1d').reset_index()
    today = yf.download(tickers=ticker, period='1d', interval='1d').reset_index()
    df = pd.concat([df, today])
    df = df.reset_index(drop=True)
    return df

def get_minute_data(ticker:str, period: str, interval=str, verbose=False): 
    df = yf.download(tickers=ticker, period=period, interval=interval).reset_index()
    df = df.rename(columns={'Datetime': 'Date'})
    df['Date'] = df['Date'].astype(str)
    return df

def get_binance_minute_data(ticker: str, days:int):
    client = Client('u5XXkm0giQce0sREaKvWgqDSZCSHdgJRgdqGgctzwcJUL3mXGw7GLw5gKm3OR49X',
    'z6xmexZdvzqFbABmru4yJhtKryEaXUH6KxDeJqAplnYCBGGpqWud8w7I2uw8LoNN')
    # Calculate the timestamps for the binance api function
    untilThisDate = datetime.datetime.now()
    sinceThisDate = untilThisDate - datetime.timedelta(days = days)
    # Execute the query from binance - timestamps must be converted to strings !
    candle = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_5MINUTE, str(sinceThisDate), str(untilThisDate))

    # Create a dataframe to label all the columns returned by binance so we work with them later.
    df = pd.DataFrame(candle, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
    # as timestamp is returned in ms, let us convert this back to proper timestamps.
    df.Date = pd.to_datetime(df.Date, unit='ms')
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    # df.set_index('dateTime', inplace=True)

    # Get rid of columns we do not need
    df = df.drop(['closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol','takerBuyQuoteVol', 'ignore'], axis=1)
    return df
