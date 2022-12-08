import pandas as pd
import numpy as np 
# import talib 
import ta 

def engineer_target(close_column: pd.Series, look_ahead: int, threshold, binary=True, verbose=False):
    shift_value = look_ahead * -1 #Invert in order to look ahead 
    target_column = ((close_column.shift(shift_value) - close_column) / close_column) * 100
    if binary==True: target_column = (target_column > threshold).astype(int)
    return target_column


def percent_change(dataframe: pd.DataFrame, close_column: pd.Series, look_back_list: list, verbose=False): 
    for look_back in look_back_list: 
        dataframe['pct_change_'+str(look_back)] = close_column.pct_change(periods=look_back)
    return dataframe

def log_returns(dataframe: pd.DataFrame, close_column: pd.Series, look_back_list: list, verbose=False):
    for look_back in look_back_list: 
        dataframe['log_return_'+str(look_back)] = np.log(close_column / close_column.shift(look_back))
    return dataframe

def rsi(dataframe: pd.DataFrame, close_column: pd.Series, look_back_list: list, verbose=False): 
    for look_back in look_back_list: 
        # dataframe['rsi_'+str(look_back)] = talib.RSI(close_column, timeperiod=look_back)
        rsi_object = ta.momentum.RSIIndicator(close= close_column, window= look_back)
        dataframe['rsi_'+str(look_back)] = rsi_object.rsi()
    return dataframe 

def rate_of_change(dataframe: pd.DataFrame, close_column: pd.Series, look_back_list: list, verbose=False):
    for look_back in look_back_list: 
    #     dataframe['rate_of_change_'+str(look_back)] = talib.ROCR100(close_column, timeperiod=look_back)
        roc_object = ta.momentum.ROCIndicator(close=close_column, window=look_back)
        dataframe['rate_of_change_'+str(look_back)] = roc_object.roc()
    return dataframe

def macd(dataframe: pd.DataFrame, close_column: pd.Series, slow_fast_signal_list: list, verbose=False):
    for tuple in slow_fast_signal_list: 
        macd_object = ta.trend.MACD(close_column, window_slow= tuple[0], window_fast=tuple[1], 
            window_sign=tuple[2], fillna=False)
        dataframe['macd_'+str(tuple[0])+'_'+str(tuple[1])+'_'+str(tuple[2])] = macd_object.macd()
        dataframe['macd_diff_'+str(tuple[0])+'_'+str(tuple[1])+'_'+str(tuple[2])] = macd_object.macd_diff()
        dataframe['macd_signal_'+str(tuple[0])+'_'+str(tuple[1])+'_'+str(tuple[2])] = macd_object.macd_signal()
    return dataframe 

def stochastic_oscillator(dataframe: pd.DataFrame, high_column: pd.Series, low_column: pd.Series,
 close_column: pd.Series, window_smoother_list: list, verbose=False):
    for tuple in window_smoother_list: 
        stoch_object = ta.momentum.StochasticOscillator(high= high_column, low= low_column, close= close_column, 
        window=tuple[0], smooth_window=tuple[1], fillna=False)
        dataframe['stochastic_'+str(tuple[0])+'_'+str(tuple[1])] = stoch_object.stoch()
        dataframe['stochastic_signal_'+str(tuple[0])+'_'+str(tuple[1])] = stoch_object.stoch_signal()
    return dataframe 