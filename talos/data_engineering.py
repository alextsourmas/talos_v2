import pandas as pd
import numpy as np 
import math 

def time_series_split(stock_df: pd.DataFrame, train_days: int, test_days: int, verbose=False): 
    stock_df = stock_df.reset_index(drop=True) #Very important to reset the index 
    total_days = len(stock_df)
    rounds = math.ceil((total_days - train_days) / test_days)
    train_df_dict = {}
    test_df_dict = {}
    moving_start = 0
    for round in range(0, rounds):
        train_df = stock_df[moving_start : moving_start + train_days]
        test_df = stock_df[moving_start + train_days : moving_start + train_days + test_days]
        train_df_dict[round] = train_df
        test_df_dict[round] = test_df
        moving_start = moving_start + test_days 
    return train_df_dict, test_df_dict
