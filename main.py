#Default imports 
import pandas as pd
import numpy as np 
import warnings
import ta
from datetime import date, datetime
warnings.filterwarnings("ignore")

#Custom function imports 
from talos.ingest import get_daily_data
from talos.ingest import get_minute_data
from talos.ingest import get_binance_minute_data
from talos.feature_engineering import engineer_target
from talos.feature_engineering import percent_change
from talos.feature_engineering import log_returns
from talos.feature_engineering import rsi
from talos.feature_engineering import rate_of_change
from talos.feature_engineering import macd
from talos.feature_engineering import stochastic_oscillator
from talos.data_engineering import time_series_split
from talos.models import rolling_xgb
from talos.backtest import generate_buy_decision
from talos.backtest import generate_buy_decision_stop_loss
from talos.backtest import p_and_l_fractional_shares
from talos.backtest import p_and_l_fractional_shares_stop_loss
from talos.results import calculate_performance
from talos.results import view_portfolio_df
from talos.results import view_stock_with_decision


def trade_btc(): 
    ticker = 'BTC-USD'
    # df = get_daily_data(ticker= ticker, period='2Y', verbose=True)
    df = get_minute_data(ticker= 'BTC-USD', period= '60d', interval='2m', verbose=False)
    df['target'] = engineer_target(close_column= df['Close'], look_ahead= 15, threshold = 0.3, binary=True)
    df = percent_change(dataframe= df, close_column= df['Close'], look_back_list= [1, 2, 3, 5, 7, 14])
    df = log_returns(dataframe= df, close_column= df['Close'], look_back_list= [1, 2, 3, 5, 7, 14])
    df = rsi(dataframe= df, close_column= df['Close'], look_back_list= [14])
    df = rate_of_change(dataframe= df, close_column= df['Close'], look_back_list= [2, 3, 5, 7, 12, 14, 20])
    df = macd(dataframe= df, close_column= df['Close'], slow_fast_signal_list= [(26, 12, 9)])
    df = stochastic_oscillator(dataframe= df, high_column= df['High'], low_column=df['Low'], 
        close_column=df['Close'], window_smoother_list=[(14, 3)])
    ta.add_all_ta_features(df, open="Open", high="High", low="Low",
                close = "Close", volume="Volume", fillna=True)

    df = df[200:].reset_index(drop=True) #drop rows where technicals empty 
    train_df_dict, test_df_dict = time_series_split(stock_df=df, train_days=1000, test_days=100)
    predictors = list(df)
    predictors.remove('Date')
    predictors.remove('target')
    final_df = rolling_xgb(predictors= predictors, target_col= 'target', pred_col_name= 'xgb_classifier_predictions',  train_df_dict= train_df_dict, 
        test_df_dict= test_df_dict, cutoff_threshold= 0.5, random_state=1)
    final_df = generate_buy_decision(dataframe= final_df, trade_signal_col_name= 'xgb_classifier_predictions')
    # final_df = generate_buy_decision_stop_loss(dataframe= final_df, close_column= 'Close', trade_signal_col_name= 'xgb_classifier_predictions',
    #      stop_loss_percent= 0.00125, verbose=True)
    final_df = p_and_l_fractional_shares(dataframe= final_df,  close_col= 'Close', buy_decision_col='buy_decision',
        initial_cash= 100000, commission_percent=0.00000, verbose=True)
    calculate_performance(dataframe= final_df, close_col= 'Close', total_portfolio_value_col= 'total_portfolio_value', verbose=True)
    view_portfolio_df(stock_df= final_df, total_portfolio_value_col= 'total_portfolio_value', ticker=ticker)
    view_stock_with_decision(stock_df= final_df, ticker=ticker, close_col='Close', buy_decision_col='buy_decision')
    return final_df




def minute_trader(ticker: str, period, interval: str, look_ahead_target: int, threshold_target: float, percent_change_list: list,
    log_returns_list: list, rsi_list: list, rate_of_change_list: list, macd_list: list, stochastic_list: list, add_all_ta_features: bool, 
    day_of_week: bool, train_periods: int, test_periods: int, xgb_cutoff: float, initial_cash: int, commission_percent: int, moving_stop_loss_points: float): 

    # df = get_daily_data(ticker= ticker, period='2Y', verbose=True)
    # df = get_minute_data(ticker= ticker, period= period, interval=interval, verbose=False)
    df = get_binance_minute_data(ticker= ticker, days= period)
    df['target'] = engineer_target(close_column= df['Close'], look_ahead= look_ahead_target, threshold = threshold_target, binary=True)
    df = percent_change(dataframe= df, close_column= df['Close'], look_back_list= percent_change_list)
    df = log_returns(dataframe= df, close_column= df['Close'], look_back_list= log_returns_list)
    df = rsi(dataframe= df, close_column= df['Close'], look_back_list= rsi_list)
    df = rate_of_change(dataframe= df, close_column= df['Close'], look_back_list= rate_of_change_list)
    df = macd(dataframe= df, close_column= df['Close'], slow_fast_signal_list= macd_list)
    df = stochastic_oscillator(dataframe= df, high_column= df['High'], low_column=df['Low'], 
        close_column=df['Close'], window_smoother_list= stochastic_list)
    if add_all_ta_features: ta.add_all_ta_features(df, open="Open", high="High", low="Low",
                close = "Close", volume="Volume", fillna=True)
    if day_of_week: 
        df['Date'] = pd.to_datetime(df['Date'])
        df['day_of_week'] = df['Date'].dt.dayofweek

    df = df[200:].reset_index(drop=True) #drop rows where technicals empty 
    train_df_dict, test_df_dict = time_series_split(stock_df=df, train_days=train_periods, test_days=test_periods)
    predictors = list(df)
    predictors.remove('Date')
    predictors.remove('target')
    final_df = rolling_xgb(predictors= predictors, target_col= 'target', pred_col_name= 'xgb_classifier_predictions',  train_df_dict= train_df_dict, 
        test_df_dict= test_df_dict, cutoff_threshold= xgb_cutoff, random_state=1)
    # final_df = generate_buy_decision(dataframe= final_df, trade_signal_col_name= 'xgb_classifier_predictions')
    final_df = generate_buy_decision_stop_loss(dataframe= final_df, close_column= 'Close', trade_signal_col_name= 'xgb_classifier_predictions',
         stop_loss_percent= 100, verbose=True)
    final_df = p_and_l_fractional_shares(dataframe= final_df,  close_col= 'Close', buy_decision_col='buy_decision',
        initial_cash= initial_cash, commission_percent=commission_percent, verbose=True)
    # final_df = p_and_l_fractional_shares_stop_loss(dataframe= final_df,  close_col= 'Close', stop_loss_points=moving_stop_loss_points, buy_decision_col='buy_decision', initial_cash= initial_cash, commission_percent= commission_percent, verbose=True)

    calculate_performance(dataframe= final_df, close_col= 'Close', total_portfolio_value_col= 'total_portfolio_value', verbose=True)
    view_portfolio_df(stock_df= final_df, total_portfolio_value_col= 'total_portfolio_value', ticker=ticker)
    view_stock_with_decision(stock_df= final_df, ticker=ticker, close_col='Close', buy_decision_col='buy_decision')
    return final_df




def daily_trader(ticker: str, period: str, look_ahead_target: int, threshold_target: float, percent_change_list: list,
    log_returns_list: list, rsi_list: list, rate_of_change_list: list, macd_list: list, stochastic_list: list, add_all_ta_features: bool, 
    day_of_week: bool, train_days: int, test_days: int, xgb_cutoff: float, initial_cash: int, comission_percent: int): 

    #Get data, feature engineering 
    df = get_daily_data(ticker= ticker, period=period, verbose=True)
    df['target'] = engineer_target(close_column= df['Close'], look_ahead= look_ahead_target, threshold = threshold_target, binary=True)
    df = percent_change(dataframe= df, close_column= df['Close'], look_back_list= percent_change_list)
    df = log_returns(dataframe= df, close_column= df['Close'], look_back_list= log_returns_list)
    df = rsi(dataframe= df, close_column= df['Close'], look_back_list= rsi_list)
    df = rate_of_change(dataframe= df, close_column= df['Close'], look_back_list= rate_of_change_list)
    df = macd(dataframe= df, close_column= df['Close'], slow_fast_signal_list= macd_list)
    df = stochastic_oscillator(dataframe= df, high_column= df['High'], low_column=df['Low'], 
        close_column=df['Close'], window_smoother_list=stochastic_list)
    if add_all_ta_features: ta.add_all_ta_features(df, open="Open", high="High", low="Low",
                close = "Close", volume="Volume", fillna=True)
    if day_of_week: 
        df['day_of_week'] = df['Date'].dt.dayofweek

    #Time series data engineering 
    df = df[200:].reset_index(drop=True) #drop rows where technicals empty 
    df = df[df['Date'] >= datetime(2020,1,1)]
    train_df_dict, test_df_dict = time_series_split(stock_df=df, train_days=train_days, test_days=test_days, verbose=True)
    predictors = list(df)
    predictors.remove('Date')
    predictors.remove('target')
    df['Date'] = pd.to_datetime(df['Date'])

    #Modeling 
    final_df = rolling_xgb(predictors= predictors, target_col= 'target', pred_col_name= 'xgb_classifier_predictions',  train_df_dict= train_df_dict, 
        test_df_dict= test_df_dict, cutoff_threshold= xgb_cutoff, random_state=1, verbose=True)

    #Normalize backtest dates, calculate performance 
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    final_df = final_df[final_df['Date'] >= datetime(2021,6,1)]
    final_df = generate_buy_decision(dataframe= final_df, trade_signal_col_name= 'xgb_classifier_predictions')

    #Get most recent prediction
    final_df = final_df.reset_index(drop=True)
    date = final_df['Date'].iloc[len(final_df)-1]
    prediction = final_df['xgb_classifier_predictions'].iloc[len(final_df)-1]
    buy_decision = final_df['buy_decision'].iloc[len(final_df)-1]
    print('Date: {}, Prediction: {}, Buy Decision: {}'.format(date, prediction, buy_decision))

    #Finish calculating performance
    final_df = p_and_l_fractional_shares(dataframe= final_df,  close_col= 'Close', buy_decision_col='buy_decision',
        initial_cash= initial_cash, commission_percent=comission_percent, verbose=True)
    calculate_performance(dataframe= final_df, close_col= 'Close', total_portfolio_value_col= 'total_portfolio_value', verbose=True)
    view_portfolio_df(stock_df= final_df, total_portfolio_value_col= 'total_portfolio_value', ticker=ticker)
    view_stock_with_decision(stock_df= final_df, ticker=ticker, close_col='Close', buy_decision_col='buy_decision')

    return final_df




#Optimize each stock 
#Loop over all stocks in the SP500, sort by profit, grab the first 20 

if __name__ == '__main__': 


    ung = daily_trader(ticker= 'UNG', period= '5Y', look_ahead_target= 1, threshold_target= 0.08, #0.08
            percent_change_list= [1, 2, 3, 5, 7, 14],
            log_returns_list= [1, 2, 3, 5, 7, 14], 
            rsi_list= [3, 7, 14, 21], 
            rate_of_change_list= [2, 3, 5, 7, 12, 14, 20], 
            macd_list= [(26, 12, 9)], 
            stochastic_list= [(14, 3)], 
            add_all_ta_features= False, 
            day_of_week = True, 
            train_days= 100, test_days= 10, 
            xgb_cutoff= 0.7, #0.7
            initial_cash= 100000, comission_percent= 0.00070)

    ung_buy_decision = ung['buy_decision'].iloc[-1]
    print(ung_buy_decision)

    # boil = daily_trader(ticker= 'BOIL', period= '5Y', look_ahead_target= 1, threshold_target= 0.7, 
    #     percent_change_list= [1, 2, 3, 5, 7, 14],
    #     log_returns_list= [1, 2, 3, 5, 7, 14], 
    #     rsi_list= [14], 
    #     rate_of_change_list= [2, 3, 5, 7, 12, 14, 20], 
    #     macd_list= [(13, 6, 4),(26, 12, 9)], 
    #     stochastic_list= [(14, 3), (26, 12)], 
    #     add_all_ta_features= False, 
    #     day_of_week = True, 
    #     train_days= 100, test_days= 10, 
    #     xgb_cutoff= 0.62, 
    #     initial_cash= 100000, comission_percent= 0.00070)

    # mstr = daily_trader(ticker= 'MSTR', period= '5Y', look_ahead_target= 1, threshold_target=0.7, #0.7
    #         percent_change_list= [1, 2, 3, 5, 7, 14],
    #         log_returns_list= [1, 2, 3, 5, 7, 14], 
    #         rsi_list= [14], 
    #         rate_of_change_list= [2, 3, 5, 7, 12, 14, 20], 
    #         macd_list= [(13, 6, 4), (26, 12, 9)], 
    #         stochastic_list= [(14, 3)], 
    #         add_all_ta_features= True, 
    #         day_of_week = True, 
    #         train_days= 310, test_days= 10, 
    #         xgb_cutoff= 0.85, 
    #         initial_cash= 100000, comission_percent= 0.00070)

    # tsla = daily_trader(ticker= 'TSLA', period= '5Y', look_ahead_target= 1, threshold_target=0.07, 
    #     percent_change_list= [1, 2, 3, 5, 7, 14, 20],
    #     log_returns_list= [1, 2, 3, 5, 7, 14, 20], 
    #     rsi_list= [14], 
    #     rate_of_change_list= [1, 2, 3, 5, 7, 12, 14, 20], 
    #     macd_list= [(13, 6, 4), (26, 12, 9)], 
    #     stochastic_list= [(14, 3)], 
    #     add_all_ta_features= False, 
    #     day_of_week = False, 
    #     train_days= 200, test_days= 10, 
    #     xgb_cutoff= 0.4, 
    #     initial_cash= 100000, comission_percent= 0.00070)

    # nail = daily_trader(ticker= 'NAIL', period= '4Y', look_ahead_target= 1, threshold_target= 0.22, 
    #         percent_change_list= [1, 2, 3, 5, 7, 14, 20],
    #         log_returns_list= [1, 2, 3, 5, 7, 14, 20], 
    #         rsi_list= [14], 
    #         rate_of_change_list= [2, 3, 5, 7, 12, 14, 20], 
    #         macd_list= [(13, 6, 4), (26, 12, 9)], 
    #         stochastic_list= [(14, 3)], 
    #         add_all_ta_features= False, 
    #         day_of_week = False, 
    #         train_days= 310, test_days= 10, 
    #         xgb_cutoff= 0.65, 
    #         initial_cash= 100000, comission_percent= 0.00070)










    # nvda = daily_trader(ticker= 'NVDA', period= '4Y', look_ahead_target= 1, threshold_target= 0.9, 
    #         percent_change_list= [1, 2, 3, 5, 7, 14, 20],
    #         log_returns_list= [1, 2, 3, 5, 7, 14, 20], 
    #         rsi_list= [14], 
    #         rate_of_change_list= [2, 3, 5, 7, 12, 14, 20], 
    #         macd_list= [(13, 6, 4)], 
    #         stochastic_list= [(14, 3)], 
    #         add_all_ta_features= False, 
    #         day_of_week = True, 
    #         train_days= 100, test_days= 10, 
    #         xgb_cutoff= 0.55, 
    #         initial_cash= 100000, comission_percent= 0.00070)
    

    # spy = daily_trader(ticker= 'SPY', period= '5Y', look_ahead_target= 1, threshold_target=0.015, #0.015
    #     percent_change_list= [1, 2, 3, 5, 7],
    #     log_returns_list= [1, 2, 3, 5, 7], 
    #     rsi_list= [14], 
    #     rate_of_change_list= [1, 2, 3, 5, 7, 12, 14, 20], 
    #     macd_list= [(26, 12, 9)], 
    #     stochastic_list= [(14, 3)], 
    #     add_all_ta_features= True, 
    #     day_of_week = True, 
    #     train_days= 200, test_days= 10, 
    #     xgb_cutoff= 0.52, 
    #     initial_cash= 100000, comission_percent= 0.00070)








 
  






    # test = daily_trader(ticker= 'GME', period= '3Y', look_ahead_target= 1, threshold_target=0.00, 
    #     percent_change_list= [1, 2, 3, 5, 7, 14],
    #     log_returns_list= [1, 2, 3, 5, 7, 14], 
    #     rsi_list= [14], 
    #     rate_of_change_list= [1, 2, 3, 5, 7, 12, 14, 20], 
    #     macd_list= [(13, 6, 4), (26, 12, 9)], 
    #     stochastic_list= [(14, 3)], 
    #     add_all_ta_features= False, 
    #     day_of_week = True, 
    #     train_days= 300, test_days= 10, 
    #     xgb_cutoff= 0.50, 
    #     initial_cash= 100000, comission_percent= 0.00070)


  






    # btc = minute_trader(ticker= 'BTCUSDT', period= 100, interval='5m', look_ahead_target= 2, threshold_target= 0.1, #best threshold target is 0.2
    #     percent_change_list= [1, 2, 3, 5, 7, 14, 21, 50, 200],
    #     log_returns_list= [1, 2, 3, 5, 7, 14, 21, 50, 200], 
    #     rsi_list= [7, 14, 21, 50, 200], 
    #     rate_of_change_list= [2, 3, 5, 7, 12, 14, 21, 50, 200], 
    #     macd_list= [(13, 6, 4), (26, 12, 9), (52, 24, 18)], 
    #     stochastic_list= [(14, 3), (28, 6), (56, 12)], 
    #     add_all_ta_features= True, 
    #     day_of_week = False, 
    #     train_periods= 1000, test_periods= 10, 
    #     xgb_cutoff= 0.95,
    #     initial_cash= 100000, commission_percent= 0.0002, moving_stop_loss_points= 2000)




























