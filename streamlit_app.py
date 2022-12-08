#Default imports 
import streamlit as st
import pandas as pd
import numpy as np 
import warnings
import ta
from datetime import date, datetime
import matplotlib.pyplot as plt
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



st.markdown("<h1 style=color: white;'>TALOS V2</h1>", unsafe_allow_html=True)
st.markdown("<h3 style=color: white;'>Trade the markets using machine learning.</h3>", unsafe_allow_html=True)
if st.button('TRADE'): 

    st.write('Running UNG...')

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

    st.write('UNG completed. Running BOIL...')

    boil = daily_trader(ticker= 'BOIL', period= '5Y', look_ahead_target= 1, threshold_target= 0.7, 
        percent_change_list= [1, 2, 3, 5, 7, 14],
        log_returns_list= [1, 2, 3, 5, 7, 14], 
        rsi_list= [14], 
        rate_of_change_list= [2, 3, 5, 7, 12, 14, 20], 
        macd_list= [(13, 6, 4),(26, 12, 9)], 
        stochastic_list= [(14, 3), (26, 12)], 
        add_all_ta_features= False, 
        day_of_week = True, 
        train_days= 100, test_days= 10, 
        xgb_cutoff= 0.62, 
        initial_cash= 100000, comission_percent= 0.00070)

    st.write('BOIL completed. Running MSTR...')

    mstr = daily_trader(ticker= 'MSTR', period= '5Y', look_ahead_target= 1, threshold_target=0.7, #0.7
            percent_change_list= [1, 2, 3, 5, 7, 14],
            log_returns_list= [1, 2, 3, 5, 7, 14], 
            rsi_list= [14], 
            rate_of_change_list= [2, 3, 5, 7, 12, 14, 20], 
            macd_list= [(13, 6, 4), (26, 12, 9)], 
            stochastic_list= [(14, 3)], 
            add_all_ta_features= True, 
            day_of_week = True, 
            train_days= 310, test_days= 10, 
            xgb_cutoff= 0.85, 
            initial_cash= 100000, comission_percent= 0.00070)

    st.write('MSTR Completed. Running TSLA...')


    tsla = daily_trader(ticker= 'TSLA', period= '5Y', look_ahead_target= 1, threshold_target=0.07, 
        percent_change_list= [1, 2, 3, 5, 7, 14, 20],
        log_returns_list= [1, 2, 3, 5, 7, 14, 20], 
        rsi_list= [14], 
        rate_of_change_list= [1, 2, 3, 5, 7, 12, 14, 20], 
        macd_list= [(13, 6, 4), (26, 12, 9)], 
        stochastic_list= [(14, 3)], 
        add_all_ta_features= False, 
        day_of_week = False, 
        train_days= 200, test_days= 10, 
        xgb_cutoff= 0.4, 
        initial_cash= 100000, comission_percent= 0.00070)

    st.write('TSLA Completed. Running NAIL...')


    nail = daily_trader(ticker= 'NAIL', period= '4Y', look_ahead_target= 1, threshold_target= 0.22, 
            percent_change_list= [1, 2, 3, 5, 7, 14, 20],
            log_returns_list= [1, 2, 3, 5, 7, 14, 20], 
            rsi_list= [14], 
            rate_of_change_list= [2, 3, 5, 7, 12, 14, 20], 
            macd_list= [(13, 6, 4), (26, 12, 9)], 
            stochastic_list= [(14, 3)], 
            add_all_ta_features= False, 
            day_of_week = False, 
            train_days= 310, test_days= 10, 
            xgb_cutoff= 0.65, 
            initial_cash= 100000, comission_percent= 0.00070)

    names = ['UNG', 'BOIL', 'MSTR', 'NAIL', 'TSLA']
    dataframes = [ung, boil, mstr, nail, tsla]
    # names = ['UNG']
    # dataframes = [ung]
    results_dfs = []
    for i in range(0, len(dataframes)): 
        dataframe = dataframes[i]
        last_row_df = dataframe[dataframe['buy_decision'].isin(['buy', 'sell'])]
        buy_decision = last_row_df['buy_decision'].iloc[-1]
        if buy_decision == 'buy': 
            st.markdown("<h3 style = color:Green>" + names[i] + ": " + "BUY" "</h3>", unsafe_allow_html=True)
        if buy_decision == 'sell': 
            st.markdown("<h3 style = color:Red>" + names[i] + ": " + "SELL" "</h3>", unsafe_allow_html=True)
        results_dfs.append(dataframe)

    for i in range(0, len(results_dfs)): 

        dataframe = results_dfs[i]
        st.markdown("<h2 style = color:White>" + names[i] + " " + 'RESULTS ANALYSIS' + "</h2>", unsafe_allow_html=True)


        dataframe = dataframe.reset_index(drop=True)
        starting_asset_value = dataframe['Close'].loc[0]
        ending_asset_value = dataframe['Close'].loc[len(dataframe) - 1]
        baseline_p_and_l =  ((ending_asset_value - starting_asset_value) / starting_asset_value) * 100
        starting_portfolio_value = dataframe['total_portfolio_value'].loc[0]
        ending_portfolio_value = dataframe['total_portfolio_value'].loc[len(dataframe) - 1]
        strategy_p_and_l = ((ending_portfolio_value - starting_portfolio_value) / starting_portfolio_value) * 100
        alpha = strategy_p_and_l - baseline_p_and_l
        st.write('Starting Value Asset: ' + str(starting_asset_value))
        st.write('Ending Value Asset: ' + str(ending_asset_value))
        st.write('Baseline P&L: ' + str(baseline_p_and_l))
        st.write('Starting Portfolio Value: ' + str(starting_portfolio_value))
        st.write('Ending Portfolio Value: ' + str(ending_portfolio_value))
        st.write('STRATEGY P&L: ' + str(strategy_p_and_l))
        st.write('ALPHA: ' + str(alpha))


        fig, ax = plt.subplots(figsize=(14,8))
        ax.plot(dataframe['total_portfolio_value'] ,linewidth=0.7, color='blue', alpha = 0.9)
        ax.set_title(names[i],fontsize=10, backgroundcolor='blue', color='white')
        ax.set_ylabel('Porfolio Value' , fontsize=18)
        ax.grid()
        plt.tight_layout()
        st.pyplot(fig)

        dataframe = dataframe.reset_index(drop=True)
        def create_buy_column(row):
            if row['buy_decision'] == 'buy': 
                return row['Close']
            else: 
                pass
        def create_sell_column(row): 
            if row['buy_decision'] == 'sell': 
                return row['Close']
            else: 
                pass 


        dataframe['sell_close'] = dataframe.apply(create_sell_column, axis=1)
        dataframe['buy_close'] = dataframe.apply(create_buy_column, axis=1)
        fig, ax = plt.subplots(figsize=(14,8))
        ax.plot(dataframe['Close'] , label = 'Close' ,linewidth=0.5, color='blue', alpha = 0.9)
        ax.scatter(dataframe.index , dataframe['buy_close'] , label = 'Buy' , marker = '^', color = 'green',alpha =1 )
        ax.scatter(dataframe.index , dataframe['sell_close'] , label = 'Sell' , marker = 'v', color = 'red',alpha =1 )
        ax.set_title(names[i] + " Price History with Buy and Sell Signals",fontsize=10, backgroundcolor='blue', color='white')
        ax.set_ylabel('Close Prices' , fontsize=18)
        legend = ax.legend()
        ax.grid()
        plt.tight_layout()
        st.pyplot(fig)
        dataframe = dataframe.drop(columns=['sell_close', 'buy_close'])