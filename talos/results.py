import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def calculate_performance(dataframe: pd.DataFrame, close_col: str, total_portfolio_value_col: str, verbose=True): 
    if verbose: print('Trading decisions made.')
    dataframe = dataframe.reset_index(drop=True)
    starting_asset_value = dataframe[close_col].loc[0]
    ending_asset_value = dataframe[close_col].loc[len(dataframe) - 1]
    baseline_p_and_l =  ((ending_asset_value - starting_asset_value) / starting_asset_value) * 100
    starting_portfolio_value = dataframe[total_portfolio_value_col].loc[0]
    ending_portfolio_value = dataframe[total_portfolio_value_col].loc[len(dataframe) - 1]
    strategy_p_and_l = ((ending_portfolio_value - starting_portfolio_value) / starting_portfolio_value) * 100
    alpha = strategy_p_and_l - baseline_p_and_l
    if verbose: print('\nStarting Value Asset: {}'.format(starting_asset_value))
    if verbose: print('Ending Value Asset: {}'.format(ending_asset_value))
    if verbose: print('Baseline P&L: {}%'.format(baseline_p_and_l))
    if verbose: print('\nStarting Portfolio Value: {}'.format(starting_portfolio_value))
    if verbose: print('Ending Portfolio Value: {}'.format(ending_portfolio_value))
    if verbose: print('STRATEGY P&L: {}%'.format(strategy_p_and_l))
    if verbose: print('ALPHA: {}%'.format(alpha))
    return None


def view_portfolio_df(stock_df: pd.DataFrame, total_portfolio_value_col: str, ticker: str):
    fig, ax = plt.subplots(figsize=(14,8))
    ax.plot(stock_df[total_portfolio_value_col] ,linewidth=0.7, color='blue', alpha = 0.9)
    ax.set_title(ticker,fontsize=10, backgroundcolor='blue', color='white')
    ax.set_ylabel('Porfolio Value' , fontsize=18)
    ax.grid()
    plt.tight_layout()
    plt.show()
    return None


def view_stock_with_decision(stock_df: pd.DataFrame, ticker: str, close_col: str, buy_decision_col: str):
    stock_df = stock_df.reset_index(drop=True)
    def create_buy_column(row):
        if row[buy_decision_col] == 'buy': 
            return row[close_col]
        else: 
            None
    def create_sell_column(row): 
        if row[buy_decision_col] == 'sell': 
            return row[close_col]
        else: 
            return None 
    stock_df['sell_close'] = stock_df.apply(create_sell_column, axis=1)
    stock_df['buy_close'] = stock_df.apply(create_buy_column, axis=1)
    fig, ax = plt.subplots(figsize=(14,8))
    ax.plot(stock_df[close_col] , label = 'Close' ,linewidth=0.5, color='blue', alpha = 0.9)
    ax.scatter(stock_df.index , stock_df['buy_close'] , label = 'Buy' , marker = '^', color = 'green',alpha =1 )
    ax.scatter(stock_df.index , stock_df['sell_close'] , label = 'Sell' , marker = 'v', color = 'red',alpha =1 )
    ax.set_title(ticker + " Price History with Buy and Sell Signals",fontsize=10, backgroundcolor='blue', color='white')
    ax.set_ylabel('Close Prices' , fontsize=18)
    legend = ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()
    stock_df = stock_df.drop(columns=['sell_close', 'buy_close'])
    return None