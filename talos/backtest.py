import pandas as pd
import numpy as np 

def generate_buy_decision(dataframe: pd.DataFrame, trade_signal_col_name: str, verbose=True): 
    dataframe = dataframe.reset_index(drop=True)
    dataframe['buy_decision'] = ''
    for row in range(0, len(dataframe)): 
        current_condition = dataframe[trade_signal_col_name].iloc[row]
        prior_condition = dataframe[trade_signal_col_name].iloc[row-1]
        if (current_condition != prior_condition) & (prior_condition == 1):
            final_decision = 'sell'
        elif (current_condition != prior_condition) & (prior_condition == 0):
            final_decision = 'buy'
        else: 
            final_decision = 'hold'
        dataframe['buy_decision'].iloc[row] = final_decision
    return dataframe 


'''
FIX THIS 
'''
def generate_buy_decision_stop_loss(dataframe: pd.DataFrame, close_column: str, trade_signal_col_name: str, stop_loss_percent: float, verbose=True): 
    dataframe = dataframe.reset_index(drop=True)
    dataframe['buy_decision'] = ''
    dataframe['moving_stop_loss'] = 0
    for row in range(0, len(dataframe)):
        current_price = float(dataframe[close_column].iloc[row])
        moving_stop_loss = float(dataframe['moving_stop_loss'].iloc[row])
        current_condition = dataframe[trade_signal_col_name].iloc[row]
        prior_condition = dataframe[trade_signal_col_name].iloc[row-1]
        if ((current_condition != prior_condition) & (prior_condition == 1)) or ((prior_condition == 1) & (current_price < moving_stop_loss)):
            final_decision = 'sell'
        elif (current_condition != prior_condition) & (prior_condition == 0):
            final_decision = 'buy'
            dataframe['moving_stop_loss'] = dataframe[close_column].iloc[row] - (dataframe[close_column].iloc[row] * stop_loss_percent)
        else: 
            final_decision = 'hold'
        dataframe['buy_decision'].iloc[row] = final_decision
    return dataframe 


def p_and_l_fractional_shares(dataframe: pd.DataFrame,  close_col: str, buy_decision_col: str, initial_cash: int, commission_percent: float, verbose=True):
    dataframe = dataframe.reset_index(drop=True)
    dataframe['shares_owned'] = ''
    dataframe['total_value_of_shares'] = ''
    dataframe['remaining_cash'] = ''
    dataframe['total_portfolio_value'] = ''
    for row in range(0, len(dataframe)): 
        todays_buy_decision = dataframe[buy_decision_col].loc[row]
        todays_close_price = dataframe[close_col].loc[row]
        if row == 0: 
            initial_close_price = dataframe[close_col].loc[0]
            initial_shares = (initial_cash - (initial_cash * commission_percent)) / initial_close_price
            initial_value_of_shares = initial_shares * initial_close_price
            initial_remaining_cash = initial_cash - initial_value_of_shares
            initial_total_portfolio_value = initial_value_of_shares + initial_remaining_cash
            if todays_buy_decision == 'buy': 
                dataframe['shares_owned'].loc[row] = initial_shares
                dataframe['total_value_of_shares'].loc[row] = initial_value_of_shares
                dataframe['remaining_cash'].loc[row] = initial_remaining_cash
                dataframe['total_portfolio_value'].loc[row] = initial_total_portfolio_value   
            if todays_buy_decision == 'hold':
                dataframe['shares_owned'].loc[row] = initial_shares
                dataframe['remaining_cash'].loc[row] = initial_remaining_cash 
                dataframe['total_value_of_shares'].loc[row] = initial_value_of_shares
                dataframe['total_portfolio_value'].loc[row] = initial_total_portfolio_value
            if todays_buy_decision == 'sell': 
                dataframe['shares_owned'].loc[row] = 0
                dataframe['total_value_of_shares'].loc[row] = 0 
                dataframe['remaining_cash'].loc[row] = initial_remaining_cash + ((initial_shares * todays_close_price) *  (1-commission_percent))
                dataframe['total_portfolio_value'].loc[row] = initial_remaining_cash + ((initial_shares * todays_close_price) *  (1-commission_percent))
        else: 
            if todays_buy_decision == 'buy': 
                shares_to_buy =(dataframe['remaining_cash'].loc[row-1] * (1-commission_percent))/ todays_close_price
                shares_owned = dataframe['shares_owned'].loc[row-1] + shares_to_buy
                total_value_of_shares = shares_owned * todays_close_price
                remaining_cash = dataframe['remaining_cash'].loc[row-1] - (shares_to_buy * todays_close_price)
                dataframe['shares_owned'].loc[row] = shares_owned
                dataframe['total_value_of_shares'].loc[row] = total_value_of_shares
                dataframe['remaining_cash'].loc[row] = remaining_cash
                dataframe['total_portfolio_value'].loc[row] = total_value_of_shares + remaining_cash
            if todays_buy_decision == 'hold': 
                yesterday_shares_owned = dataframe['shares_owned'].loc[row-1]
                total_value_of_shares = yesterday_shares_owned * todays_close_price
                yesterday_remaining_cash = dataframe['remaining_cash'].loc[row-1]
                dataframe['shares_owned'].loc[row] = yesterday_shares_owned
                dataframe['remaining_cash'].loc[row] = yesterday_remaining_cash
                dataframe['total_value_of_shares'].loc[row] = total_value_of_shares
                dataframe['total_portfolio_value'].loc[row] = total_value_of_shares + yesterday_remaining_cash
            if todays_buy_decision == 'sell': 
                yesterday_shares_owned = dataframe['shares_owned'].loc[row-1]
                shares_to_sell = yesterday_shares_owned
                yesterday_remaining_cash = dataframe['remaining_cash'].loc[row-1]
                dataframe['shares_owned'].loc[row] = 0
                dataframe['total_value_of_shares'].loc[row] = 0 
                dataframe['remaining_cash'].loc[row] = yesterday_remaining_cash + ((shares_to_sell * todays_close_price) * (1-commission_percent))
                dataframe['total_portfolio_value'].loc[row] = yesterday_remaining_cash + ((shares_to_sell * todays_close_price) * (1-commission_percent))   
    return dataframe

def p_and_l_fractional_shares_stop_loss(dataframe: pd.DataFrame,  close_col: str, stop_loss_points: float, buy_decision_col: str, initial_cash: int, commission_percent: float, verbose=True):
    dataframe = dataframe.reset_index(drop=True)
    dataframe['shares_owned'] = ''
    dataframe['total_value_of_shares'] = ''
    dataframe['remaining_cash'] = ''
    dataframe['total_portfolio_value'] = ''
    dataframe['moving_stop_loss'] = 0
    for row in range(0, len(dataframe)): 
        todays_buy_decision = dataframe[buy_decision_col].loc[row]
        todays_close_price = dataframe[close_col].loc[row]
        if row == 0: 
            initial_close_price = dataframe[close_col].loc[0]
            initial_shares = (initial_cash - (initial_cash * commission_percent)) / initial_close_price
            initial_value_of_shares = initial_shares * initial_close_price
            initial_remaining_cash = initial_cash - initial_value_of_shares
            initial_total_portfolio_value = initial_value_of_shares + initial_remaining_cash
            if todays_buy_decision == 'buy': 
                dataframe['shares_owned'].loc[row] = initial_shares
                dataframe['total_value_of_shares'].loc[row] = initial_value_of_shares
                dataframe['remaining_cash'].loc[row] = initial_remaining_cash
                dataframe['total_portfolio_value'].loc[row] = initial_total_portfolio_value  
                dataframe['moving_stop_loss'] = initial_close_price - stop_loss_points
            if todays_buy_decision == 'hold':
                dataframe['shares_owned'].loc[row] = initial_shares
                dataframe['remaining_cash'].loc[row] = initial_remaining_cash 
                dataframe['total_value_of_shares'].loc[row] = initial_value_of_shares
                dataframe['total_portfolio_value'].loc[row] = initial_total_portfolio_value
            if todays_buy_decision == 'sell': 
                dataframe['shares_owned'].loc[row] = 0
                dataframe['total_value_of_shares'].loc[row] = 0 
                dataframe['remaining_cash'].loc[row] = initial_remaining_cash + ((initial_shares * todays_close_price) *  (1-commission_percent))
                dataframe['total_portfolio_value'].loc[row] = initial_remaining_cash + ((initial_shares * todays_close_price) *  (1-commission_percent))
        else: 
            if todays_buy_decision == 'buy': 
                shares_to_buy =(dataframe['remaining_cash'].loc[row-1] * (1-commission_percent))/ todays_close_price
                shares_owned = dataframe['shares_owned'].loc[row-1] + shares_to_buy
                total_value_of_shares = shares_owned * todays_close_price
                remaining_cash = dataframe['remaining_cash'].loc[row-1] - (shares_to_buy * todays_close_price)
                dataframe['shares_owned'].loc[row] = shares_owned
                dataframe['total_value_of_shares'].loc[row] = total_value_of_shares
                dataframe['remaining_cash'].loc[row] = remaining_cash
                dataframe['total_portfolio_value'].loc[row] = total_value_of_shares + remaining_cash
                dataframe['moving_stop_loss'] = todays_close_price - stop_loss_points
            if todays_buy_decision == 'hold': 
                yesterday_shares_owned = dataframe['shares_owned'].loc[row-1]
                total_value_of_shares = yesterday_shares_owned * todays_close_price
                yesterday_remaining_cash = dataframe['remaining_cash'].loc[row-1]
                dataframe['shares_owned'].loc[row] = yesterday_shares_owned
                dataframe['remaining_cash'].loc[row] = yesterday_remaining_cash
                dataframe['total_value_of_shares'].loc[row] = total_value_of_shares
                dataframe['total_portfolio_value'].loc[row] = total_value_of_shares + yesterday_remaining_cash
            if todays_buy_decision == 'sell': 
                yesterday_shares_owned = dataframe['shares_owned'].loc[row-1]
                shares_to_sell = yesterday_shares_owned
                yesterday_remaining_cash = dataframe['remaining_cash'].loc[row-1]
                dataframe['shares_owned'].loc[row] = 0
                dataframe['total_value_of_shares'].loc[row] = 0 
                dataframe['remaining_cash'].loc[row] = yesterday_remaining_cash + ((shares_to_sell * todays_close_price) * (1-commission_percent))
                dataframe['total_portfolio_value'].loc[row] = yesterday_remaining_cash + ((shares_to_sell * todays_close_price) * (1-commission_percent))  
            if row < len(dataframe)-1:  
                if float(dataframe['Close'].loc[row+1]) < float(dataframe['moving_stop_loss'].loc[row+1]): 
                    dataframe['buy_decision'].loc[row+1] = 'sell'
    return dataframe
