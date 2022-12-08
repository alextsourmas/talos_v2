import pandas as pd
import numpy as np 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def rolling_xgb(predictors: list, target_col: pd.Series, pred_col_name: str,  train_df_dict: dict, test_df_dict: dict, cutoff_threshold: float,
    random_state=1, feature_importance=True, verbose=True): 
    for round in range(0, len(train_df_dict)): 
        if verbose: print('Running XGB round {} of {}'.format(round+1, len(train_df_dict)))
        train_df = train_df_dict[round]
        test_df = test_df_dict[round]
        xgb = XGBClassifier(random_state=random_state)
        xgb.fit(train_df[predictors], train_df[target_col])
        xgb_preds = xgb.predict_proba(test_df[predictors])[:,0]
        xgb_preds[xgb_preds>=cutoff_threshold] = 1
        xgb_preds[xgb_preds<cutoff_threshold] = 0
        xgb_preds = pd.Series(xgb_preds, index=test_df.index, name=pred_col_name)
        combined = pd.concat([test_df, xgb_preds], axis=1)
        if round == 0: 
            final_df = combined
        else: 
            final_df = pd.concat([final_df, combined], ignore_index=True)
    final_df = final_df.reset_index(drop=True)
    if feature_importance: 
        feature_important = xgb.get_booster().get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())
        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
        data.nlargest(40, columns="score").plot(kind='barh', figsize = (20,10)) ## plot top 40 features
    benchmark = final_df[target_col].value_counts()[1] /(final_df[target_col].value_counts()[0] + final_df[target_col].value_counts()[1] )
    precision = precision_score(final_df[target_col], final_df[pred_col_name].astype(int))
    recall = recall_score(final_df[target_col], final_df[pred_col_name].astype(int))
    accuracy = accuracy_score(final_df[target_col], final_df[pred_col_name].astype(int))
    if verbose: print('Benchmark precision: {}'.format(benchmark))
    if verbose: print('XGB Precision: {}'.format(precision))
    if verbose: print('XGB Recall: {}'.format(recall))
    if verbose: print('XGB Accuracy: {}'.format(accuracy))
    return final_df 