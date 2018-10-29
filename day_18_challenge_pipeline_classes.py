from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt

class MergeColumns(TransformerMixin):
    def __init__(self, column_one, column_two):
        self.column_one = column_one
        self.column_two = column_two
    
    def fit(self, x, y= None):
        return self
    
    def transform(self, x):
        x[self.column_one] = x[self.column_one].fillna(0) + x[self.column_two].fillna(0)
        x = x.drop(self.column_two, axis=1)
        return x

class ImputeValue(TransformerMixin):
    def __init__(self, column, value):
        self.column = column
        self.value = value
    
    def fit(self, x, y= None):
        return self
    
    def transform(self, x):
        x[self.column].fillna(self.value, inplace=True)
        
        return x

class ConvertZeroToN(TransformerMixin):
    def __init__(self, column):
        self.column = column
    
    def fit(self, x, y= None):
        return self
    
    def transform(self, x):
        x[self.column][x[self.column] == '0'] = 'N'
        
        return x

class ConvertStringDateToYear(TransformerMixin):
    def __init__(self, column):
        self.column = column
    
    def fit(self, x, y= None):
        return self
    
    def transform(self, x):
        x[self.column] = pd.to_datetime(x[self.column], format='%Y-%m-%d', errors='coerce')
        x['YEAR'] = x[self.column].dt.year
        
        return x

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, x, y = None):
        return self
    
    def transform(self, x):
        return x.loc[:, self.columns]


def compare_predictions(x, y, finalpipeline, mean_price):
    # generate predictions:
    # Note: finalpipeline must have already been fit.
    
    predictions = finalpipeline.predict(x)
    y = y.reset_index()
    y.drop('index', axis=1, inplace=True)
    
    # a "lazy prediction" is where we return the average value of the target for every prediction.
    lazy_predictions = np.full(predictions.shape, mean_price)
    
    final_predictions = pd.DataFrame(pd.concat([y, 
                                                pd.Series(predictions), 
                                                pd.Series(lazy_predictions)], axis=1))
    final_predictions.rename(columns={'PRICE': 'True values',
                                      0: 'Predicted values',
                                      1: 'Lazy Predicted values'}, inplace=True)
    
    rmse_lazy = sqrt(mean_squared_error(y, lazy_predictions))
    mae_lazy = mean_absolute_error(y, lazy_predictions)
    r2_lazy = r2_score(y, lazy_predictions)
    
    rmse = sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print('RMSE Lazy Predictor', rmse_lazy)
    print('MAE Lazy Predictor', mae_lazy)
    print('R^2 Lazy Predictor', r2_lazy)
    print()
    print('RMSE', rmse)
    print('MAE', mae)
    print('R^2', r2)
    print()
    print('RMSE Improvement:', rmse_lazy - rmse)
    print('MAE Inprovement:', mae_lazy - mae)
    print('R^2 Improvement:', abs(r2_lazy - r2))
    
    
    plt.figure(figsize=(20,10))

    plt.plot(final_predictions.index, final_predictions['True values'], c='red', label='True Values')
    plt.plot(final_predictions.index, final_predictions['Predicted values'], c='blue', label='Predicted Values')
    plt.plot(final_predictions.index, final_predictions['Lazy Predicted values'], c='black', label='Lazy Predicted Values')
    plt.legend(loc='best')
    plt.show()
    return predictions