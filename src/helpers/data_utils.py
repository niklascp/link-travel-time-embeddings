import datetime

import numpy as np
import pandas as pd

from . import api

def split_normal_days_train_val(train_normal_days):
    """
    This function splits the train data in train and validation datasets used for HyperParameter tuning and validation.
    The split is done such that the validation data set includes exactly one of each day types (e.g. weekdays for normal days).
    """
    train_normal_days_per_date = train_normal_days.groupby([train_normal_days.index.date, train_normal_days['day_type']]).count()
    train_normal_days_per_date.index.names = ['date', 'day_type']
    val_dates = train_normal_days_per_date.reset_index(level= 0).groupby(lambda x: x)['date'].max().values
    val_map = np.isin(train_normal_days.index.date, val_dates)
    return train_normal_days.loc[~val_map], train_normal_days.loc[val_map]

def split_special_days_train_val(train_special_days, p = .1, random_seed = 42):
    """
    This function splits the special days train data in train and validation datasets used for HyperParameter tuning and validation.

    """
    if random_seed:
        np.random.seed(random_seed)
    val_map = np.random.choice([True, False], p=[p, 1-p], size=len(train_special_days))
    return train_special_days.loc[~val_map], train_special_days.loc[val_map]

def normalize_special_days(link_ref, train_special_days):
    """
    Normalizes special days based on nearby normal days
    """
    # This is all just to reduce the number of needed request to the API service for adjacent special days:
    # 1. Find the start/end date for each special day type
    special_days_start_end_date = pd.DataFrame({
        'start_date': train_special_days.reset_index().groupby('day_type')['time'].min().dt.date,
        'end_date': train_special_days.reset_index().groupby('day_type')['time'].max().dt.date
    })
    # 2. All special day dates
    special_days_dates = np.unique(train_special_days.index.date)
    # 3. Extend the start_date and end_date until we clear any other special days
    special_days_start_end_date_corrected = pd.DataFrame(special_days_start_end_date.copy())
    for day_type, start_date in special_days_start_end_date['start_date'].iteritems():
        while np.isin(start_date, special_days_dates):
            start_date = start_date - datetime.timedelta(days=1)
        special_days_start_end_date_corrected.loc[day_type, 'start_date'] = start_date + datetime.timedelta(days=1)
    for day_type, end_date in special_days_start_end_date['end_date'].iteritems():
        while np.isin(end_date, special_days_dates):
            end_date = end_date + datetime.timedelta(days=1)
        special_days_start_end_date_corrected.loc[day_type, 'end_date'] = end_date
    # Finally query API to get mean and std
    for (start_date, end_date), g in special_days_start_end_date_corrected.reset_index().groupby(['start_date', 'end_date']):
        n = (end_date - start_date) // pd.to_timedelta('1D')
        special_days_norm = api.link_travel_time_n_preceding_normal_days(link_ref, start_date, n)
        special_days_norm['link_travel_time_exp'] = special_days_norm.rolling(window=20, center=True, min_periods=1)['link_travel_time'].mean().round(1)
        special_days_start_end_date_corrected.loc[g['day_type'], 'mean'] = special_days_norm['link_travel_time_exp'].mean()
        special_days_start_end_date_corrected.loc[g['day_type'], 'std'] = special_days_norm['link_travel_time_exp'].std()
        
    train_special_days_ = train_special_days.copy()
    train_special_days_['mean'] = special_days_start_end_date_corrected.loc[train_special_days_['day_type'], 'mean'].values
    train_special_days_['std'] = special_days_start_end_date_corrected.loc[train_special_days_['day_type'], 'std'].values
    return train_special_days_