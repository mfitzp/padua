__author__ = 'mfitzp'

import pandas as pd
import numpy as np

def remove_columns_matching(df, column, match):
    df = df.copy()
    mask = df[column].values != match
    return df.iloc[mask, :]


def remove_columns_containing(df, column, match):
    df = df.copy()
    mask = [match not in str(v) for v in df[column].values]
    return df.iloc[mask, :]


def remove_reverse(df):
    """
    Remove rows with a + in the 'Reverse' column

    :param df: Pandas dataframe
    :return: Pandas dataframe
    """
    return remove_columns_containing(df, 'Reverse', '+')


def remove_potential_contaminants(df):
    """
    Remove rows with a + in the 'Contaminants' column

    :param df: Pandas dataframe
    :return: Pandas dataframe
    """
    return remove_columns_containing(df, 'Potential contaminant', '+')


def remove_only_identified_by_site(df):
    """
    Remove rows with a + in the 'Only identified by site' column

    :param df: Pandas dataframe
    :return: Pandas dataframe
    """
    return remove_columns_containing(df, 'Only identified by site', '+')


def filter_localization_probability(df, threshold=0.75):
    """
    Remove rows with a localization probability below 0.75

    :param df: Pandas dataframe
    :param threshold: Cut-off below which rows are discarded (default 0.75)
    :return: Pandas dataframe
    """
    df = df.copy()
    localization_probability_mask = df['Localization prob'].values >= threshold
    return df.iloc[localization_probability_mask, :]


def minimum_valid_values_in_group(df, levels, n=1, invalid=np.nan):
    """
    Filter dataframe by at least n valid values in at least one group.
    
    """

    # Filter by at least 7 (values in class:timepoint) at least in at least one group
    if invalid is np.nan:
        dfx = ~np.isnan(df)
    else:
        dfx = df != invalid
    
    dfc = dfx.astype(int).sum(axis=1, level=levels)
    
    dfm = dfc.max(axis=1) > n
    
    mask = dfm.values
    
    return df.iloc[mask, :]
    
    
    
    
    

