# -*- coding: utf-8 -*-

import pandas as pd


def features_separated_by_class(df):
    """
    This function devides a DataFrame containing multiple features by their
    class indices.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing multiple columns of features and one column for
        the class indices for every row of features.

    Returns
    -------
    sep_dfs : dict
        Dictionary with class indices as keys and the corresponding rows in a
        new DataFrame per index as value.

    """
    sep_dfs = dict()
    
    for i in range(len(df.index)):
        row = df.iloc[i,:]
        c = row.iloc[-1]
        row = row.iloc[:-1]
        if c not in sep_dfs:
            sep_dfs[c] = pd.DataFrame()
        sep_dfs[c] = sep_dfs[c].append(row, ignore_index=True)
    
    return sep_dfs


def get_mean_std_num(df):
    """
    This function calculates the mean and standard deviation for every column
    in a DataFrame and its number of rows.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing multiple columns of features and one column for
        the class indices for every row of features.

    Returns
    -------
    tuple
        Tuple containing the mean and standard deviation for every column in df
        and the number of rows.

    """
    return (df.mean(numeric_only=True), df.std(numeric_only=True), len(df.index))


def get_mean_std_num_per_class(df):
    """
    Seperates a DataFrame by class indices and calculates the mean, standard
    deviation and number of rows per class.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing multiple columns of features and one column for 
        the class indices for every row of features.

    Returns
    -------
    msn : dict
        Dictionary with class indices as keys and tuples containing the mean
        and standard deviation for every columnt in the class separated df and
        the number of rows as values.

    """
    sep_dfs = features_separated_by_class(df)
    msn = dict()
    for c, row in sep_dfs.items():
        msn[c] = get_mean_std_num(row)
        
    return msn


        