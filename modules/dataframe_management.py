# -*- coding: utf-8 -*-

import pandas as pd
import os


def features_separated_by_class(df):
    separated_dfs = dict()
    
    for i in range(len(df.index)):
        row = df.iloc[i,:]
        c = row.iloc[-1]
        row = row.iloc[:-1]
        if c not in separated_dfs:
            separated_dfs[c] = pd.DataFrame()
        separated_dfs[c] = separated_dfs[c].append(row, ignore_index=True)
    
    return separated_dfs

def get_mean_std_num(df):
    return (df.mean(numeric_only=True), df.std(numeric_only=True), len(df.index))

def get_mean_std_num_per_class(df):
    separated_dfs = features_separated_by_class(df)
    mean_std_nums = dict()
    for c, row in separated_dfs.items():
        mean_std_nums[c] = get_mean_std_num(row)
        
    return mean_std_nums


        