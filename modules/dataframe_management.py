# -*- coding: utf-8 -*-

import pandas as pd
import os


def features_separated_by_class(df):
    separated_dfs = dict()
    
    for i in range(len(df.index)):
        row = df.iloc[i,:]
        c = row.iloc[3]
        row = row.iloc[4:]
        if c not in separated_dfs:
            separated_dfs[c] = pd.DataFrame()
        separated_dfs[c] = separated_dfs[c].append(row, ignore_index=True)
    
    # for _, d in separated_dfs.items():
    #     print(get_mean_std_num(d)[2])
    #     print("------------------------------------------")

    return separated_dfs

def get_mean_std_num(df):
    return (df.mean(numeric_only=True), df.std(numeric_only=True), len(df.index))

def get_mean_std_num_per_class(df):
    separated_dfs = features_separated_by_class(df)
    mean_std_nums = dict()
    for c, row in separated_dfs.items():
        mean_std_nums[c] = get_mean_std_num(row)
        
    return mean_std_nums
        
        
# df = pd.read_csv(os.path.join(".", "data", "features.csv"), sep=';')#, usecols=[3,4,5,6,7])
# # print(df)
# msn = get_mean_std_num_per_class(df)
# # print(msn)

        