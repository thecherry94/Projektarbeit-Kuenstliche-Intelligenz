# -*- coding: utf-8 -*-

import math
import pandas as pd
import modules.dataframe_management as dm
import os
from random import randrange

def gaussian_probability(x, mean, std):
    probabilities = list()
    for i, x_i in x.items():
        p_i =  (1 / (math.sqrt(2 * math.pi) * std[i])) * math.exp(-((x_i-mean[i])**2 / (2 * std[i]**2 )))
        probabilities.append(p_i)
    return probabilities


def class_probabilities(row, mean_std_nums):
    # get the number of all training data elements for all classes
    num_testdata = sum([mean_std_nums[c][2] for c in mean_std_nums])

    probabilities = dict()
    for c, mean_std_num in mean_std_nums.items():
        # calculate the a priori probability
        prior = mean_std_nums[c][2]/float(num_testdata)
        probabilities[c] = prior
        mean, std, _= mean_std_num
        likelihood = gaussian_probability(row, mean, std)
        for p in likelihood:
            probabilities[c] *= p    
    return probabilities

