# -*- coding: utf-8 -*-
"""
This file provides functions for naive bayes classification.
"""
from math import sqrt
from math import exp
from math import pi

def gaussian_probability(x, mean, std):
    """
    Probability density function of a Gaussian Distribution.

    Parameters
    ----------
    x : pandas.core.series.Series
        Series containing one value for every feature.
    mean : pandas.core.series.Series
        Series containing the mean value for every feature.
    std : pandas.core.series.Series
        Series containing the standard deviation for every feature.

    Returns
    -------
    probabilities : list
        The calculated gaussian probabilities for every feature.

    """
    
    probabilities = list()
    for i, x_i in x.items():
        p_i =  ((1 / (sqrt(2*pi) * std[i])) 
                * exp(-((x_i-mean[i])**2 / (2 * std[i]**2 ))))
        probabilities.append(p_i)
    
    return probabilities


def class_probabilities(row, mean_std_nums):
    """
    Calculates the probability of a test row for every possible class.

    Parameters
    ----------
    row : pandas.core.series.Series
        Series from test data containing one value for every feature.
    mean_std_nums : dict
        Dictionary containing the mean, standard deviation and number of rows
        in the training data.

    Returns
    -------
    probabilities : dict
        Dictionary containing the probability (value) for every possible class
        (key).

    """
    
    num_all = sum([mean_std_nums[c][2] for c in mean_std_nums])

    probabilities = dict()
    for c, mean_std_num in mean_std_nums.items():
        prior = mean_std_nums[c][2]/float(num_all)
        probabilities[c] = prior
        mean, std, _ = mean_std_num
        likelihood = gaussian_probability(row, mean, std)
        for p in likelihood:
            probabilities[c] *= p  
            
    return probabilities


