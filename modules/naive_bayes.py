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
    num_testdata = sum([mean_std_nums[c][0][2] for c in mean_std_nums])

    probabilities = dict()
    for c, mean_std_num in mean_std_nums.items():
        # calculate the a priori probability
        prior = mean_std_nums[c][0][2]/float(num_testdata)
        probabilities[c] = prior
        mean, std, _= mean_std_num
        likelihood = gaussian_probability(row, mean, std)
        for p in likelihood:
            probabilities[c] *= p
        # evidence += probabilities[c]    
    return probabilities


classes = {0 : "bottle opener",
           1 : "can opener",
           2 : "corc screw",
           3 : "multi tool"}


parent_dir = os.getcwd()
wd = os.getcwd().split(os.sep)
data_dir = wd[0] + os.sep
for folder in wd[1:]:
    data_dir = os.path.join(data_dir, folder)
    if folder == "ai-assignment":
        break
data = os.path.join(data_dir, "data", "features.csv")
# os.path.join(".", "data", "features.csv")

df = pd.read_csv(data, sep=';')#, usecols=[3,4,5,6,7])
# df = df.iloc[:-10, :]
df_test = dm.features_separated_by_class(df)#df.iloc[570:580, 4:]
# df_test = df_test[3]
# print(df)
# print(df.iloc[:-1, :])
# print(df.iloc[-1, :])
# print(df)
msn = dm.get_mean_std_num_per_class(df)
# print(df_test.apply(class_probabilities, axis=1, args=(msn)))
# correct_ratio = 0
# num = len(df_test.index)

for i in range(15):
    rdm_class = randrange(0, 4)
    df_class = df_test[rdm_class]
    rdm_img_idx = randrange(0, len(df_class.index))
    probs = class_probabilities(df_class.iloc[rdm_img_idx, :], msn)
    prediction = classes[max(probs, key=probs.get)]
    print("Predicted:", prediction, "\t\tTarget:", classes[rdm_class], "\t\tCorrect?:", prediction == classes[rdm_class])


# for _, test in df_test.iterrows():
#     probs = class_probabilities(test, msn)#df.iloc[-1, 4:])
#     # print(probs)#pd.Series([0.5,2,3,4], index=df.columns[4:])))
#     # print(max(probs.values()))#.values()))
#     # print(max(probs, key=probs.get))
#     max_idx = max(probs, key=probs.get)
#     # if max_idx == 3:
#     #     correct_ratio += 1
#     print(classes[max_idx])
    
# print(correct_ratio/num)
# print(msn)