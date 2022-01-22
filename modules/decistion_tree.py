# -*- coding: utf-8 -*-

import pandas as pd
import os


classes = {0 : "bottle opener",
           1 : "can opener",
           2 : "corc screw",
           3 : "multi tool"}

df = pd.read_csv(os.path.join(".", "data", "features.csv"), sep=';')#, usecols=[3,4,5,6,7])
# df = df.iloc[:, :]
df_test = df.iloc[570:580, 4:]

# for _, test in df_test.iterrows():
#     probs = class_probabilities(test, msn)
#     print(classes[max(probs, key=probs.get)])

