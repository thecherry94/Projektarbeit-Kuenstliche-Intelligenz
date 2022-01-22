import DecisionTree as dt
from collections import Counter

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

# Returns a forest of decision trees
def build_forest(train, max_depths, min_sizes):
    assert(len(max_depths) == len(min_sizes))
    return [dt.build_tree(train, max_depths[i], min_sizes[i]) for i in range(len(max_depths))]

def predict(forest, row):
    preds = []
    for tree in forest:
        preds.append(dt.predict(tree, row))
    return most_common(preds)