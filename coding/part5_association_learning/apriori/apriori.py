#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.15
author: yasin sahin
written to construct Apriori model

"""

import pandas as pd
from apyori import apriori

# importing dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# creating list from dataset
transactions = []
# writing transactions from dataset to list
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

# training apriori function
rules = apriori(transactions = transactions, min_support = 0.003, \
                min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
    
results = list(rules)

# putting results well organised into a Pandas Dataframe
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

# changing list to Pandas Dataframe
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', \
                                                               'Right Hand Side', \
                                                               'Support', 'Confidence', \
                                                               'Lift'])

# sorting results according to lfit    
results_wrt_lift = resultsinDataFrame.nlargest(n = 10, columns = 'Lift')