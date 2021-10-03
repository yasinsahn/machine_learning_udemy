#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.17
author: yasin sahin
written to construct upper confidence bound reinforcement learning model

"""

# importing necessary libraries
import pandas as pd
import math
import matplotlib.pyplot as plt


# importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# implementing upper confidence bound algorithm

N = dataset.shape[0] # number of rounds
d = dataset.shape[1] # number of ads
ads_selected = [] # creating empty list to write selected ad in each round
# creating a list with length of candidate numbers to write total number of selections of each ad
numbers_of_selections = [0] * d 
# creating a list with length of candidate numbers to write total reward of each ad
sums_of_rewards = [0] * d
total_reward = 0 # variable to write given total reward

for n in range(0,N): # iterating for each round
    ad = 0 # variable to write selected ad number
    max_upper_bound = 0 # maximum of upper bound
    for i in range(0,d): # iterating for each ad
        if (numbers_of_selections[i] > 0): # checking if ad selection is zero to avoid infinity
            average_reward = sums_of_rewards[i] / numbers_of_selections[i] # calculating average reward
            delta_i = math.sqrt((3/2) * math.log(n + 1) / numbers_of_selections[i]) # calculating delta
            upper_bound = average_reward + delta_i # calculating upper bound
        else:
            upper_bound = 1e400 # assigning upper bound to high value to avoid infinity
        if (upper_bound > max_upper_bound): # checking if upper bound is bigger than current maximum upper bound
            max_upper_bound = upper_bound # updating maximum upper bound
            ad = i # updating selected ad number
    ads_selected.append(ad) # appending selected ad for current round
    numbers_of_selections[ad] += 1 # incrementing ad selection count for selected ad
    reward = dataset.values[n, ad] # determining the reward for current selected ad
    sums_of_rewards[ad] += reward # incrementing total reward wrt current selected ad
    total_reward += reward # incrementing total given reward

    
# visualizing results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of Times Each Ad was Selected')
plt.show()