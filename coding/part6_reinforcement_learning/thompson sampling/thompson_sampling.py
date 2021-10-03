#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.17
author: yasin sahin
written to construct thompson sampling reinforcement learning model

"""

# importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import random

# importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# implementing upper confidence bound algorithm

N = dataset.shape[0] # number of rounds
d = dataset.shape[1] # number of ads
ads_selected = [] # list created to write selected ads in each round
numbers_of_rewards_1 = [0] * d # list to increment when selected ad has reward 1
numbers_of_rewards_0 = [0] * d # list to increment when selected ad has reward 0
total_reward = 0 # cumulative reward

# implementing thampson sampling algorithm
for n in range(0, N):# iterating for each round
    ad = 0 # variable to write selected ad number
    max_random = 0 # maximum of beta distribution random number
    for i in range(0, d): # iterating for each ad
        random_beta = \
            random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1) # beta distribution random number
        if (random_beta > max_random): # checking if random number is greater than maximum random number
            max_random = random_beta # updating maximum random number
            ad = i # updating selected ap
    ads_selected.append(ad) # saving selected ad number
    reward = dataset.values[n, ad] # taking round reward for current ad selection
    if reward == 1:
        numbers_of_rewards_1[ad] += 1 # incerementing selected ad score if reward is 1
    else:
        numbers_of_rewards_0[ad] += 1 # incerementing selected ad score if reward is 0
    total_reward += reward # calculating total reward given

# visualizing results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of Times Each Ad was Selected')
plt.show()