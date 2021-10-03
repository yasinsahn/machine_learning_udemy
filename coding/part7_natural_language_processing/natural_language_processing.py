#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.17
author: yasin sahin
written to construct a bag-of-words natural language processing model

"""
# import nltk
# nltk.download('stopwords') NEEDED TO DOWNLOAD STOPTWORDS AT THE FIRST IMPLEMENTATION

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# importing dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# cleaning texts
corpus = []
for i in range(0, dataset.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # substituting non-letters with space
    review = review.lower() # making all letters lowercase letter
    review = review.split() # splitting text into different words
    ps = PorterStemmer() # initiliazing porter stemmer object
    
    # avoiding stopwords in english and applying stemming
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not') # removing not from stopwords
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review) # merging words into text again
    corpus.append(review) # appending each review into corpus

# creating bag-of-words model
cv = CountVectorizer() # creating CountVectorizer object instant
x = cv.fit_transform(corpus).toarray() # creating sparse matrix to construct independent variables
y = dataset.iloc[:,-1].values

# splitting train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# fitting Naive Bayes Classifier to model
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# predicting classifier Results for test set
y_pred = classifier.predict(x_test)

# constructing confusion matrix
c_m = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred) # calculating accuracy score
print(f' Accuracy is : {accuracy}')
