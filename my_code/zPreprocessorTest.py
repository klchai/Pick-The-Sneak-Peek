#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 08:04:23 2017

@author: isabelleguyon

This is an example of program that tests the Iris challenge Preprocessor class.
Another style is to incorporate the test as a main function in the Data manager class itself.
"""
from zDataManager import DataManager
from zPreprocessor import Preprocessor
input_dir = "../public_data"
output_dir = "../res"

basename = 'movies'
D = DataManager(basename, input_dir) # Load data
print("*** Original data ***")
print D

Prepro = Preprocessor()
 
# Preprocess on the data and load it back into D
D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
D.data['X_test'] = Prepro.transform(D.data['X_test'])
  
# Here show something that proves that the preprocessing worked fine
print("*** Transformed data ***")
print D

# Preprocessing gives you opportunities of visualization:
# Scatter-plots of the 2 first principal components
# Scatter plots of pairs of features that are most relevant
import matplotlib.pyplot as plt
X = D.data['X_train']
Y = D.data['Y_train']
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
#plt.colorbar()
plt.show()
