#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 08:04:23 2017

@author: salmon

This is an example of program that tests the PSP challenge Classifier class.
You can also find a test in the main function of the class zClassifier itself.

"""
import numpy as np
from zDataManager import DataManager
from zClassifier import Classifier, Classifier2
from sklearn.metrics import accuracy_score 
from sklearn.cross_validation import cross_val_score
from zPreprocessor import Preprocessor

input_dir = "../public_data"
output_dir = "../res"

basename = 'movies'
D = DataManager(basename, input_dir) # Load data
print D

Prepro = Preprocessor()

# Preprocess on the data and load it back into D
D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
D.data['X_test'] = Prepro.transform(D.data['X_test'])

def ebar(score, sample_num):
    '''ebar calculates the error bar for the classification score (accuracy or error rate)
    for sample_num examples'''
    return np.sqrt(1.*score*(1-score)/sample_num)

myclassifier = Classifier2()
 
# Train
alfa=.0
Ytrue_tr = D.data['Y_train']
X=D.data['X_train']
myclassifier.fit(X, Ytrue_tr, 20, X.shape[0]/2)

# Making predictions
Ypred_tr = myclassifier.predict(D.data['X_train'])
Ypred_va = myclassifier.predict(D.data['X_valid'])
Ypred_te = myclassifier.predict(D.data['X_test'])  

# We can compute the training success rate 
acc_tr = accuracy_score(Ytrue_tr, Ypred_tr)
# but it might be a bit optimistic compared to the validation and the test accuracy
# that we can't compute (except by making submissions to Codalab)
# So, we use cross-validation:
acc_cv = cross_val_score(myclassifier, X, Ytrue_tr, scoring='accuracy', cv=5, fit_params={'maxn':20, 'tr_samples':X.shape[0]/2}) # 'maxn':20,'tr_samples':X.shape[0]/2})

print "One sigma error bars:"
print "Training Accuracy = %5.2f +-%5.2f" % (acc_tr, ebar(acc_tr, Ytrue_tr.shape[0]))
print "Cross-validation Accuracy = %5.2f +-%5.2f" % (acc_cv.mean(), acc_cv.std())
