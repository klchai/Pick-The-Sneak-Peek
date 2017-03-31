#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 08:04:23 2017

@author: isabelleguyon

This is an example of program that reads data and has a few display methods.

Add more views of the data getting inspired by previous lessons:
    Histograms of single variables
    Data matrix heat map
    Correlation matric heat map

Add methods of exploratory data analysis and visualization:
    PCA or tSNE
    two-way hierachical clustering (combine with heat maps)

The same class could be used to visualize prediction results, by replacing X by
the predicted values (the end of the transformation chain):
    For regression, you can 
        plot Y as a function of X.
        plot the residual a function of X.
    For classification, you can 
        show the histograms of X for each Y value.
        show ROC curves.
    For both: provide a table of scores and error bars.
"""

# Add the sample code in the path
mypath = "../sample_code"
from sys import argv, path
from os.path import abspath
path.append(abspath(mypath))

# Graphic routines
import seaborn as sns; sns.set()

# Data types
import pandas as pd

# Mother class
import data_manager

class DataManager(data_manager.DataManager):
    '''This class reads and displays data. 
       With class inheritance, we do not need to redefine the constructor,
       unless we want to add or change some data members.
       '''
       
#    def __init__(self, basename="", input_dir=""):
#        ''' New contructor.'''
#        DataManager.__init__(self, basename, input_dir)
        # So something here
    
    def toDF(self, set_name):
        ''' Change a given data subset to a data Panda's frame.
            set_name is 'train', 'valid' or 'test'.'''
        DF = pd.DataFrame(self.data['basename_'+set_name])
        # For training examples, we can add the target values as
        # a last column: this is convenient to use seaborn
        # Look at http://seaborn.pydata.org/tutorial/axis_grids.html for other ideas
        if set_name == 'train':
            Y = self.data['Y_train']
            DF = DF.assign(target=Y)          
        return DF

    def DataStats(self, set_name):
        ''' Display simple data statistics'''
        DF = self.toDF(set_name)
        print DF.describe()
    
    def ShowScatter(self, var1, var2, set_name):
        ''' Show scatter plots.'''
        DF = self.toDF(set_name)
        if set_name == 'train':
            sns.pairplot(DF.ix[:, [var1, var2, "target"]], hue="target")
        else:
            sns.pairplot(DF.ix[:, [var1, var2]])

    def ShowSomethingElse(self):
        ''' Surprise me.'''
        # This does not need to be finished and tested code.
        # A sketch with what you intend to do written in English (or French) is OK.
        pass
    

if __name__=="__main__":
    # We can use this to run this file as a script and test the DataManager
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data"
        output_dir = "../res"
    else:
        input_dir = argv[1]
        output_dir = argv[2]
        
    print("Using input_dir: " + input_dir)
    print("Using output_dir: " + output_dir)
        
    basename = 'movies'
    D = DataManager(basename, input_dir)
    print D
    
    D.DataStats('train')
    D.ShowScatter(1, 2, 'train')