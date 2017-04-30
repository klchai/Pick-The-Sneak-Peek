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
import matplotlib.pyplot as plt

# Data types
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

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
        DF = pd.DataFrame(self.data['X_'+set_name])
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
        print(DF.describe())
    
    def ShowScatter(self, var1, var2, set_name):
        ''' Show scatter plots.'''
        DF = self.toDF(set_name)
        if set_name == 'train':
            sns.pairplot(DF.ix[:, [var1, var2, "target"]], hue="target")
        else:
            sns.pairplot(DF.ix[:, [var1, var2]])
        sns.plt.show()

    def ShowDistplot(self, var1, set_name):
        ''' Show histogram plots.'''
        DF = self.toDF(set_name)
        sns.distplot(DF.ix[:, [var1]])
        sns.plt.show()

    def ShowJointplot(self, var1, var2, set_name):
        '''Show joint plot.'''
        DF = self.toDF(set_name)
        sns.jointplot(x=var1, y=var2, data=DF, kind='kde')
        sns.plt.show()

    def ShowHeatmap(self, choose, set_name):
        '''Show Heatmap plot.'''
        DF = self.toDF(set_name)
        sns.heatmap(DF.ix[choose, :])
        sns.plt.show()

    def ShowCorrcoefHeatmap(self, set_name):
        '''Show Heatmap of Correlation Matrix'''
        DF = self.toDF(set_name)
        sns.heatmap(np.corrcoef(np.array(DF)))
        sns.plt.show()

    def PCA(self, num, set_name):
        DF = self.toDF(set_name)
        del DF['target']
        pca = PCA(n_components=num)
        return pca.fit_transform(np.array(DF))

    def ShowJointGrid_PCA(self, num, set_name):
        from mpl_toolkits.mplot3d import Axes3D
        data = self.PCA(num, set_name)
        DF = self.toDF(set_name)
        fig = plt.figure()
        if num == 3:
            ax = fig.gca(projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=DF['target'], cmap=plt.cm.spectral)
        elif num == 2:
            ax = fig.gca()
            ax.scatter(data[:, 0], data[:, 1], c=DF['target'], cmap=plt.cm.spectral)
        plt.show()

    def ShowJointPlot_PCA(self, set_name):
        data = self.PCA(2, set_name)
        DF = self.toDF(set_name)
        sns.jointplot(x=data[:, 0], y=data[:, 1], kind='kde')
        sns.plt.show()
    

if __name__=="__main__":
    # We can use this to run this file as a script and test the DataManager
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data"
        output_dir = "../res"
    else:
        input_dir = argv[1]
        output_dir = argv[2];
        
    print("Using input_dir: " + input_dir)
    print("Using output_dir: " + output_dir)
    
    basename = 'movies'
    D = DataManager(basename, input_dir)
    print(D)
    
    D.DataStats('train')
    D.ShowScatter(1, 2, 'train')
    D.ShowDistplot(1, 'train')
    D.ShowJointplot(1, 2, 'train')
    D.ShowJointPlot_PCA('train')
    D.ShowHeatmap(range(10), 'train')
    D.ShowCorrcoefHeatmap('train')
    D.ShowJointGridofPCA(2, 'train')
    D.ShowJointGridofPCA(3, 'train')