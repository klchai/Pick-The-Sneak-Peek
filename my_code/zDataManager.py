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
import numpy

# Mother class
import data_manager

class DataManager(data_manager.DataManager):
    '''This class reads and displays data. 
       With class inheritance, we do not need to redefine the constructor,
       unless we want to add or change some data members.
       '''
       
    def __init__(self, basename="", input_dir=""):
        ''' New contructor.'''
        data_manager.DataManager.__init__(self, basename, input_dir)
        self.labelnums = len(self.data['Y_train'][0,:])
        self.label_name = None
        # So something here

    def getLabel_name(self,label_name):
        self.label_name = label_name[0:self.labelnums]

    def ToArray(self,x):
        if isinstance(x,numpy.ndarray):
            return x
        else:
            return x.toarray()

    def toDF(self, set_name):
        ''' Change a given data subset to a data Panda's frame.
            set_name is 'train', 'valid' or 'test'.'''
        DF = pd.DataFrame(self.ToArray(self.data['X_' + set_name]))
        # For training examples, we can add the target values as
        # a last column: this is convenient to use seaborn
        # Look at http://seaborn.pydata.org/tutorial/axis_grids.html for other ideas
        if set_name == 'train':
            Y = self.data['Y_train']
            for i in range(self.labelnums):
                if self.label_name == None:
                    DF['label_' + str(i)] = Y[:, i]
                else:
                    DF[self.label_name[i]] = Y[:,i]
        return DF


    def DataStats(self, set_name):
        ''' Display simple data statistics'''
        DF = self.toDF(set_name)
        print(DF.describe())
    
    def Init_ShowScatter(self, var1, var2, set_name,label=None):
        DF = self.toDF(set_name)
        ''' Show scatter plots.'''
        if label == None:
            plt.scatter(DF[var1], DF[var2], marker='o', color='r', label='1', s=20)
            plt.xlabel('feature_' + str(0))
            plt.ylabel('feature_' + str(5))
        else:
            plt.scatter(DF[var1][DF[label] == 1], DF[var2][DF[label] == 1], marker='o', color='r', label='1', s=20)
            plt.scatter(DF[var1][DF[label] == 0], DF[var2][DF[label] == 0], marker='o', color='b', label='0', s=20)
            plt.legend(loc='upper right')
            plt.xlabel('feature_' + str(0))
            plt.ylabel('feature_' + str(5))
            plt.title(label)
        plt.show()

    def ShowScatter(D, var1, var2, set_name, label):
        ''' Show scatter plots.'''
        DF = D.toDF(set_name)
        if set_name == 'train':
            sns.pairplot(DF.ix[:, [var1, var2, label]], hue=label)
        else:
            sns.pairplot(DF.ix[:, [var1, var2]])
        sns.plt.show()

    def ShowDistplot(self, set_name,var1,var2=None):
        ''' Show histogram plots.'''
        DF = self.toDF(set_name)
        if var2==None:
            g = sns.FacetGrid(DF)
        else:
            g = sns.FacetGrid(DF,var2)
        g.map(sns.distplot, var1)
        sns.plt.show()

    def ShowJointplot(self, var1, var2, set_name):
        '''Show joint plot.'''
        DF = self.toDF(set_name)
        sns.jointplot(x=var1, y=var2, data=DF, kind='kde')
        sns.plt.show()

    def ShowHeatmap(self, set_name,choose=None):
        DF = self.toDF(set_name)
        if choose == None:
            featurelist = [0, 1,2,3,4,5,6,7,8,9]
            if set_name == 'train':
                choose = featurelist + self.label_name
            else:
                choose = featurelist
        sns.heatmap(DF.ix[0:50, choose])
        sns.plt.show()


    def ShowCorrcoefHeatmap(self, set_name,choose=None):
        '''Show Heatmap of Correlation Matrix'''
        DF = self.toDF(set_name)
        corr = DF.corr(method='pearson', min_periods=1)
        sns.heatmap(corr)
        sns.plt.show()

    def PCA(self, num, set_name):
        pca = PCA(n_components=num)
        return pca.fit_transform(np.array(self.ToArray(self.data['X_' + set_name])))

    def ShowJointGrid_PCA(self, num, set_name,label):
        from mpl_toolkits.mplot3d import Axes3D
        data = self.PCA(num, set_name)
        DF = self.toDF(set_name)
        fig = plt.figure()
        if num == 3:
            ax = fig.gca(projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=DF[label], cmap=plt.cm.spectral)
        elif num == 2:
            ax = fig.gca()
            ax.scatter(data[:, 0], data[:, 1], c=DF[label], cmap=plt.cm.spectral)
        plt.show()

    def ShowJointPlot_PCA(self, set_name):
        data = self.PCA(2, set_name)
        sns.jointplot(x=data[:, 0], y=data[:, 1], kind='kde')
        sns.plt.show()
