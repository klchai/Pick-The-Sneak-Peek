#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 08:04:23 2017

@author: isabelleguyon

This is an example of program that tests the Iris challenge Data Manager class.
Another style is to incorporate the test as a main function in the Data manager class itself.
"""
from zDataManager import DataManager
#input_dir = "../public_data"
#output_dir = "../res"
mypath = "../sample_code"
from sys import argv, path
from os.path import abspath
path.append(abspath(mypath))

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
    print D
    
    D.DataStats('train')
    D.ShowScatter(1, 2, 'train')
    
#basename = 'movies'
#D = DataManager(basename, input_dir)
#print D
#   
#D.DataStats('train')
#D.ShowScatter(1, 2, 'train')