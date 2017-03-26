#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 08:04:23 2017

@author: isabelleguyon

This is an example of program that tests the Iris challenge Data Manager class.
Another style is to incorporate the test as a main function in the Data manager class itself.
"""
from zDataManager import DataManager
input_dir = "../public_data"
output_dir = "../res"

    
basename = 'movies'
D = DataManager(basename, input_dir)
print D
    
D.DataStats('train')
D.ShowScatter(1, 2, 'train')
