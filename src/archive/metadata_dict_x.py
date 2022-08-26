# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 17:00:03 2022

@author: siobh
"""

import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from pandas.api.types import infer_dtype
import pickle
from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
from pandas import read_csv
import seaborn as sn
import matplotlib.pyplot as plt

#coefficient of varitation
cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100 

#see if column has a constant value
def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

def process_label(input_str):
    #takes in the label string from the metadata and splits into useful parts
    #returns a dictionary
    #delimiter is hyphen, only one split as we want to split on first hyphen which delimits
    #example input string: ='V0009 - DEMO1_Mo: Date of birth (mo) (suppressed)DEMO1. What is your date of birth?Taken from: Survey of Prison Inmates, United States, 2016.'
    #example R input string: RV0048: Temporarily suppressedTaken from: Survey of Prison Inmates, United States, 2016.
    #remove the variable name before the hyphen
    
        #there are a few recoded variables with a different description format
    if input_str.startswith('RV'):
        catlabel=input_str.split(':',1)[0]
        varname=input_str.split(':',1)[0]
        var_desc=input_str.split(':',1)[1].replace('Taken from: Survey of Prison Inmates, United States, 2016.','')
    
    else:
    
        varname=input_str.split('-',1)[0]
        alldesc=input_str.split('-',1)[1]

        #split out the category label from the description
        split_catlabel=alldesc.split(':',1)
        #print(split_catlabel)
        catlabel=split_catlabel[0]
        var_desc=split_catlabel[1].replace('Taken from: Survey of Prison Inmates, United States, 2016.','')

    #return a dictionary
    output={'varname':varname,'catlabel':catlabel,'var_desc':var_desc}

    return output


metadata=pd.read_csv(r'C:\Users\siobh\OneDrive\Masters\Dissertation\us_prisonsers\data\processing_config\variable_metadata_scraped.csv',header=0)