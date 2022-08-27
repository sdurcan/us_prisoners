# -*- coding: utf-8 -*-
import os
print(os.getcwd())
os.chdir('C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers')
print(os.getcwd())

import src.load as load
import src.utils as utils
import src.dataset_processor as dataset_processor
import src.prep as prep
(print('stop'))
#from src import load, utils, dataset_processor, prep
import pandas as pd
import numpy as np
import copy
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel

# evaluate RFE for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

# to prep for passing auto ml encode, impute, scale all need to be set to 0; auto ml expects unencoded and unscaled features
#demse_violent_subset=prep(th=20,inc_max_range=1,encode=0,impute=0,scale=0,ordinal_off=0)

#load the dataframe from the tsv file
path=r'downloaded_package/DS0001/37692-0001-Data.tsv'
prisoners=pd.read_csv(path,sep="\t",keep_default_na=False,na_values=[' '])

#pass the full dataframe to a filter function that keeps instances with inmate types 3,11,8 and controlling offense violent
violent=load.filter_violent_offenses_sentenced(prisoners)

#to prep for passing directly to a model or aif360 by deriving, encoding, imputing and scaling
encoded_violent_subset=prep.prep(violent,enc=1, scale=1, impute=1, years='ordinal', th=25,low_freq_code_counts=0)

#save prepped subset
fdir=os.getcwd()
prefix='encoded_violent_subset'
utils.name_and_pickle(encoded_violent_subset,fdir,prefix,ext='pkl')
print(f'Saving {prefix} in {fdir}')
#set and drop target column the dataset processor
#column name is dynamic based on the threshold
target='sentence_above_25yrs'

#drop the target column
X=encoded_violent_subset.drop(labels=target,axis=1)
#this column has a bug in it meaning all values are nan
X.drop('pp_low_freq_codes_sum',axis=1,inplace=True)
y=copy.deepcopy(encoded_violent_subset[target])

#split into test and training data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=25)







