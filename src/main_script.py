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
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, chi2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree

from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, accuracy_score, f1_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

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

#Decision tree with Kbest
pipe=Pipeline([('selector',SelectKBest()),('classifier',DecisionTreeClassifier())])

#[5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,281]
parameters={'classifier__max_depth':[5,10,20],
            'classifier__criterion':['gini','entropy'],
            'classifier__max_leaf_nodes':[10,20,30],
            'classifier__min_samples_split':[10,20,30],
            'classifier__min_samples_leaf':[10,20,30],
            'selector__k':[5,10,15,20,25,30,35,40,45,50,100,150,200,250,281],
            'selector__score_func':[mutual_info_classif, chi2]}

test_parameters={'classifier__max_depth':[5],
            'classifier__criterion':['gini','entropy'],
            'classifier__max_leaf_nodes':[10],
            'classifier__min_samples_split':[10],
            'classifier__min_samples_leaf':[10],
            'selector__k':[5,25],
            'selector__score_func':[mutual_info_classif, chi2]}

grid = GridSearchCV(pipe, parameters, cv=5,verbose=3,n_jobs=8,refit=False).fit(X_train, y_train)
result_df = pd.DataFrame.from_dict(grid.cv_results_, orient='columns')
result_df.to_csv('gridsearchresults.csv')







