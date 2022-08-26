# -*- coding: utf-8 -*-
"""
Created on Sun May 22 15:51:44 2022

@author: siobh
"""
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
import numpy as np

#remove low variance

def remove_low_variance(X,threshold=0.8):
    sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
    return sel.fit_transform(X)

def k_best(x,y,k=100,measure=chi2):
    # f_classif, mutual_info_classif
    return SelectKBest(measure, k=k).fit_transform(x, y)

def select_from_model(model_,X,y,prefit=True):

    selector = SelectFromModel(model_)
    selector = selector.fit(X, y) 

    selected=selector.transform(X)

    return selected

    

    
