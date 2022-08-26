# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 13:49:55 2022

@author: siobh
"""

import sys
sys.path.insert(1, "../")  

import numpy as np
np.random.seed(0)

from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

from IPython.display import Markdown, display

'''
features=np.array of features
labels=label for each instance (positive or negative)
protected_attributes-np.array subset of features for which fairness is desired
feature_names= list of features
label_names- names describing each label (harsh or not harsh)
protected_attribute_names (list(str))
privileged_protected_attributes
unprivileged_protected_attributes 
instance_names
instance weights- could apply survey weights
instance_weights_name- column in df
'''