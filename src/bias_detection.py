# -*- coding: utf-8 -*-
"""
Created on Sun May  8 15:08:23 2022

@author: siobh
"""

# Load all necessary packages
import sys
sys.path.insert(1, "../")  

import numpy as np
np.random.seed(0)

from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

from IPython.display import Markdown, display

#Step 2 Load dataset, specifying protected attribute

dataset_orig = GermanDataset(
    protected_attribute_names=['age'],                                                            
    privileged_classes=[lambda x: x >= 25],     
    features_to_drop=['personal_status', 'sex'] 
)

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]

#One simple test is to compare the percentage of favorable results for the privileged and unprivileged groups
#subtracting the former percentage from the latter. 
#A negative value indicates less favorable outcomes for the unprivileged groups. 
#This is implemented in the method called mean_difference on the BinaryLabelDatasetMetric class

metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Original training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())