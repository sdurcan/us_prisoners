# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 13:49:55 2022

@author: siobh
"""

from src import dataset_processor
from src import utils
from src import load
from src import prep2
prep=prep2.prep



from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

from IPython.display import Markdown, display
import pandas as pd

prepped_subset=pd.read_pickle(r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/output/prepped_nopp_offcounts.csv')

victim_sex=[col for col in prepped_subset.columns if 'victim_male' in col][0]

victim_white=[col for col in prepped_subset.columns if 'victim_white' in col][0]

offender_white=[col for col in prepped_subset.columns if 'victim_white' in col][0]

protected={offender_white:1,victim_sex:1,victim_white:1}

sentence_col='sentence_above_20yrs'

aif360_prisoners=BinaryLabelDataset(favorable_label=0.0, unfavorable_label=1.0, df=prepped_subset,label_names=['sentence_above_20yrs'],protected_attribute_names=list(protected.keys()))
#drop freq counts
#label_name=sentence_col

#privileged_groups (list(dict)) – Privileged groups. Format is a list of dicts where the keys are protected_attribute_names 
#and the values are values in protected_attributes. Each dict element describes a single group. See examples for more details.
#unprivileged_groups (list(dict)) – Unprivileged groups in the same format as

#privileged_groups=[{'V1952-x0_2':1},{'victim_white-x0_1.0':1},{'victim_sex-x0_0.0':1}]
#unprivileged_groups=[{'V1952-x0_2':0},{'victim_white-x0_1.0':0},{'victim_sex-x0_0.0':0}]


#privileged_groups=[{'victim_white-x0_1.0':1}]
#unprivileged_groups=[{'victim_white-x0_1.0':0}]

protected_tuples=[ ({'victim_white-x0_1.0':1},{'victim_white-x0_1.0':0}),({'V1952-x0_2':1},{'V1952-x0_2':0}),({'victim_sex-x0_0.0':0},{'victim_sex-x0_0.0':1})]

#privileged_groups=[{'V1952-x0_2':1}]
#unprivileged_groups=[{'V1952-x0_2':0}]

#privileged_groups=[{'victim_sex-x0_0.0':1}]
#unprivileged_groups=[{'victim_sex-x0_0.0':0}]

for thing in protected_tuples:
    print(thing)
    privileged_groups=thing[1]
    print(privileged_groups)
    unprivileged_groups=thing[0]
    print(unprivileged_groups)

    metric = BinaryLabelDatasetMetric(aif360_prisoners, unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
    
    print(metric.statistical_parity_difference())
    print(metric.disparate_impact())
    print(metric.smoothed_empirical_differential_fairness())
    print(metric.consistency())
    