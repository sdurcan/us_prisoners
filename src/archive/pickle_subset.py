# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:45:39 2022

@author: siobh
"""

from src import model
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from src import load
from sklearn.model_selection import train_test_split
from src import dataset_processor as dp
import copy
import utils

th=20
prisoners,subset3=load.import_prisoners()

#returns a dataframe to pass into dataset_processor.encode_and_scale()
config=load.import_config()

#initialise dataset_processor object
dataset_processor=dp.dataset_processor(subset3,config,target="sentence_length",th=th)

dataset_processor.calc_protected_attr()
dataset_processor.calc_target()

subset3=dataset_processor.dataset_out

fdir=r'C:\Users\siobh\OneDrive\Masters\Dissertation\us_prisonsers\output/'
utils.name_and_pickle(subset3,fdir,'subset3_th20_original',ext='pkl')