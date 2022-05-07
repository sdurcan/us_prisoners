# -*- coding: utf-8 -*-

from src import model
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from src import load
from sklearn.model_selection import train_test_split
from src import dataset_processor as dp
import copy
import utils

prisoners,subset3=load.import_prisoners()
subset=subset3
target='sentence_length'
th=20
stand=0
norm=1
test_size=0.2
Model='mlp'


#returns a dataframe to pass into dataset_processor.encode_and_scale()
config=load.import_config()

#initialise dataset_processor object
dataset_processor=dp.dataset_processor(subset,config,target="sentence_length",th=th)

#calculate sentence length before doing the preprocessing
dataset_processor.calc_target()

#apply one hot encoding and scaling
#can stop scaling of continuous variables by setting stand=0 and norm=0
dataset_processor.encode_and_scale(stand=stand,norm=norm)

#get the name of the target column the dataset processor
#this is dynamic based on the threshold
target=dataset_processor.target_name
prepped_subset=dataset_processor.dataset_out

#save prepped subset
fdir='C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/output/processed_subset/'
prefix='subset'
utils.name_and_pickle(prepped_subset,fdir,prefix,ext='pkl')

#dropped any columns containing information about the target
X=prepped_subset.drop(labels=target,axis=1)
y=copy.deepcopy(prepped_subset[target])

#split into test and training data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=25)

#then we create the model
md=model.model(X_train, X_test, y_train, y_test,model=Model)

#then we get the info about the model
md.train()
md.predict()
md.get_scores()



