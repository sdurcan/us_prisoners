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
stand=1
norm=0
test_size=0.2
Model='logr'


#returns a dataframe to pass into dataset_processor.encode_and_scale()
config=load.import_config()
#print('Config type after load',type(config))

#inittialise dataset_processor object
dataset_processor=dp.dataset_processor(subset,config,target="sentence_length",th=th)

#calculate sentence length before doing the preprocessing
dataset_processor.calc_target()

#can stop scaling of continuous variables by setting stand=0 and norm=0
dataset_processor.encode_and_scale(stand=stand,norm=norm)

#we can get the encoded dataset from the dataset processor
target=dataset_processor.target_name
prepped_subset=dataset_processor.dataset_out

#save prepped subset
#fdir=r'C:\Users\siobh\OneDrive\Masters\Dissertation\us_prisonsers\output/processed_subset'
fdir='C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/output/processed_subset/'
prefix='subset'
utils.name_and_pickle(prepped_subset,fdir,prefix,ext='pkl')


X=prepped_subset.drop(labels=target,axis=1)
y=copy.deepcopy(prepped_subset[target])

#we need to split into test and training data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=25)

#then we create the model
md=model.model(X_train, X_test, y_train, y_test,model=Model)

#then we get the info about the model
md.train()
md.predict()
md.get_scores()


###OPTIONS
#model
#stand
#norm
#ytype binary,kr
#test_train_split
#loop through different subsets?
#loop through options





'''
norms=[0,1]
stands=[0,1]
class_counts=[2,5,20]
train_sizes=[0.1,0.2,0.3]
model_list=['auto_class','mlp','lg']

#mlp params
activations=['identity’, ‘logistic’, ‘tanh’, ‘relu']
solvers=['lbfgs’,‘sgd’, ‘adam']
alphas=[]
learning_rates=['constant’, ‘invscaling’, ‘adaptive']


#mlp configurations
#curated bin edges

results={}

count=0
print(count)
for subset in subsets:
    for stand in stands:
        for norm in norms:
            for class_counts in class_counts:
                for train_size in train_sizes:
                    for Model in model_list:
                    #get results of model trained on this set of variables
                    
                        md=model.model(subset1,violent_variables,model=Model,pred_classes=class_counts,stand=stand,norm=norm,ytype="binary", target="sentence_length",train_size=train_size)
    
                        md.train()
                        md.predict()
                        md.get_scores()
                        #add results to dictionary
                        results[count]={}
                        results[count]['model']=Model
                        results[count]['classes']=class_counts
                        results[count]['Stand']=stand
                        results[count]['Norm']=norm
                        results[count]['Train size']=train_size
                        count=+1
                        print(count,Model,train_size,class_counts,norm,stand)

'''



