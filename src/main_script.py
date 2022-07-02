# -*- coding: utf-8 -*-

import model
import load
import model 
import pandas as pd
import numpy as np
import copy
import utils
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

import dataset_processor as dataset_processor
from matplotlib import pyplot
import feature_selection as fs
#from src import load
#from src import model
#from src import dataset_processor as dp

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

'''
import warnings
warnings.filterwarnings("ignore")
'''

#README
#Download the file available here and put in /ICSPRdownload

prisoners,subset=load.import_prisoners(subset=3)
#subset=subset4
target='sentence_length'
th=20
stand=1
norm=1
test_size=0.2
Model='logr'

#returns a dataframe to pass into dataset_processor.encode_and_scale()
#config=load.import_config()

#TODO: rename config to variable dict
#initialise dataset_processor object
#TO DO: if config is blank, use default config
dp=dataset_processor.dataset_processor(subset)

#some variables need to be derived before passing to encoder scaler
#the original fields will be deleted and new entries created in the config dictionary for the derived keys
dp.set_sentence(th=th)
dp.set_offenses()
dp.set_protected_attr()
dp.set_victim_injuries()
dp.set_victim_relationship()

#at this point, the dp.dataset atribute holds the derived vars, but full encoding has not taken place
partial_prepped=dp.dataset
#save prepped subset
#fdir='C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/output/processed_subset/'
fdir='C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/output/processed_subset/'
prefix='collapsed_subset'
utils.name_and_pickle(partial_prepped,fdir,prefix,ext='pkl')



#apply one hot encoding and scaling
#have to call this after other vals
#can stop scaling of continuous variables by setting stand=0 and norm=0
#TODO: make it possible that variables don't have to be encoded
dp.encode_and_scale(stand=stand,norm=norm)



#TODO: remove skipped encoding

#get the name of the target column the dataset processor
#this is dynamic based on the threshold
target=dp.sent_th_name
prepped_subset=dp.dataset_out

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
#md.get_weights()

#pass trained model to feature selector
#prefit must be set to true
#returns a transformed version of x we can then retrain on
#we don't need to worry about y; still all the same rows
#best_features=fs.select_from_model(md.lgclf,X_train,y_train,prefit=True)

lgclf = LogisticRegression(random_state=225,max_iter=100000,solver='liblinear',penalty='l1')
        
#self.y_pred=self.lgclf.predict(self.x_test)
#self.train_score=self.lgclf.score(self.x_train,self.y_train)
#self.test_score=self.lgclf.score(self.x_test,self.y_test)


rfe = RFE(estimator=lgclf,n_features_to_select=100)
model = lgclf()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=5, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

'''
'''
selector = SelectFromModel(md.lgclf,prefit=True)

#X_train.columns
#selector = selector.fit(X, y) 

best_X_train=selector.transform(X_train)
best_X_test=selector.transform(X_test)


fdir='C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/output/processed_subset_topf/'
prefix='top_f_subset'
#go back to original dataset (with y vals) to save to use in autosklearn
topfx=selector.transform(X)
y1=y.values.reshape(1, -1)
topfy=selector.transform(y1)
utils.name_and_pickle(topfx,fdir,prefix,ext='pkl')
utils.name_and_pickle(topfy,fdir,prefix,ext='pkl')

#save for use in auto sklearn


#retrain model
#then we recreate the model
md=model.model(best_X_train, best_X_test, y_train, y_test,model='mlp')

#then we get the info about the model
md.train()
md.predict()
md.get_scores()
#md.get_weights()

rfe = RFE(estimator=md.lgclf, n_features_to_select=596)
# fit the model
rfe.fit(X, y)
# transform the data
X, y = rfe.transform(X, y)
'''