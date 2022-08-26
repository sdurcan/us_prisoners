# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:59:35 2022

@author: siobh
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
#from src import dataset_processor as ds
import dataset_processor as ds
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from pprint import pprint
import sklearn.datasets
import sklearn.metrics
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

#import autosklearn.classification

class model:
    
    def __init__(self,X_train, X_test, y_train, y_test,model='logr'):
        self.x_train=X_train
        self.x_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        #self.target=target
        self.model=model
          
        #self.train()
        #self.predict()
        #self.get_scores()
               
    def linr(self):

        self.reg = LinearRegression().fit(self.x_train, self.y_train)
        #The best possible score is 1.0 and it can be negative
        print(self.reg.score(self.x_train, self.y_train))

        #test predictor
        #The best possible score is 1.0 and it can be negative
        self.y_pred=self.reg.predict(self.x_test)
        #y_pred = np.minimum(14600., np.maximum(0., reg.predict(x_test)))

        self.test_score=self.reg.score(self.x_test,self.y_test)
        self.train_score=self.reg.score(self.x_train,self.y_train)
        self.make_plot()
        
        print(self.train_score)
        print(self.test_score)

        return self.y_pred, self.train_score,self.test_score
    
    def logr(self):
        self.lgclf = LogisticRegression(random_state=225,max_iter=100000,solver='liblinear',penalty='l1').fit(self.x_train, self.y_train)
        
        self.y_pred=self.lgclf.predict(self.x_test)
        self.train_score=self.lgclf.score(self.x_train,self.y_train)
        self.test_score=self.lgclf.score(self.x_test,self.y_test)

        return self.y_pred, self.train_score,self.test_score
    
    def mlp(self):
        
        self.mlp = MLPClassifier(max_iter=100000,solver='adam',alpha= 3).fit(self.x_train, self.y_train)
        self.y_pred=self.mlp.predict(self.x_test)
        self.train_score=self.mlp.score(self.x_train,self.y_train)
        self.test_score=self.mlp.score(self.x_test,self.y_test)
        
        self.params=self.mlp.get_params()
        self.train_proba=self.mlp.predict_proba(self.x_train)
        self.test_proba=self.mlp.predict_proba(self.x_test)
        #predict_log_proba(X)
        
        #print(self.train_score)
        #print(self.test_score)
    
    def rf(self):
        # make predictions using random forest for classification

        # define dataset

        # define the model
        self.rf = RandomForestClassifier()
        # fit the model on the whole dataset
        self.rf.fit(self.x_train, self.y_train)

        # evaluate the model
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(self.rf, self.x_train, self.y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        # report performance
        print('Rf Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    
    '''
    def auto_classifier(self):
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120,per_run_time_limit=30,tmp_folder='/tmp/autosklearn_classification_example_tmp',)
        automl.fit(self.x_train, self.y_train)

        self.y_pred = automl.predict(self.x_test)
        
        self.test_score=sklearn.metrics.accuracy_score(self.y_test, self.y_pred)
        
        print(automl.leaderboard())
    '''

    def get_scores(self):
        print('Train score',round(self.train_score,2))
        print('Test score',round(self.test_score,2))
    
    def train(self):
        
        if self.model=="lnr":
            self.linr()
        
        elif self.model=="logr":
            self.logr()
        
        elif self.model=="mlp":
            self.mlp()
        
        elif self.model=="rf":
            self.rf()
        
        elif self.model=='auto_class':
            self.auto_classifier()
        
        else:
         raise ValueError("Unknown model type. Please extend or check input")
    
    def predict(self):
        
        return self.y_pred
        
        #if self.model=="lr":

            #self.y_pred=self.lr()[0]
    
    '''
    def make_plot(self):
        #up until now we have propogated indexes from original full file of 25k through each copy or slice of data
        #this is to make sure correct rows are being joined
        #we now need to copy and reset indexes, otherwise the plot x axis shows up to 25k 
        x=list(self.x_test.reset_index().index)
        y=self.y_test.values
        y1= np.minimum(5000., np.maximum(100000., self.y_pred))
        #y1=self.reg.predict(self.x_test)#[0:220]


        plt.scatter(x, y, color = 'red',label="True labels",alpha=0.8)
        plt.scatter(x, y1, color = 'blue',alpha=0.8,label="Predictions")
        plt.title('Actual vs Predictions')
        plt.xlabel('Instance')
        plt.ylabel(self.target)
        plt.legend()
        #plt.ylim(bottom=0,top=0.0002)
        plt.show()
    '''
    
    def get_weights(self):
        #save the coefficients of the model
        importance=list(self.lgclf.coef_[0])
        #need absolute values to assess importance
        imp=list(importance)
        imp=[abs(ele) for ele in importance]
        classes=[np.sign(ele) for ele in importance]
        fnames=self.x_train.columns
        f_weights=tuple(zip(fnames, imp, classes))

        self.f_weights=list(f_weights)

        self.f_weights.sort(key=lambda x:x[1])
        
        self.f_pos=[[feature,weight] for (feature,weight,sign) in f_weights if sign==1]

        self.f_neg=[[feature,weight] for (feature,weight,sign) in f_weights if sign==-1]
        
        '''
        for i,v in enumerate(f_weights):
        	print(v)
        # plot feature importance
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()
        '''

    
