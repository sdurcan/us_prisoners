# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:59:35 2022

@author: siobh
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
from src import dataset_processor as ds
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from pprint import pprint
import sklearn.datasets
import sklearn.metrics

#import autosklearn.classification

class model:
    
    def __init__(self,dataset,config_file,model="lg",pred_classes=0,stand=0,norm=0,ytype="lr", target="sentence_length",train_size=0.8):
        self.data_in=dataset
        self.config_file=config_file
        self.model=model
        self.stand=stand
        self.norm=norm
        self.ytype=ytype
        self.target=target
        self.train_size=train_size
        self.pred_classes=pred_classes
        
        self.load_variable_config()
        self.preprocess()
        
        
        #self.train()
        #self.predict()
        #self.get_scores()
        
        #set ytype
        if self.model=='lr':
            self.ytype=='cont'
        elif self.model in ['lg','mlp']:
            if pred_classes==2:
                self.ytype=='binary'
            if pred_classes >2:
                self.ytype=='multiclass'
        
    
    #TO DO: function that loads the config from a file and puts it into dict form
    def load_variable_config(self):
        variables=pd.read_csv(self.config_file, index_col=0)
        #TO DO: change this to treatment?
        #depending on target variable, some columns may need to be treated different
        #for example, some questions are skipped by the survey in certain subsets of the population
        #then these would be treated as nan
        
        if self.target=="sentence_length":
            self.config=variables[variables['treatment_violent_lr'].isin(['cont_wnans','one_hot','transform'])]
        else:
            #A column giving default treatment of columns
            #if the columns don't have a treatment type then they won't be picked up
            self.config=variables[variables[self.treatment].isin(['cont_wnans','one_hot','transform'])]
        
    def preprocess(self):
            
        #initialise dataset_processor object and encode and process variables
        #TO DO: rename encodconing as processing maybe?
        self.processor=ds.dataset_processor(self.data_in,self.config,ytype=self.ytype,pred_classes=self.pred_classes,stand=self.stand,norm=self.norm,train_size=self.train_size)
        
        

        
        #print('after processing, before sentence length calculated',type(self.processor.dataset_out))
        #print('nans',self.processor.dataset_out.isna().sum().sum())
        #print(len(self.processor.dataset_out.index))
        #print(self.processor.dataset_out.index[0:10])
    
        

        #calculate the sentence length feature to predict
        if self.target=="sentence_length":
            self.processor.calc_sentence_length()
        else:
         raise ValueError('Unknown target variable')
         
        self.dataset_out=self.processor.dataset_out

    
        #print('after calculating snentence length',type(self.processor.dataset_out))
        #print('nans',self.processor.dataset_out.isna().sum().sum())
        #print(len(self.processor.dataset_out.index))
        #print(self.processor.dataset_out.index[0:10])
        #print('nans sent length',self.processor.dataset_out['binary_sentence_length'].isna().sum().sum())
    

        #run split method
        self.processor.split()

        
        #print('after splitting',type(self.processor.dataset_out))
        #print('nans',self.processor.dataset_out.isna().sum().sum())
        #print(len(self.processor.dataset_out.index))
        #print(self.processor.dataset_out.index[0:10])

        #print('y vals after split',type(self.processor.dataset_out))
        #print('nans',self.processor.y_train.isna().sum().sum())
        #print(len(self.processor.dataset_out.index))
        #print(self.processor.y_train.index[0:10])
    

        #split created new attributes- assign them
        self.x_train, self.y_train, self.x_test, self.y_test=self.processor.x_train, self.processor.y_train, self.processor.x_test, self.processor.y_test
        
        '''
        print('y-train type',type(self.y_train))
        print(self.y_train.isna().sum().sum())
        print(len(self.y_train.index))
        print(self.y_train.index[0:10])
        
        print('X-train type',type(self.x_train))
        print(self.x_train.isna().sum().sum())
        print(len(self.x_train.index))
        print(self.x_train.index[0:10])
        '''
        
    def lr(self):

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
    
    
    def lg(self):
        self.lgclf = LogisticRegression(random_state=0,max_iter=1000000).fit(self.x_train, self.y_train)
        
        self.y_pred=self.lgclf.predict(self.x_test)
        self.train_score=self.lgclf.score(self.x_train,self.y_train)
        self.test_score=self.lgclf.score(self.x_test,self.y_test)
        

        
        return self.y_pred, self.train_score,self.test_score
    
    def mlp(self):
        
        self.mlp = MLPClassifier(max_iter=100000,solver='adam').fit(self.x_train, self.y_train)
        self.y_pred=self.mlp.predict(self.x_test)
        self.train_score=self.mlp.score(self.x_train,self.y_train)
        self.test_score=self.mlp.score(self.x_test,self.y_test)
        
        self.params=self.mlp.get_params()
        self.train_proba=self.mlp.predict_proba(self.x_train)
        self.test_proba=self.mlp.predict_proba(self.x_test)
        #predict_log_proba(X)
        
        #print(self.train_score)
        #print(self.test_score)
    
    def auto_classifier(self):
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120,per_run_time_limit=30,tmp_folder='/tmp/autosklearn_classification_example_tmp',)
        automl.fit(self.x_train, self.y_train)

        self.y_pred = automl.predict(self.x_test)
        
        self.test_score=sklearn.metrics.accuracy_score(self.y_test, self.y_pred)
        
        print(automl.leaderboard())

    def get_scores(self):
        print('Train score',round(self.train_score,2))
        print('Test score',round(self.test_score,2))
    
    def train(self):
        
        if self.model=="lr":
            self.lr()
        
        elif self.model=="lg":
            self.lg()
        
        elif self.model=="mlp":
            self.mlp()
        
        elif self.model=='auto_class':
            self.auto_classifier()
        
        else:
         raise ValueError("Unknown model type. Please extend or check input")
    
    def predict(self):
        
        return self.y_pred
        
        #if self.model=="lr":

            #self.y_pred=self.lr()[0]
    
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
    
    
