# -*- coding: utf-8 -*-

from src import model
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
#define inputs
#dataset
#config
#target

###OPTIONS
#model
#stand
#norm
#ytype binary,kr
#test_train_split

#loop through different subsets?
#loop through options


#import survey of prison inmates
#important to keep the empty string set to na, otherwise something goes wrong with data types, which impacts value counts and more
prisoners=pd.read_csv(r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/downloaded_package/DS0001/37692-0001-Data.tsv',sep="\t",keep_default_na=False,na_values=[' '])
#import variables config
#violent_variables=pd.read_csv(r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/violent_variables.csv', index_col=0)
violent_variables=r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/processing_config/violent_variables.csv'



#filter to top 3 prisoner types and violent crime
#doing this before normalising data types in order to reduce size of dataframe
#V0401==6;  single crime, single or flat sentence with specified amount of time
#V0412==6; multiple crimes,single or flat sentence with specified amount of time
subset1=prisoners[(prisoners['V0062']==1) & (prisoners['V0063'].isin([3,11,8]))& ((prisoners['V0401']==6) | (prisoners['V0412']==6))]
#print('Index length',len(subset1.index))

#subset2- include non-specified amount of time

#subset3- include more prisoner types

#subset4- include non-specifed amount of time and more prisoner types


#subset6- include other types of crime?


subsets=[subset1]
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

md=model.model(subset1,violent_variables,model='mlp',pred_classes=2,stand=1,norm=0,ytype="binary", target="sentence_length",train_size=0.2)

#md.train()

x_train,y_train,x_test,y_test=md.x_train,md.y_train,md.x_test,md.y_test


