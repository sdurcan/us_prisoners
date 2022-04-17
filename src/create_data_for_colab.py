# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 16:18:37 2022

@author: siobh
"""
from src import model
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle

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

md=model.model(subset1,violent_variables,model='mlp',pred_classes=2,stand=1,norm=0,ytype="binary", target="sentence_length",train_size=0.2)

#serialise the data for use in colab
#pickle.dump(md.dataset_out,open('C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/output/processed_data.pkl','wb'))
md.dataset_out.to_csv('C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/output/processed_data.csv')
print('processed data saved')
