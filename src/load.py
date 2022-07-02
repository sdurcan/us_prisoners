# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 14:00:23 2022

@author: siobh
"""
import pandas as pd
import numpy as np
import copy
import pickle


#import survey of prison inmates
#important to keep the empty string set to na, otherwise something goes wrong with data types, which impacts value counts and more
prisoners=pd.read_csv(r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/downloaded_package/DS0001/37692-0001-Data.tsv',sep="\t",keep_default_na=False,na_values=[' '])
#import variables config
#violent_variables=pd.read_csv(r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/violent_variables.csv', index_col=0)
violent_variables=r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/processing_config/violent_variables.csv'


def import_prisoners(path="",subset=3):
    if path=="":
        path=r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/downloaded_package/DS0001/37692-0001-Data.tsv'
    
    #print('Path type is',type(path))
    prisoners=pd.read_csv(path,sep="\t",keep_default_na=False,na_values=[' '])
    
    #need to sort mixed vals in 'V0772' contains state information
    
    subset=create_sentence_subsets(prisoners,subset=subset)
    #37692-0001-Data
    return prisoners, subset

def import_starter_config(path=""):
    
    #violent variables
    path=r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/processing_config/violent_variables.csv'

    variables=pd.read_csv(path, index_col=0)
    #print('#### config type after reading in df',type(variables))
    #TO DO: change this to treatment?
    #depending on target variable, some columns may need to be treated different
    #for example, some questions are skipped by the survey in certain subsets of the population
    #then these would be treated as nan
    #changed approach, now have a column about wether to include these
    config=variables[variables['include_violent_sent_predictor']==1]
    #config=config['treatment'].isin(['cont_wnans','one_hot','transform','binary_wnans'])
    return config

def create_sentence_subsets(prisoners,subset):
    #takes in the dataset of us prisonerss as a dataframe and splits it into three subsets
    
    #filter to top 3 prisoner types and violent crime
    #doing this before normalising data types in order to reduce size of dataframe
    #V0401==6;  single crime, single or flat sentence with specified amount of time
    #V0412==6; multiple crimes,single or flat sentence with specified amount of time
    #V0062- controlling offence type is violent
    #V0063- inmate type
    #3 = Inmate NOT incarcerated for a parole or Probation Violation and NOT on parole or probation at time of arrest
    #11 = Probation Violator WITH new sentenced Offenses
    #8 = Parole Violator WITH new sentenced Offenses
    if subset==1:
        subset=prisoners[(prisoners['V0062']==1) & (prisoners['V0063'].isin([3,11,8]))& ((prisoners['V0401']==6) | (prisoners['V0412']==6))]
    
    elif subset==2:
    #subset2
    #extend the above data set to include those with a flat sentence with indeterminate (life etc)
    #excluding intermittent weekend/nights sentences
        subset=prisoners[  (prisoners['V0062']==1) & (prisoners['V0063'].isin([3,11,8]))  & (  (prisoners['V0401'].isin([1,2,3,4,6])) | (prisoners['V0412'].isin([1,2,3,4,6]))  ) ]
    
    elif subset==3:
    #subset3
    #extend the above dataset to include prisoners with a range or indeterminate sentence
    #as well as those with a flat sentence of life
    #edit 401 and 412: not in -9-8,-1,4
    #reapce with V0400-1 or 2 OR V0411==1 or 2
        subset=prisoners[  (prisoners['V0062']==1) & (prisoners['V0063'].isin([3,11,8])) & (prisoners['V0400'].isin([1,2]) | prisoners['V0411'].isin([1,2])) & (  (~prisoners['V0401'].isin([-9,-1,4])) | (~prisoners['V0412'].isin([-9,-1,4]))  ) ]

    elif subset==4:
        subset=prisoners[  (prisoners['V0062']==1) ]
        
    return subset

def offense_variables(path=""):
    
    if path=="":
        path=r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/processing_config/offense_variables'
    
    offense_variables=pickle.load( open( path, "rb" ) )
    return offense_variables

def victim_injuries(path=""):
    
    if path=="":
        path=r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/processing_config/victim_injuries.pkl'
    
    victim_injuries=pickle.load( open( path, "rb" ) )
    return victim_injuries