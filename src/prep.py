# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:35:03 2022

@author: siobh
"""
from src import dataset_processor
import copy


def prep(subset, enc=1, scale=1, impute=1, years='ordinal', th=20,inc_max_range=1,low_freq_code_counts=0):
    
    #returns a dataframe to pass into dataset_processor.encode_and_scale()

    #initialise dataset_processor object
    dp=dataset_processor.dataset_processor(subset)
    print('Dataset processor object initialised')

    #years are set to ordinal, update config dataframe
    if years=='ordinal':
        dp.config.loc[dp.config['year']==1, 'enc_scale']='ordinal'
   
    #some variables need to be derived before passing to encoder scaler
    #the original fields will be deleted and new entries created in the config dictionary for the derived keys
    dp.set_sentence(th=th, inc_max_range=inc_max_range, impute=impute)
    print('Sentence calculated')
    dp.set_offenses(low_freq_code_counts=low_freq_code_counts)
    print('Offenses set')
    dp.set_protected_attr()
    print('Protected attributes set')
    dp.set_victim_injuries()
    print('Victim injuries set')
    dp.set_victim_relationship()
    print('Victim relationship set')
    dp.set_victim_age()
    print('Victim age set')
    #at this point, the dp.dataset atribute holds the derived vars, but full encoding has not taken place
    print('Starting encoding and scaling')
    dp.process(enc=enc,scale=scale)
    #cast to pandas datatypes to pass to autosklearn
    prepped_subset=copy.deepcopy(dp.dataset_out)

    return prepped_subset