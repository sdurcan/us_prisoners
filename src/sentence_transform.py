# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:42:16 2022

@author: siobh
"""

import copy as copy
import numpy as np
import operator
import pickle
from src import sentence_transform_config
import pandas as pd

class sentence_transformer:
    
    """

    
    On instantiation, makes target column '1' for any rows where the sentence length is above the threshold 
    Provides updated dataset with new sentence_threshold column that is '1' if the sentence length for that
    prisoner is above the given threshold. The dataset_out method returns a dataset with the new column present 
    and all columns that were used to derive sentence length removed. 
    
    In the Survey of US Prisoners, the sentence length is stored in different variables depending on the sentence type.
    Some sentence lengths are categorical (i.e., Life) whilst others are continuous (i.e., 25 years) whilst some are range
    i.e., (25 years to life). Where a sentence is given as a range, there is a maximum and minimum length. If either the maximum or minimum number of years is
    above the threshold, the target variable is set to the positive class.  Categorical variables indicate a sentence of life of life+ are set to the positive class. 
    
    Called by dataset_processor class, not in instantiation of the data_processor object
    Called after dataset pre processing and before encoding and scaling continuous variables 
    
    Continuous variable names are loaded from a dictionary in the file sentence_length_variables in the derive/sentence folder.

    
    ...
    
    Attributes
    ----------
    dataset: dataframe
        A dataset of the 2016 Survey of US Prisoners. 
        This should contain columns that wil be used to calculate the sentence length
    
    th: int
        The threshold for binary classification. Sentence length in years. 
        
    sent_th_name: str
        The name of the sentence threshold variable to be added to the dataset. f'sentence_above_{self.th}yrs'
        
    original_cols: list of str
        The list of columns in dataset at the point of instantiation
    
    one_if_any: list of str
        A list of columns in the dataframe (created during instantiation)
        A '1' in any of these columns indicates that the target column should be 1

    Methods
    -------
    prep_cols(cols, fill_w_mean=[-9, -2, -1], fill_w_0=[-8, np.nan])
        Fills 'Don't Know/Refuse' and N/A values
        This is the same pre processing as applied in the dataset processor
    
    load_sentence_length_variable_names()
        Loads  a dictionary from the pickled file sentence_length_variables
        Provides reference dictionary when converting days and months to years
    
    convert_to_years
        For each dictionary of {'sentence type':{'y':varname,'m':varname,'d':varname}}
        Converts days and months to years and creates new column with total years
    
    binarise_continuous
        Passes each continuous variable in sentence_length_variables to
        the binarise_this_col function and appends col name to one_if_any
    
    binarise_this_col(self, conv_col)
        Takes a column (a 'years' varaible passed from binarise_continuous)
        and sets values >= threshold as 1, otherwise 0
    
    binarise_categorical
        For each of the columns defining sentences of life etc, replaces life +
        with 1, otherwise 0
    
    make_one_if_any
        Sets the target variable to '1' for any rows where at least one of the 
        'one if any' columns (indicating a sentence length above the threshold)
        are 1. 
    
    """
    def __init__(self, dp, th, inc_max_range=0, impute=0):
        
        self.dp=dp
        self.dataset=copy.deepcopy(self.dp.dataset)
        #take column list at instantiation to diff later
        self.orig_cols = self.dataset.columns
        self.impute=impute
        self.th = th
        self.sent_th_name = f'sentence_above_{self.th}yrs'
        self.to_keep=[self.sent_th_name]
        self.inc_max_range=inc_max_range
        #load the dictionaries of continuous variable sets
        #self.load_sentence_length_variable_names()
        self.load_inputs()
        self.transform()
       # store the cols that we have created and are deleting so they can be examined
        #self.get_cols_created()
        #drop the cols_created as part of processing, apart from the target variable
        self.drop(self)
        self.update_config(self)
        #copy over with the new dataset
        self.dp.dataset=self.dataset
    
    def flatten_cont_cols(self):
        self.cont_vars = []
        for sent_type, dmy in self.dmys.items():
            for unit, varname in dmy.items():
                self.cont_vars.append(varname)
                
    def prep_cont_cols(self, cols, missing=[-9, -2, -1], skipped=[-8, np.nan]):
        
        self.dataset[cols].replace(skipped, 0, inplace=True)
        
        if self.impute==1:
            for col in cols:
                self.dataset[col].replace(missing,self.dataset[col].mean(), inplace=True)
        
        if self.impute==0:
            for col in cols:
                self.dataset[col].replace(missing,np.nan, inplace=True)
    
    '''
    def load_sentence_length_variable_names(self,path=""):
        
        if path=="":
            path=r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/src/derive_vars/sentence/sentence_length_variables'
        
        self.dmys=pickle.load( open( path, "rb" ) )
    '''
    
    def load_inputs(self):
        self.cat_vars=['V0401','V0412','V0439']
        self.denoms = {'d': 365, 'm': 12, 'y': 1}
        self.shared_info=self.dp.config.loc[self.dp.config['feeds_mutual_info']=='sentence_length'].index
        #print('Shared info',list(self.shared_info))

        self.max_vars=['single_range_max', 'multi_range_max','multi2_range_max']

        self.dmys=sentence_transform_config.dmys
        if self.inc_max_range==0:
            for var in self.max_vars:
                del self.dmys[var]
        
        self.configdata={self.sent_th_name:{'enc_scale':'none','description':'sentence length collapsed','protected_characteristic':0,'include_violent_sent_predictor':1} }
        

    def transform(self):
        #empty list to add columns to
        self.one_if_any = []
        self.flatten_cont_cols()

        self.prep_cont_cols(self.cont_vars)
        #converts days and months into whole years
        #this is needed because value of 'months' and 'days' exceeds 12 and 365 respectively

        self.convert_to_years()
        
        #binarise the new 'years' columns as above or below threshold
        #appends these to self.one_if_any list

        self.binarise_continuous()
        
        #binarise the categorical columns as life+ or not
        #appends these to the self.one_if_any list

        self.binarise_categorical()
        
        # looks at all columns that have been binarised and makes target col 1 if any are 1
        self.make_one_if_any(self.sent_th_name)

    def convert_to_years(self):
        # takes in a dictionary of dictionaries
        # outer dict the sentence type or grouping of the fields to be added
        # inner dict provide a time unit (d/m/y) as key, and a
        # column name as the value. Converts the days and months into year values
        # convert each month and day field to years and then sum to create new field


        dmys_conv = {}

        self.converted_years = []

        for sent_type, dmy in self.dmys.items():
            dmys_conv[sent_type] = {}
            for unit, colname in dmy.items():
                # not skipping over years- instead divide it by 1
                # then can append everything to the list for summing
                # add y to indicate it's been converted into years
                newcol = colname+'y'

                # use np.floor because we want to round down to the nearest integer
                self.dataset[newcol] = np.floor(
                    self.dataset[colname]/self.denoms[unit])

                # add this to dictionary for summing
                dmys_conv[sent_type][unit] = newcol

            # name for converted years
            newyrs = sent_type+'c'

            # sum values add the converted value to the main dataset
            self.dataset[newyrs] = sum([self.dataset[yr]
                                      for yr in dmys_conv[sent_type].values()])

            # this will be picked up by binarise_continuous function
            self.converted_years.append(self.dataset[newyrs])

    def binarise_continuous(self):

        #print('Checking if continuous columns are above threshold')
        # we don't need to prep the continuous variables as they have already been pre processed

        for v in self.converted_years:

            vname = v.name

            self.binarise_this_col(vname)
            #print('Now have this many cols',len(self.dataset.columns))

            # get the most recent out name
            self.one_if_any.append(self.dataset[self.out_name].name)

    def binarise_this_col(self, conv_col):

        out_name = f'{conv_col}_above{self.th}'

        # then indicate if over or above threshold
        self.dataset.loc[self.dataset[conv_col] < self.th, out_name] = 0
        self.dataset.loc[self.dataset[conv_col] >= self.th, out_name] = 1

        # print(self.dataset[out_name].value_counts(dropna=False))

        #print(f'Binarised {self.dataset[out_name].name}')
        # update out_name so it can be accessed by binarise_continuous
        self.out_name = out_name

    def binarise_categorical(self):
        #print('Setting categorical sentences: life or death sentences will be set above threshold')
        
        if self.impute==1:

            # we need to do the flat sentences, which contain categorical info
            # 6 the most common value within the dataset that answered this question- going to be raplced with 0 anyway
            self.dataset['single_sent_life'] = copy.deepcopy(
                self.dataset['V0401'].replace(to_replace=[-9, -2, -1, 5, 6, np.nan], value=0))
            # we can then replace the values that indicate a life or a death sentence with 997 for adding later
            self.dataset['single_sent_life'].replace(
                to_replace=[1, 2, 3, 4], value=1, inplace=True)
            self.one_if_any.append(self.dataset['single_sent_life'].name)
     
    
            #prisoners with multiple offences
            self.dataset['multi_sent_life'] = copy.deepcopy(
                self.dataset['V0412'].replace(to_replace=[-9, -2, -1, 5, 6, np.nan], value=0))
            self.dataset['multi_sent_life'].replace(
                to_replace=[1, 2, 3, 4], value=1, inplace=True)
            self.one_if_any.append(self.dataset['multi_sent_life'].name)
            
            #multi sentences, one sentence for multiple offenses or the longest sentence covered multiple offenses
            self.dataset['multi2_sent_life'] = copy.deepcopy(
                self.dataset['V0439'].replace(to_replace=[-9, -2, -1, 5, 6, np.nan], value=0))
            self.dataset['multi2_sent_life'].replace(
                to_replace=[1, 2, 3, 4], value=1, inplace=True)
            self.one_if_any.append(self.dataset['multi2_sent_life'].name)
        
        if self.impute==0:
            self.dataset['single_sent_life'] = copy.deepcopy(self.dataset['V0401'].replace([-9, -2, -1, 5, 6, np.nan], [np.nan, np.nan,np.nan, 0, 0, 0]))
            # we can then replace the values that indicate a life or a death sentence with 997 for adding later
            self.dataset['single_sent_life'].replace( to_replace=[1, 2, 3, 4], value=1, inplace=True)
            self.one_if_any.append(self.dataset['single_sent_life'].name)

            #prisoners with multiple offences
            self.dataset['multi_sent_life'] = copy.deepcopy(
                self.dataset['V0412'].replace([-9, -2, -1, 5, 6, np.nan], [np.nan, np.nan,np.nan, 0, 0, 0]))
            self.dataset['multi_sent_life'].replace(
                to_replace=[1, 2, 3, 4], value=1, inplace=True)
            self.one_if_any.append(self.dataset['multi_sent_life'].name)
        
            #multi sentences, one sentence for multiple offenses or the longest sentence covered multiple offenses
            self.dataset['multi2_sent_life'] = copy.deepcopy(
                self.dataset['V0439'].replace([-9, -2, -1, 5, 6, np.nan], [np.nan, np.nan,np.nan, 0, 0, 0]))
            self.dataset['multi2_sent_life'].replace(
                to_replace=[1, 2, 3, 4], value=1, inplace=True)
            self.one_if_any.append(self.dataset['multi2_sent_life'].name)

            

    def make_one_if_any(self, newcol, test_val=1):


        # the binarised continuous columns and categoricl columns are included in the list
        to_look=self.one_if_any
        self.dataset['sum'] = self.dataset[to_look].sum(axis=1)

        #print('Check summed up to 1 only')
        #print(self.dataset[temp_name].value_counts(dropna=False))
        
        '''
        to_check = np.where(self.dataset[temp_name] > 1)
        check_df = self.dataset[[var for var in self.one_if_any]]
        final = check_df.iloc[to_check]
        final.to_csv('to_check.csv')
        #print(final)
        '''
        self.dataset[self.sent_th_name] = np.where(self.dataset['sum'] >= test_val, 1, 0)
        
        '''
        #print(self.dataset['sum'].value_counts(dropna=False))
        print(114)
        self.dataset.loc [self.dataset['sum']<1, self.sent_th_name]=0
        print(113)
        self.dataset.loc [self.dataset['sum']>=1, self.sent_th_name]=1
        '''
        print(self.dataset[self.sent_th_name].value_counts(dropna=False))
    
    @staticmethod
    def drop(transformer):
        
        to_drop=list(set(transformer.dataset.columns)-set(transformer.orig_cols))
        
        for var in transformer.shared_info:
            to_drop.append(var)
            
        to_drop = [col for col in to_drop if col not in transformer.to_keep]

        #print('Columns created and now being dropped', to_drop)
        #print('Saving to file')

        # drop the columns we've created along the way
        transformer.dataset.drop(labels=to_drop, inplace=True, axis=1)

    @staticmethod
    def update_config(transformer):
        '''Updates to config dictionary stored under the dataset processor when passed a dictionary containing new variables'''
        new_config=pd.DataFrame.from_dict(transformer.configdata,orient='index')
        #print(new_config)
        transformer.dp.config=pd.concat([transformer.dp.config,new_config],axis=0)
