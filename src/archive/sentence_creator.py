# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:42:16 2022

@author: siobh
"""

import copy as copy
import numpy as np
import operator
import pickle

class sentence_creator:
    
    """
    Called by the calc_target static method within the dataset_processor class .
    Called from the main_script, not in instantiation of the data_processor object
    Called after dataset pre processing and before encoding and scaling continuous variables 
    
    On instantiation, makes target column '1' for any rows where the sentence length is above the threshold 
    Provides updated subset with target variable column added and all informing columns removed
    
    In the Survey of US Prisoners, the sentence length is stored in different variables depending on the sentence type.
    
    All continuous variables are loaded from a dictionary in the file sentence_length_variables in the data/processing_config)
    Where a sentence is given as a range, there is a maximum and minimum length. If either the maximum or minimum number of years is
    above the threshold, the target variable is set to the positive class. 
    
    Categorical variables indicate a sentence of life of life+ are set to the positive class. 
    
    ...
    
    Attributes
    ----------
    subset: dataframe
        A subset of the 2016 Survey of US Prisoners. 
        This should contain columns that wil be used to calculate the sentence length
    
    th: int
        The threshold for binary classification. Sentence length in years. 
        
    target_name: str
        The name of the target variable to be returned. f'sentence_above_{self.th}yrs'
        
    original_cols: list of str
        The list of columns in subset at the point of instantiation
    
    on_if_any: list of str
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
    def __init__(self, subset, th):

        #take column list at instantiation to diff later
        self.orig_cols = subset.columns
        
        self.subset = subset
        self.th = th
        self.target_name = f'sentence_above_{self.th}yrs'

        #empty list to add columns to
        self.one_if_any = []

        #load the dictionaries of continuous variable sets
        self.load_sentence_length_variable_names()
        
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
        self.make_one_if_any(self.target_name)

        # store the cols that we have created and are deleting so they can be examined
        self.get_cols_created()
        #drop the cols_created as part of processing, apart from the target variable
        self.drop_non_target()

    def prep_cols(self, cols, fill_w_mean=[-9, -2, -1], fill_w_0=[-8, np.nan]):

        for col in cols:
            #print('Col being prepped', col)
            #print('Mean is',self.subset[col].mean())
            self.subset[col].replace(fill_w_mean, inplace=True)
            self.subset[col].replace(fill_w_0, 0, inplace=True)
    
    def load_sentence_length_variable_names(self,path=""):
        
        if path=="":
            path=r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/processing_config/sentence_length_variables'
        
        self.dmys=pickle.load( open( path, "rb" ) )

    def convert_to_years(self):
        # takes in a dictionary of dictionaries
        # outer dict the sentence type or grouping of the fields to be added
        # inner dict provide a time unit (d/m/y) as key, and a
        # column name as the value. Converts the days and months into year values
        # convert each month and day field to years and then sum to create new field
        denoms = {'d': 365, 'm': 12, 'y': 1}

        # preprocess these columns
        to_prep = []
        for sent_type, dmy in self.dmys.items():
            for unit, varname in dmy.items():
                to_prep.append(varname)

        self.prep_cols(to_prep)

        dmys_conv = {}

        self.converted_years = []

        for sent_type, dmy in self.dmys.items():
            # add a di
            dmys_conv[sent_type] = {}
            for unit, colname in dmy.items():
                # not skipping over years- instead divide it by 1
                # then can append everything to the list for summing

                # add y to indicate it's been converted into years
                newcol = colname+'y'

                # use np.floor because we want to round down to the nearest integer
                self.subset[newcol] = np.floor(
                    self.subset[colname]/denoms[unit])

                # add this to dictionary for summing
                dmys_conv[sent_type][unit] = newcol

            # name for converted years
            newyrs = sent_type+'c'
            #print('new yrs', newyrs)
            # sum the values
            # dmys_conv[sent_type][newyrs]=sum(dmys_conv[sent_type].values())

            # sum values add the converted value to the main subset
            #print(dmys_conv[sent_type].values())
            self.subset[newyrs] = sum([self.subset[yr]
                                      for yr in dmys_conv[sent_type].values()])

            # this will be picked up by binarise_continuous function
            self.converted_years.append(self.subset[newyrs])

    def binarise_continuous(self):

        #print('Checking if continuous columns are above threshold')
        # we don't need to prep the continuous variables as they have already been pre processed

        for v in self.converted_years:

            vname = v.name

            self.binarise_this_col(vname)
            #print('Now have this many cols',len(self.subset.columns))

            # get the most recent out name
            self.one_if_any.append(self.subset[self.out_name].name)

    def binarise_this_col(self, conv_col):
        # def binarise_this_col(self,conv_col,fill_w_mode=[-9,-2,-1],fill_w_0=[-8,np.nan])

        out_name = f'{conv_col}_above{self.th}'
        # print('********!')
        # print(self.subset[conv_col].value_counts(dropna=False))

        # then indicate if over or above threshold
        self.subset.loc[self.subset[conv_col] < self.th, out_name] = 0
        self.subset.loc[self.subset[conv_col] >= self.th, out_name] = 1

        # print(self.subset[out_name].value_counts(dropna=False))

        #print(f'Binarised {self.subset[out_name].name}')
        # update out_name
        self.out_name = out_name

    def binarise_categorical(self):
        #print('Binarising categorical columns- life or death sentences will be set above threshold')

        #print('Starting the flat categorical sentences')
        # we need to do the flat sentences, which contain categorical info
        # 6 the most common value within the subset that answered this question- going to be raplced with 0 anyway
        self.subset['single_sent_life'] = copy.deepcopy(
            self.subset['V0401'].replace(to_replace=[-9, -2, -1, 5, 6, np.nan], value=0))
        # we can then replace the values that indicate a life or a death sentence with 997 for adding later
        self.subset['single_sent_life'].replace(
            to_replace=[1, 2, 3, 4], value=1, inplace=True)
        self.one_if_any.append(self.subset['single_sent_life'].name)

        #print('Value count for categorical col 1')
        # print(self.subset['single_sent_life'].value_counts(dropna=False))

        # we can do the same for prisoners with multiple offences
        self.subset['multi_sent_life'] = copy.deepcopy(
            self.subset['V0412'].replace(to_replace=[-9, -2, -1, 5, 6, np.nan], value=0))
        self.subset['multi_sent_life'].replace(
            to_replace=[1, 2, 3, 4], value=1, inplace=True)
        self.one_if_any.append(self.subset['multi_sent_life'].name)
        
        #TODO Add V0439 (multi sentences, one sentence for multiple offenses or the longest sentence covered multiple offenses)
        self.subset['multi2_sent_life'] = copy.deepcopy(
            self.subset['V0439'].replace(to_replace=[-9, -2, -1, 5, 6, np.nan], value=0))
        self.subset['multi2_sent_life'].replace(
            to_replace=[1, 2, 3, 4], value=1, inplace=True)
        self.one_if_any.append(self.subset['multi2_sent_life'].name)

    def make_one_if_any(self, newcol, test_val=1):

        temp_name = 'sum'

        # the binarised continuous columns and categoricl columns are included in the list
        self.subset[temp_name] = self.subset[list(self.one_if_any)].sum(axis=1)

        #print('Check summed up to 1 only')
        #print(self.subset[temp_name].value_counts(dropna=False))

        to_check = np.where(self.subset[temp_name] > 1)
        check_df = self.subset[[var for var in self.one_if_any]]
        final = check_df.iloc[to_check]
        final.to_csv('to_check.csv')
        #print(final)

        self.subset[self.target_name] = np.where(self.subset[temp_name] >= test_val, 1, 0)

    def get_cols_created(self):
        self.cols_created = self.get_col_diff()

    def get_col_diff(self):
        return set(self.subset.columns)-set(self.orig_cols)

    def drop_non_target(self):

        to_drop = [col for col in self.get_col_diff() if col !=self.target_name]

        #print('Columns created and now being dropped', to_drop)
        #print('Saving to file')

        # drop the columns we've created along the way
        self.subset.drop(labels=to_drop, inplace=True, axis=1)
