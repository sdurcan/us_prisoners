# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:44:48 2022

@author: siobh
"""
###
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from ast import literal_eval
import copy
import pandas as pd
import warnings

#take a pandas series column and a config file defining processing steps and processes it
class var_processor:
    
    """
    A class initiated by the dataset_processor to process a variable
    Provides an output_col which is appended to the output_dataframe from the dataset_processor that is
    then passed to learning model

    ...

    ####Attributes
    ----------
    dataset : pandas dataframe
        the dataset that is being processed by the dataset_processor
    colname : str
        the name of the column being processed
    col : pandas series
        the pandas series being processed
    config: pandas dataframe
        a dataframe that lists all variables as the index and indicates
        how they should be processed
    nans : list
        derived from the config file, a list of values to treat as nans
    col_nans_processed
        captures the state of the column after -8 replaced with 0
        populated by the process_cont_wnans method
    output_col
        captures the states of the column after np.nan values replaced with
        most common value, then is updated with standardised and normalised values
        
        for categorical data, the encoded array

    prepped_col_reshaped: stores a numpy array of df values to be passed to the encoder
    enc_fitted: stores encoder object after fitting to the prepped_col_reshaped array
    found_categories: stores a list of n categories found by encoder
    onehot_array: array with one column the n categories, populated with 1 or 0
    output_categories: readable categories created by appending colname to category number
    
    ####Methods
    -------
    get_nans
        Reads the nan values for this variable from the config dataframe and 
        updates the nans attributes
    
    process
        Determines how the variable should be treated from the config file
        and calls the relevant method on the data
        Categorical varaibles should be coded as 'one_hot'
        Continuous varaibles that use (usuallly negatvive) integers to 
        indicate set values should be 'cont_w_nans'

    min_max_scaling
        utility function used in normalising
    
    process_cont_wnans
        1) Replace skips (-8)
        Analysis indicated that -8 is used instead of blank values or nan values
        If -8 is excluded from the nan list in the config file, this means 'skips are allowed'
        -8 might be in the nan list when it's expected that a question would not be skipped in a subset
        self.co.nans_processed captures the state of the column after -8 replaced with 0
        
        2) Replace np.nan values
        The other nans should have information, so we should be replacing those values intelligently
        The dataset does not appear to contain np.nan values
        An warning will be raised if one is found
        
        3) Standardise data
        If self.stand==1, the column is standardised
    
        4) Normalise data
        If self.norm==1, then column is normalised
    
    prep_nans_one_hot
        ###1) Replace np.nan
        #If skips are allowed (-8,-99 EXcluded from nan list), replace np.nan with -8 (representing a skip)
        #They will get their own category when encoded
        #Empty strings are converted to np.nan so will be included in this

        ###2) Replace nan values defined in config file
        #then replace the values given as nans for this variable with as na/empty value
        #they will then be given their own category by the encoder
        #if -8 is nan (i.e. there should be no skip possible on the question) it will be in the nan list
    
    one_hot_encode
        One hot encode categorical data using sklearn one hot encoder.
        As nans are replaced with new values, can do this with all values
        and nans will get own category
    
    """

    #config- a dataframe (should be a dict?)
    #colname- string
    #dataset- pandas dataframe
    
    def __init__(self,dataset, colname, config,stand=0,norm=0,target='sentence_length'):
        self.stand=stand
        self.target=target
        self.norm=norm
        self.colname = colname
        #print(colname)
        self.col=dataset[colname]
        self.dataset = dataset
        self.config=config
        self.treatment=config['treatment_violent_lr'][colname]
        #empty nan list to populate
        self.nans=[]
        self.enc=OneHotEncoder(sparse=False)
        self.output_categories=[]        

        self.get_nans()
        self.process()

        
    def get_nans(self):
        try:
            self.nans=list(literal_eval(self.config['nans'][self.colname]))
        except:
            single_nan=literal_eval(self.config['nans'][self.colname])
            self.nans.append(single_nan)

    def process(self):
        #print('Treatment is {}'.format(self.treatment))
        if self.treatment=='one_hot':
            self.prep_nans_one_hot()
            self.one_hot_encode()
            
        elif self.treatment=='cont_wnans':
            self.process_cont_wnans()
        
        else:
            print(f"Error, treatment {self.treatment}")
               
    def min_max_scaling(self):
        series=self.output_col
        return (series - series.min()) / (series.max() - series.min())
  
    def yprocess_cont_wnans(self):
 
        ####1) Replacing skips (-8)
        #missing values to be replaced with a 0 if skip is allowable
        #we know skip is allowable if -8 is not in nans
        #for example, if we expect that a question is required in the subset under analysis
        if '-8' not in self.nans:
            self.col_nans_processed=self.col.replace(np.nan,0)
        else:
            #print(self.colname)
            self.col_nans_processed=self.col
        
        ####2) Replacing np.nan values
        #nans that are not skips should be replaced intelligently
        #replaced with most frequent value        
        #if skips are not allowed, then -8 will be on the nan list and will also be replaced
        if np.nan in self.col_nans_processed.values:
            warnings.warn('np.nan values found and replaced with most common value')
            mc=self.col_nans_processed.value_counts(dropna=False).idxmax()
            self.output_col=self.col_nans_processed.replace(self.nans, mc)
        else:
            self.output_col=self.col_nans_processed

        #update name of pandas series so that the variable name propogates through to output df    
        self.output_col.name=self.colname
        
        ####3) Replace nans listed in config file with a 0
        #can't be left as np.nan because will cause errors
        self.output_col.replace(self.nans, 0,inplace=True)
        
        ####4) Standardise values
        #need to check if this is one of the target variables before standardising
        if self.target=='sentence_length':
            if self.colname not in ['V0402','V0403','V0404','V0413','V0414','V0415']:

                
                if self.stand==1:
                    self.output_col=(self.output_col-self.output_col.mean())/self.output_col.std()
                else:
                    self.output_col=self.col_nans_processed
                
                ###4) Normalise values
                if self.norm==1:
                    self.output_col = self.min_max_scaling()
            


    def process_cont_wnans(self):
        
        #-8 means the logic skipped it, so it should be replaced with zero
        #the other nans should have information, so we should be replacing those values intelligently
        self.col_nans_processed=copy.deepcopy(self.col.replace(-8,0))
        #print('During nan processing')
        #print(sorted(self.col_nans_processed.index)[0:10])
        #instead of -8, skips sometimes have np.nan so we want to make these zeroes
        self.col_nans_processed=self.col_nans_processed.replace(np.nan,0)
        #print(sorted(self.col_nans_processed.index)[0:10])
        
        #then we need to identify the nan values for this column and replace them with np.nan
        #this allows the pandas fill na class to be used
        self.col_nans_processed=self.col_nans_processed.replace(self.nans,np.nan)
        #print(sorted(self.col_nans_processed.index)[0:10])
        
        #then use the pandas interpolate function
        #issue with this is that it produces negative numbers and also produces decimal numbers
        #only positive intergers are relevant
        #self.col_nans_processed=self.col_nans_processed.interpolate(method='polynomial',order=2)
        
        #try replacing with mean instead
        self.col_nans_processed=self.col_nans_processed.replace(np.nan, round(self.col_nans_processed.mean()))
        #print(sorted(self.col_nans_processed.index)[0:10])
            
        self.output_col=copy.deepcopy(self.col_nans_processed)
        #print(sorted(self.output_col.index)[0:10])
        
        #it's continous so not splitting column- we can just keep the same column name
        self.output_col.name=self.colname
    
    def prep_nans_one_hot(self):
        
        """
        ####1) Replacing np.nan
        #If skips are allowed (-8,-99 EXcluded from nan list), replace np.nan with -8 (representing a skip)
        #They will get their own category when encoded
        #Empty strings are converted to np.nan so will be included in this

        ####2) Replace nan values defined in config file
        #then replace the values given as nans for this variable with as na/empty value
        #they will then be given their own category by the encoder
        #if -8 is nan (i.e. there should be no skip possible on the question) it will be in the nan list


        """
        ###1) Replacing missing and np.nan values
        #If skips are allowed (-8,-99 EXcluded from nan list), replace np.nan with -8 (representing a skip)
        #They will get their own category when encoded
        #Empty strings are converted to np.nan so will be included in this
        
        
        if '-8' not in str(self.nans):
            self.col_nans_processed=self.col.replace(np.nan,-8)
        else:
            self.col_nans_processed=self.col
        
        if '99' not in str(self.nans):
            self.col_nans_processed.replace(99,-8,inplace=True)
        else:
            self.col_nans_processed=self.col
            
        '''
        values=self.col_nans_processed.values

        ###2) Check for strings
        #the encoder needs the values to be the same datatype
        #this checks that 
        #should only happen for states V0772
        #replace with string to match other input
        if any(isinstance(v,str) for v in values):
            print([v for v in values if type(v) != str])
            self.col_nans_processed.replace(to_replace, 'nan',inplace=True)
            print('any', self.colname)
         else:
        '''
        
        ####3) Replace nan values defined in config file
        #then replace the values given as nans for this variable with as na/empty value
        #they will then be given their own category by the encoder
        #if -8 is nan (i.e. there should be no skip possible on the question) it will be in the nan list
        
        #CHANGE APPROACH: nan values all goe to -8
            
        self.col_nans_processed.replace(self.nans, -8,inplace=True)

    
    def one_hot_encode(self):
        """One hot encode categorical data using sklearn one hot encoder.
        As nans are replaced with new values, can do this with all values
        and nans will get own category"""
        

        #take the index to be appended back later
        #this is needed for joining dataframes
        orig_index=self.col_nans_processed.index
        
        #have to reshape column to encoder
        #df.values array gives an np array that can be reshaped
        self.prepped_col_reshaped=self.col_nans_processed.values.reshape(-1, 1)
        
        #call fit, but not fit transform
        #categories can be extracted
        self.enc_fitted = self.enc.fit(self.prepped_col_reshaped)
        
        #save onehot categories
        self.found_categories=self.enc_fitted.categories_
        #self.found_category_count=len(list(self.found_categories))

        #need to do transform x to return the array from the encoder object
        self.onehot_array=self.enc_fitted.transform(self.prepped_col_reshaped)
        
        #create readable output categories
        output_categories=[]
        for category in self.found_categories:
            for item in category:
                output_categories.append(f"{self.colname}-{str(item)}")
        
        self.output_categories=output_categories
        #change onehot array to pandas series and reapply orig_index for join
        self.output_col=pd.DataFrame(self.onehot_array, columns=self.output_categories, index=orig_index)
