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
from sklearn.preprocessing import OrdinalEncoder
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
    
    def __init__(self,dataset, colname, config,enc=1,scale=1,impute=1):
        self.colname = colname
        self.enc=enc
        self.scale=scale
        self.impute=impute
        #print(colname)
        self.col=dataset[colname]
        #the dataset_in (unprocessed is what's passed in)
        self.dataset = dataset
        self.config=config
        #print('Config type at var processor',type(config))
        self.enc_scale=config['enc_scale'][colname]
        #print(self.treatment)
        #empty nan list to populate
        self.dkref=[]
        self.missing=[]
        
        self.output_categories=[]        

        self.get_dkref()
        self.get_missing()
        self.process()
        
        #run some checks on the encoding and raise warnings if needed
        if self.enc_scale=="one_hot":
            if self.enc==1:
                #self.check_encoded_values()
                self.check_output()
                self.check_if_binary(self.output_col,self.colname)

    
    #TODO: improve this so that literal_eval is not used as it's bad practice
    def get_dkref(self):
        if pd.isnull(self.config['dk_ref'][self.colname]):
            self.dk_ref=[]
        else:
            try:
                self.dkref=list(literal_eval(self.config['dk_ref'][self.colname]))
            except:
                single_dkref=literal_eval(self.config['dk_ref'][self.colname])
                self.dkref.append(single_dkref)
                #self.dkref=single_dkref

    def get_missing(self):
        if pd.isnull(self.config['dk_ref'][self.colname]):
            self.missing=[]
        
        else:
            try:
                self.missing=list(literal_eval(self.config['missing'][self.colname]))
            except:
                single_missing=literal_eval(self.config['missing'][self.colname])
                self.missing.append(single_missing)

    def process(self):
        if self.enc_scale =='one_hot':
            if self.colname=='V0772':
                self.prep_nans_states()
            
            else:
                self.prep_nans_cat()
            
            if self.enc==1:
                self.one_hot_encode()
            
            else: 
                self.output_col=self.col_nans_processed
            
        elif self.enc_scale=='scale':
            self.process_cont_wnans()
        
        elif self.enc_scale=='ordinal':
            self.prep_nans_cat()
            
            if self.enc==1:
                self.ordinal_encode()
            else: 
                self.output_col=self.col_nans_processed
        
        elif self.enc_scale=="none":
            self.output_col=self.dataset[self.colname]
        
        else:
            print(f"Error, enc_scale value of {self.enc_scale}")
               
    def min_max_scaling(self):
        series=self.output_col
        return (series - series.min()) / (series.max() - series.min())

    #TODO: make this a static class
    def process_cont_wnans(self):
        
        #scaling or no scaling, we need to replace dk_refs etc
        
        #replace missing values defined in config with zero
        self.col_nans_processed=copy.deepcopy(self.col.replace(self.missing,0))

        #instead of -8, skips sometimes have np.nan so we want to make these zeroes
        self.col_nans_processed=self.col_nans_processed.replace(np.nan,0)
        
        if self.impute==1:
            if self.colname=='ctrl_count':
                #print(self.col_nans_processed.value_counts(dropna=False))
                self.col_nans_processed=self.col_nans_processed.replace(self.dkref,0)
            else:
                #identify the don't know and refused values for this column and replace them with meaningful values
                self.col_nans_processed=self.col_nans_processed.replace(self.dkref, round(self.col_nans_processed.mean()))
        else:
            #otherwise make these np.nan to identify them as missing values
            self.col_nans_processed=self.col_nans_processed.replace(self.dkref, np.nan)
            
        self.output_col=copy.deepcopy(self.col_nans_processed)
        
        #it's continous so not splitting column- we can just keep the same column name
        self.output_col.name=self.colname
        
        if self.scale==1:
            self.output_col=self.min_max_scaling()
           

    def prep_nans_states(self):
        #separate function to prep_nans_cat because need to have -1 and -2 as strings 
        #because this variable stores all values as strings
        #replace DK/REF with most common state if impute=1, or put as nan if impute=0

        if self.impute==1:
            self.col_nans_processed=self.col.replace(['-1','-2',np.nan],[self.col.mode(),self.col.mode(),self.col.mode()])
        if self.impute==0:
            self.col_nans_processed=self.col.replace(['-1','-2',np.nan],[np.nan,np.nan,'-8'])
            
            
    def prep_nans_cat(self):
        """
        ####1) Replacing np.nan
        #If skips are allowed (-8,-99 Excluded from nan list), replace np.nan with -8 (representing a skip)
        #Need to have a value because it's not a missing value, it's a skip. They will get their own category when encoded
        #Empty strings are converted to np.nan so will be included in this

        ####2) Replace missing values (dk_ref values defined in config file
        #then replace the values given as nans for this variable with as na/empty value
        #they will then be given their own category by the encoder
        #if -8 is nan (i.e. there should be no skip possible on the question) it will be in the nan list

        """
        ###1) Replacing missing, blank and np.nan values
        #for all the ones that are skipped or empty data put in a -8
        #They will get their own category when encoded
        #Empty strings are converted to np.nan so will be included in this
        #Where a column has been derived and added to config, nans should still be replaced with -8
        #Unless the 'missing' values have been set, they will remain
        self.col_nans_processed=self.col.replace(np.nan,-8)
        self.col_nans_processed.replace(self.missing,-8,inplace=True)

      
        #for dk and ref value as defined in the file- these are missing values
        #if impute=1, replace these with a meaningful value
        #if impute=0, replace with np.nan (will be passed to scikit learn)
        #pandas is asking for replace list to be the same length as list of values to replace
        if self.impute==1:
            replacements=[self.col.mode() for val in self.dkref]
            self.col_nans_processed.replace(self.dkref,replacements,inplace=True)
        
        if self.impute==0:
            replacements=[np.nan for val in self.dkref]
            self.col_nans_processed.replace(self.dkref,replacements,inplace=True)
            
        
        #self.output_col=copy.deepcopy(self.col_nans_processed)
    
    def ordinal_encode(self):
        
        ord_encoder = OrdinalEncoder()
        #get the index
        index=self.col_nans_processed.index
        self.output_col=self.col_nans_processed.to_numpy()
        self.output_col=self.output_col.reshape(-1, 1)

        self.output_col = ord_encoder.fit_transform(self.output_col)

        self.output_col=self.output_col.reshape(-1, 1)
        #back to df for joining
        self.output_col=pd.DataFrame(self.output_col)
        #put name back on to match


        

        #put index back on to match
        self.output_col.index=index
        self.output_col.rename(columns={0:self.colname},inplace=True)
        #print(self.output_col.value_counts(dropna=False))


 


    def one_hot_encode(self):
        """One hot encode categorical data using sklearn one hot encoder.
       To avoid 'dummy variable trap', the encoder will drop -8"""
        
        #take the index to be appended back later
        #this is needed for joining dataframes
        orig_index=self.col_nans_processed.index
        
        #have to reshape column to encoder
        #df.values array gives an np array that can be reshaped
        self.prepped_col_reshaped=self.col_nans_processed.values.reshape(-1, 1)
        

        self.enc=OneHotEncoder(sparse=False,drop='if_binary')
 
        
        #call fit, but not fit transform
        #categories can be extracted
        self.enc_fitted = self.enc.fit(self.prepped_col_reshaped)
        
        #save onehot categories
        #TODO: remove -8

        self.found_categories=self.enc_fitted.categories_

            
        #need to do transform x to return the array from the encoder object
        self.onehot_array=self.enc_fitted.transform(self.prepped_col_reshaped)
 
        #create readable output categories
        output_categories=[]
        for item in self.enc.get_feature_names_out():
                output_categories.append(f"{self.colname}-{str(item)}")
     
        self.output_categories=output_categories
        #change onehot array to pandas series and reapply orig_index for join
        self.output_col=pd.DataFrame(self.onehot_array, columns=self.output_categories, index=orig_index)
        
        if len(self.output_col.columns) > 1:
            
            #print(self.output_col.columns)
            #print('More than one output col',f"{self.colname}-x0_-8.0")
            
            if 'x0_-8.0' in self.enc.get_feature_names_out():
            #if 'f"{self.colname}-x0_-8.0' in self.output_col.columns:
                
                self.output_col.drop(f"{self.colname}-x0_-8.0",axis=1,inplace=True)
                #print('Dropped a column')
                
            #print(self.output_col.columns)
            
    def check_encoded_values(self):
        """Checks that the count in the encoded column corresponding to each category in the original column
        is greater than or equal to the value count in the original column. Because of don't know, refuse
        we can't do an exact check"""

        if self.colname != 'V0772':
            #the value counts in the original dataset
            val_counts=self.dataset[self.colname].value_counts()
            
            #don't want to compare values for dkref and missing values
            #as these will have been replaced by the nan processing
            #in cases where the original data never had dkref or missing vals, to_drop will be 0
            to_drop=set(val_counts.index).intersection(self.dkref+self.missing)

            #to_drop=set(self.dkref+self.missing).intersection(val_counts.index)
            val_counts.drop(to_drop,inplace=True)
            #a dict is easier to loop through
            val_counts=val_counts.to_dict()
                        
            for val,count in val_counts.items():
                #dervice colname
                if self.colname != 'V0772':
                    enc_name=f"{self.colname}-{str(val)}"
                elif self.colname=='ctrl_count':
                    enc_name=f"{self.colname}-{str(val)}"
                else:
                    enc_name=f"{self.colname}-{str(val)}"
                
                newcol=self.output_col[enc_name]
                
                if sum(newcol)>=count:
                    pass
                else:
                    print(self.colname,val,'encoded column has fewer values')

    def check_output(self):
        #the columns should sum to 1
        if self.enc_scale=='one_hot':
            newcols= self.encoded_colnames(self.output_col,self.colname)
           # if self.check_rows_sum_1(self.output_col,newcols)==False:
                #print(self.colname,'encoded columns do not sum to 1')
            
    @staticmethod
    def encoded_colnames(dataset,orig_colname):
        return [colname for colname in dataset.columns if orig_colname in colname]

    @staticmethod
    def check_rows_sum_1(dataset,colnames):
        check=dataset[colnames].sum(axis=1)
        return set(check.values) == {1}

    @staticmethod
    def reduce_if_binary(dataset,orig_colname):
        #if dataset_processor.check_if_binary(dataset,orig_colname):
            #then we remove one of the values
        if var_processor.check_if_binary(dataset,orig_colname):
            print(orig_colname,'Binary')
    
    @staticmethod
    def check_if_binary(dataset,orig_colname):
        #dataset should be the dataset_out following the encoding
        to_check=var_processor.encoded_colnames(dataset,orig_colname)
        to_check= [colname for colname in to_check if '-8' not in  colname]
        
        if len(to_check)<=2:
            return  sum(dataset[to_check].sum(axis=0))==len(dataset.index)
        else:
            return False