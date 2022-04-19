# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:42:16 2022

@author: siobh
"""

import copy as copy
import numpy as np

class sentence_creator:
    
    #called by dataset processor before encoding and scaling to calculate sentence variables and add them to the dataset
    
    def __init__(self,subset,subset_num=3,bi=True,th=10,multi_cat=False, auto_cats=0, bin_edges=[],continuous=False):
        #user can specify if they want to binarise the sentence length (default threshold 10 years)
        #specify if they want multiple categories- requires number of categories (auto_cats) to 
        #be generated using qcut or a list of bin edges (pandas cut) to be specified
        #continuous can be used on subset 1 to calculate #of days of sentence
        self.subset=subset
        self.subset_num=subset_num
        self.bi=bi
        self.th=th
        self.multi_cat=multi_cat
        self.auto_cats=auto_cats
        self.bin_edges=bin_edges
        
        if self.bi==True:
            self.binarise_sentence()
            
    def binarise_col(self,orig_col,fill_w_mean=[-9,-2,-1],fill_w_0=[-8,np.nan]):
    #used in binarise sentence
     
        out_name=f'{orig_col}_above{self.th}'
        #subset=copy.deepcopy(subset_in)
        
        self.subset[out_name]=copy.deepcopy(self.subset[orig_col]).replace(to_replace=fill_w_mean,value=self.subset[orig_col].mean())
        self.subset[out_name].replace(to_replace=[-8,np.nan],value=0,inplace=True)
         
        #df[“column_name”] = np.where(df[“column_name”]==”some_value”, value_if_true, value_if_false)
        #then indicate if over or above threshold
        self.subset.loc[self.subset[out_name]<self.th,out_name]=0
        self.subset.loc[self.subset[out_name]>=self.th,out_name]=1
        #subset.loc[prisoners[f'single_range_min_{th}ys']>=th,f'single_range_min_{th}ys']=1
        #print(f'Binarised {self.subset[out_name].name}')
        #update out_name
        self.out_name=out_name

    
    def binarise_sentence(self):
    #takes either the entire prisoner dataset or one of the subsets and a sentence threshold
    #need to specify which subset so that the right sentence length variables are used
    #determines if sentence is above that threshold
    #life is included as upper threshold (i.e., np.inf)
    #where there are indeterminate sentences, if the min or max is above the threshold, that instance
    #will be counted as above the threshold
        subset_num=self.subset_num
        #check which columns to look at
        
        all_sent_length_vars=['V0402','V0413','V0401','V0412','V0405','V0406','V0417','V0418']
        
        if subset_num==3:
            sent_length_vars=['V0402','V0413','V0401','V0412','V0405','V0406','V0417','V0418']
        elif subset_num==2:
            sent_length_vars=['V0402','V0413','V0401','V0412']
        elif subset_num==1:
            sent_length_vars=['V0402','V0413']
            
        #subset3 will contain all: ['V0402','V0405','V0406','V0413','V0417','V0418']
        #subset2 will not contain V0405,V0406,V0417,V0418
        #subset1 will not contain V0401, V0402, V0405,V0406,V0417,V0418
        
        #V0402:FLAT SINGLE the number of years for prisoners being sentence for a single crime with a flat sentence an a specific number of years
        #V0413:FLAT MULTI the length of the max sentence if it's a flat sentence for prisoners being sentenced for multiple crimes
        
        #V0401: the type of flat sentence for inmates being sentenced for a single crime with a flat sentence- indicates life and death
        #V0412: the type of flat sentence for inmates being sentenced for a multiple crimes with a flat sentence-- indicates life and death
        #1 (Life), 2 (Life Plus Additional Years), 3 (Life Without Parole), 4 (Death), -9 (Missed),-2 (Refusal),-1 (Don't know)-8 (Skipped)
        
        #V0405 INDETERMINATE SINGLE contains the minimum number of years for a single sentence if it's a range
        #V0406 INDETERMINATE SINGLE contains the maximum number of years for a single sentence if it's a range
        
        #V0417 INDETERMINATE MULTI contains minimum number of years for the longest sentence they are being sentenced for
        #V0418 INDETERMINATE MULTI contains the max number of years for the longest sentence they are being sentenced for
        
        #print('# columns starting with',len(self.subset.columns))
        to_sum=[]
        for v in sent_length_vars:
            #print(v)
            #print('Th is',self.th)
            #will assign columns to original dataframe
            self.binarise_col(v)
            #print('Now have this many cols',len(self.subset.columns))
        
            #subset[f'{v}_above_{th}']=new
            #get the most recent out name
            to_sum.append(self.subset[self.out_name])
        
        #sum the columns
        count=0
        for col in to_sum:
            if count==0:
                self.subset['above_thr_exc_life']=col
            else:
                self.subset['above_thr_exc_life']=self.subset['above_thr_exc_life']+col
            count=+1
        
        #print('Starting the flat categorical sentences')
        #we need to do the flat sentences, which contain categorical info
        #6 the most common value within the subset that answered this question- going to be raplced with 0 anyway
        self.subset['single_sent_life']=copy.deepcopy(self.subset['V0401'].replace(to_replace=[-9,-2,-1,5,6,np.nan],value=0))
        #we can then replace the values that indicate a life or a death sentence with 997 for adding later
        self.subset['single_sent_life'].replace(to_replace=[1,2,3,4],value=1,inplace=True)
        to_sum.append(self.subset['single_sent_life'])
        
        #we can do the same for prisoners with multiple offences
        self.subset['multi_sent_life']=copy.deepcopy(self.subset['V0412'].replace(to_replace=[-9,-2,-1,5,6,np.nan],value=0))
        self.subset['multi_sent_life'].replace(to_replace=[1,2,3,4],value=1,inplace=True)
        to_sum.append(self.subset['multi_sent_life'])
        
        count=0
        for col in to_sum:
            #print(type(col))
            #print(col.name)
            if count==0:
                self.subset['above_thr_inc_life']=col
                #print('Aftrt 124, this many cols',len(self.subset))
            else:
                self.subset['above_thr_inc_life']=self.subset['above_thr_inc_life']+col
            count=+1
        
        to_drop=[col.name for col in to_sum]
        
        #drop the columns we've created along the way
        self.subset.drop(labels=to_drop,inplace=True,axis=1)
        #and the sentence length variables- we don't want to keep these
        self.subset.drop(labels=sent_length_vars,inplace=True,axis=1)
        
        self.cols_created=['above_thr_exc_life','above_thr_inc_life']
        
        print('Columns created: above_thr_exc_life,above_thr_inc_life')
        print(f'Columns deleted: {all_sent_length_vars}')
