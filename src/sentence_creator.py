# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:42:16 2022

@author: siobh
"""

import copy as copy
import numpy as np
import operator


class sentence_creator:
    
    #called by dataset processor before encoding and scaling to calculate sentence variables and add them to the dataset

    #takes either the entire prisoner dataset or one of the subsets and a sentence threshold
    #determines if sentence is above that threshold
    #life is included as upper threshold (i.e., np.inf)
    #where there are indeterminate sentences, if the min or max is above the threshold, that instance
    #will be counted as above the threshold
                
    #V0402:FLAT SINGLE the number of years for prisoners being sentence for a single crime with a flat sentence an a specific number of years
    #V0413:FLAT MULTI the length of the max sentence if it's a flat sentence for prisoners being sentenced for multiple crimes
    
    #V0401: the type of flat sentence for inmates being sentenced for a single crime with a flat sentence- indicates life and death
    #V0412: the type of flat sentence for inmates being sentenced for a multiple crimes with a flat sentence-- indicates life and death
    #1 (Life), 2 (Life Plus Additional Years), 3 (Life Without Parole), 4 (Death), -9 (Missed),-2 (Refusal),-1 (Don't know)-8 (Skipped)
    
    #V0405 INDETERMINATE SINGLE contains the minimum number of years for a single sentence if it's a range
    #V0406 INDETERMINATE SINGLE contains the maximum number of years for a single sentence if it's a range
    
    #V0417 INDETERMINATE MULTI contains minimum number of years for the longest sentence they are being sentenced for
    #V0418 INDETERMINATE MULTI contains the max number of years for the longest sentence they are being sentenced for
    
    def __init__(self,subset,th):
        
        #need this list so we can diff and delete relevant cols
        self.orig_cols=subset.columns
        self.subset=subset
        self.th=th
        self.target_name=f'sentence_above_{self.th}yrs'
        self.cont_sent_length_vars=['V0406','V0405','V0417','V0418']
        self.cat_sent_length_vars=['V0401','V0412']
        self.one_if_any=[]
        
        self.collapse_ymd()

        self.binarise_continuous()
        self.binarise_categorical()
        #print('one if any list')
        #print(self.one_if_any)
        print('vc for one if any list')
        #print(type(thing) for thing in self.one_if_any)
        #debugging- get value counts
        print(self.one_if_any)
        for col in self.one_if_any:
            print(self.subset[col].value_counts(dropna=False))
            
        self.make_target_one_if_any()
        
        #store the cols that we have created and are deleting so they can be examined
        self.get_cols_created()
        
        self.drop_non_target()
        
        print('Name of target col is', self.target_name)

    def prep_cols(self,cols,fill_w_mean=[-9,-2,-1],fill_w_0=[-8,np.nan]):
        for col in cols:
            print('Col being prepped', col)
            #print('Mean is',self.subset[col].mean())
            self.subset[col].replace(fill_w_mean,inplace=True)
            self.subset[col].replace(fill_w_0,0,inplace=True)
    
    def collapse_ymd(self,fill_w_mode=[-9,-2,-1],fill_w_0=[-8,np.nan]):
        print('Collapsing year, months and days')
        
        self.prep_cols(['V0402','V0413','V0403','V0404','V0414','V0415'])
        
        #single offence single/flat sentence (years: V0402, months: V0403,days: V0404)
        self.subset['single_offence_length']=(self.subset['V0402']*365)+(self.subset['V0403']*30.5)+(self.subset['V0404'])

        #multiple offences (years: V0413, months: V0414, days: V0415)
        self.subset['multiple_offence_length']=(self.subset['V0413']*365)+(self.subset['V0414']*30.5)+(self.subset['V0415'])
        #test['multiple_offence_length']=(test['V0413']*365)+(test['V0414']*30.5)+(test['V0415'])
        
        #combine both of the above into a continuous value
        self.subset['combined_sentence_length']=(self.subset['single_offence_length']+self.subset['multiple_offence_length'])

        #convert to year because of change in how we're doing it
        self.subset['combined_sentence_length']=round(self.subset['combined_sentence_length']/365,2)
        
        self.cont_sent_length_vars.append('combined_sentence_length')
        
    def binarise_continuous(self):
        print('Checking if continuous columns are above threshold')
 
        #if v in index only looks at calcs relevant to this subset
        self.prep_cols(self.cont_sent_length_vars)
        #cont_sent_length_vars=[v for v in self.cont_sent_length_vars if v in self.subset.index]
        print('Cont sent length vars',self.cont_sent_length_vars)
        for v in self.cont_sent_length_vars:

            print(v)
            print('Th is',self.th)
            #will assign columns to original dataframe
            self.binarise_this_col(v)
            #print('Now have this many cols',len(self.subset.columns))
        
            #get the most recent out name
            #debugging- print value counts
            print('Value count for binarise_continuous')
            print(self.subset[self.out_name].value_counts(dropna=False))
            self.one_if_any.append(self.subset[self.out_name].name)

    def binarise_this_col(self,orig_col,fill_w_mode=[-9,-2,-1],fill_w_0=[-8,np.nan]):

        out_name=f'{orig_col}_above{self.th}'

        #then indicate if over or above threshold
        self.subset.loc[self.subset[orig_col]<self.th,out_name]=0
        self.subset.loc[self.subset[orig_col]>=self.th,out_name]=1

        print(f'Binarised {self.subset[out_name].name}')
        #update out_name
        self.out_name=out_name
    
    def binarise_categorical(self):
        print('Binarising categorical columns- life or death sentences will be set above threshold')
        
        #print('Starting the flat categorical sentences')
        #we need to do the flat sentences, which contain categorical info
        #6 the most common value within the subset that answered this question- going to be raplced with 0 anyway
        self.subset['single_sent_life']=copy.deepcopy(self.subset['V0401'].replace(to_replace=[-9,-2,-1,5,6,np.nan],value=0))
        #we can then replace the values that indicate a life or a death sentence with 997 for adding later
        self.subset['single_sent_life'].replace(to_replace=[1,2,3,4],value=1,inplace=True)
        self.one_if_any.append(self.subset['single_sent_life'].name)
        
        print('Value count for categorical col 1')
        print(self.subset['single_sent_life'].value_counts(dropna=False))
        
        #we can do the same for prisoners with multiple offences
        self.subset['multi_sent_life']=copy.deepcopy(self.subset['V0412'].replace(to_replace=[-9,-2,-1,5,6,np.nan],value=0))
        self.subset['multi_sent_life'].replace(to_replace=[1,2,3,4],value=1,inplace=True)
        self.one_if_any.append(self.subset['multi_sent_life'].name)

        print('Value count for categorical col 2')
        print(self.subset['multi_sent_life'].value_counts(dropna=False))
        
    def make_target_one_if_any(self,test_val=1,new_val=1,other_val=0):
        
        #df['hasimage'] = np.where(df['photos']!= '[]', True, False)
        
        #print('Target one if any==1',self.one_if_any)

        temp_name='sum'
    
        #df['sum'] = df[list(df.columns)].sum(axis=1).
        #slist=[self.subset[col] for col in self.one_if_any]
        self.subset[temp_name] = self.subset[list(self.one_if_any)].sum(axis=1)
        
        self.subset[self.target_name] = np.where(self.subset[temp_name]==test_val, new_val,other_val)
        
        #self.subset.loc[]
        #df.loc[df['column name'] condition, 'new column name'] = 'value if condition is met'
        
        print('vc after creation')
        print(self.subset[self.target_name].value_counts(dropna=False))

    def get_cols_created(self):
        self.cols_created=self.get_col_diff()
    
    def get_col_diff(self):
        return set(self.subset.columns)-set(self.orig_cols)                                  
        
    def drop_non_target(self):

        to_drop=[col for col in self.get_col_diff() if col != self.target_name]
        
        print('Columns created and now being dropped', to_drop)
        #print('Saving to file')
    
        #drop the columns we've created along the way
        self.subset.drop(labels=to_drop,inplace=True,axis=1)

