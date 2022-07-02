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
        
        #only the ones we want to binarise
        self.cont_sent_length_vars=['V0406','V0405','V0417','V0418']
        #unused
        self.cat_sent_length_vars=['V0401','V0412']
        
        #the columns that are binarised get added here
        self.one_if_any=[]
        
        self.convert_to_years()
        
        self.binarise_continuous()
        self.binarise_categorical()
        #print('one if any list')
        #print(self.one_if_any)
        #print('vc for one if any list')
        #print(type(thing) for thing in self.one_if_any)
        #debugging- get value counts
        #print(self.one_if_any)
        #for col in self.one_if_any:
            #print(self.subset[col].value_counts(dropna=False))
        
        #looks at all columns that have been binarised and makes target col 1 if any are 1
        self.make_one_if_any(self.target_name)
        
        #store the cols that we have created and are deleting so they can be examined
        self.get_cols_created()
        print(self.get_cols_created())
        
        self.drop_non_target()
        
        print('Name of target col is', self.target_name)

    def prep_cols(self,cols,ctype='cont',fill_w_mean=[-9,-2,-1],fill_w_0=[-8,np.nan]):

        for col in cols:
            #print('Col being prepped', col)
            #print('Mean is',self.subset[col].mean())
            self.subset[col].replace(fill_w_mean,inplace=True)
            self.subset[col].replace(fill_w_0,0,inplace=True)

    def convert_to_years(self):
        #takes in a dictionary of dictionaries 
        #outer dict the sentence type or grouping of the fields to be added
        #inner dict provide a time unit (d/m/y) as key, and a 
        #column name as the value. Converts the days and months into year values
        #convert each month and day field to years and then sum to create new field
        denoms={'d':365,'m':12,'y':1}

        #VO402- years sentenced to prison, single offence
        #V0403- months sentenced to prison, single offence
        #V0404- days sentenced to prison, single offence
        single_flat={'d':'V0404','m':'V0403','y':'V0402'}
        
        #V0405- min years sentenced to prison, single offence
        #V0407- min months sentenced to prison, single offence
        #V0409- min days sentenced to prison, single offence
        single_range_min={'d':'V0409','m':'V0407','y':'V0405'}
        
        #V0406- max years sentenced to prison, single offence
        #V0408- max months sentenced to prison, single offence
        #V0410-max days sentenced to prison, single offence
        single_range_max={'d':'V0410','m':'V0408','y':'V0406'}

        #V0413- max years, multi flat
        #V0414- max months, multi flat
        #V0415- max days, multi flat
        multi_flat={'d':'V0415','m':'V0414','y':'V0415'}

        #V0417, max sentence min years, multi range
        #V04019- max sentence min months, multi range
        #V0421-max sentence min days, multi range
        multi_range_min={'d':'V0417','m':'V0419','y':'V0421'}
        
        #V0418- max sentence max years, multi range
        #V0420- max sentence max months, multi range
        #V0422- max sentence max days, multi range
        multi_range_max={'d':'V0422','m':'V0420','y':'V0418'}
        
        #V0440- years, multi 2 flat
        #V0441- months, multi 2 flat
        #V0442- days, multi 2 flat
        multi2_range_flat={'d':'V0442','m':'V0441','y':'V0440'}

        #V0444- min years, multi 2 range
        #V0446- min months, multi 2 range
        #V0448- min days, multi 2 range
        multi2_range_min={'d':'V0448','m':'V0446','y':'V0444'}
        
        #V0445- max years, multi 2 range
        #V0447- max months- multi 2 range
        #V0449- max days, multi 2 range  
        multi2_range_max={'d':'V0449','m':'V0447','y':'V0445'}
        #TODO: preprocess columns
        #TODO : check that only one of the converted year columns is populated for each row
        

        
        self.dmys={'single_range_min':single_range_min,'single_range_max':single_range_max,'single_flat':single_flat, \
                   'multi_flat':multi_flat,'multi_range_min':multi_range_min,'multi_range_max':multi_range_max,\
                       'multi2_range_flat':multi2_range_flat,'multi2_range_min':multi2_range_min,'multi2_range_max':multi2_range_max}
        
        #preprocess these columns
        to_prep=[]
        for sent_type, dmy in self.dmys.items():
            for unit,varname in dmy.items():
                to_prep.append(varname)
        
        self.prep_cols(to_prep)
        
        dmys_conv={}
        
        self.converted_years=[]
        
        for sent_type, dmy in self.dmys.items():
                #add a di
                dmys_conv[sent_type]={}
                for unit , colname in dmy.items():
                    #not skipping over years- instead divide it by 1
                    #then can append everything to the list for summing
 
                    #add y to indicate it's been converted into years
                    newcol=colname+'y'
                    
                    #use np.floor because we want to round down to the nearest integer
                    self.subset[newcol]=np.floor(self.subset[colname]/denoms[unit])
                    
                    #add this to dictionary for summing
                    dmys_conv[sent_type][unit]=newcol
                
                #name for converted years
                newyrs=sent_type+'c'
                print('new yrs', newyrs)
                #sum the values
                #dmys_conv[sent_type][newyrs]=sum(dmys_conv[sent_type].values())
                
                #sum values add the converted value to the main subset
                print(dmys_conv[sent_type].values())
                self.subset[newyrs]=sum([self.subset[yr] for yr in dmys_conv[sent_type].values()])
                
                #this will be picked up by binarise_continuous function
                self.converted_years.append(self.subset[newyrs])

    def xrationalise_dmy(self):
        
        max_vals={'m':11,'d':365}
        
        pair={'m':'y','d':'m'}
        
        single_flat={'d':'V0404','m':'V0403','y':'V0402'}
        
        #sequence is important as days need to be changed before months
        single_range_min={'d':'V0409','m':'V0407','y':'V0405'}
        single_range_max={'d':'V0410','m':'V0408','y':'V0406'}
        
        dmys=[single_range_min,single_range_max,single_flat]
        
        for dmy in dmys:
            for unit , col in dmy.items():
                #skip over years
                if unit != 'y':
                    #check if any of the values in the column exceeds the max value
                    #if it does, update it's higher col
                    #only higher col as there are no decimals

                    #some_value = The value that needs to be replaced
                    #value = The value that should be placed instead
                    #df.loc[ df[“column_name”] == “some_value”, “column_name”] = “value”
                    
                    newcol=dmy[pair[unit]]+'y'
                    print(newcol)
                    #current val is the val currently in the higher column
                    currentval=self.subset[dmy[pair[unit]]]
                    maxv=max_vals[unit]
                    #self.subset.loc[self.subset[col] > maxv], self.subset[newcol] = currentval+ np.floor(self.subset[col]/max_vals[unit])
                    self.subset.loc[self.subset[col] > maxv, newcol] = currentval                     
                                       
                    #if we want to rationalise the lower value replace current val with the modulo after the max value
                    #entry%max_vals[key]
        
        print('new col value counts')
        print(self.subset['V0407r'].value_counts())
    
    def xcollapse_ymd(self,fill_w_mode=[-9,-2,-1],fill_w_0=[-8,np.nan]):
        #print('Collapsing year, months and days')
        

        
        #V0400- range or indet, single
        #V0401- type of flat, single
        #VO402- years sentenced to prison, single offence
        #V0403- months sentenced to prison, single offence
        #V0404- days sentenced to prison, single offence
    
        #V0405- min years sentenced to prison, single offence
        #V0406- max years sentenced to prison, single offence
        #V0407- min months sentenced to prison, single offence
        #V0408- max months sentenced to prison, single offence
        #V0409- min days sentenced to prison, single offence
        #V0410-max days sentenced to prison, single offence
        
        #V0411- single/flat or indet, multi
        #V0412- type of single/flat   , multi     
        #V0413- max years, multi flat
        #V0414- max months, multi flat
        #V0415- max days, multi flat
        
        #V0416- range of time, indeterminate sentence, multi range
        #V0417, max sentence min years, multi range
        #V0418- max sentence max years, multi range
        #V04019- max sentence min months, multi range
        #V0420- max sentence max months, multi range
        #V0421-max sentence min days, multi range
        #V0422- max sentence max days, multi range
    
        #V0437- single or indet mutli
        #V0438- single or indent multi2?
        #V0439- type or single/flat multi2
        #V0440- years, multi 2
        #V0441- months, multi 2
        #V0442- days, multi 2
        
        #V0444- min years, multi 2
        #V0445- max years, multi 2
        #V0446- mon months, multi 2
        #V0447- max months- multi 2
        #V0448- min days, multi 2
        #V0449- max days, multi 2
        

        
        #TODO: mutli2 flat offence length
        #TODO: add multi2 flat offence length to combined offence length
        #TODO: add multi2 categorical to binarise_categorical
        #TODO: add columns to prep_cols list
        

        
        self.prep_cols(['V0402','V0413','V0403','V0404','V0414','V0415'])
        
        #single offence single/flat sentence (years: V0402, months: V0403,days: V0404)
        self.subset['single_offence_length']=(self.subset['V0402']*365)+(self.subset['V0403']*30.5)+(self.subset['V0404'])

        #multiple offences (years: V0413, months: V0414, days: V0415)
        self.subset['multiple_offence_length']=(self.subset['V0413']*365)+(self.subset['V0414']*30.5)+(self.subset['V0415'])
        #test['multiple_offence_length']=(test['V0413']*365)+(test['V0414']*30.5)+(test['V0415'])
        
        #multi2 offence length
        
        #combine both of the above into a continuous value
        self.subset['combined_sentence_length']=(self.subset['single_offence_length']+self.subset['multiple_offence_length'])

        #convert to year because of change in how we're doing it
        self.subset['combined_sentence_length']=round(self.subset['combined_sentence_length']/365,2)
        
        self.cont_sent_length_vars.append('combined_sentence_length')
    
        
    def binarise_continuous(self):
        
        #print('Checking if continuous columns are above threshold')
        #we don't need to prep the continuous variables as they have already been pre processed

        for v in self.converted_years:

            vname=v.name
            #print(v)
            #print('Th is',self.th)
            #will assign columns to original dataframe

            self.binarise_this_col(vname)
            #print('Now have this many cols',len(self.subset.columns))
        
            #get the most recent out name
            #debugging- print value counts
            #print('Value count for binarise_continuous')
            #print(self.subset[self.out_name].value_counts(dropna=False))
            self.one_if_any.append(self.subset[self.out_name].name)

    def binarise_this_col(self,conv_col):
    #def binarise_this_col(self,conv_col,fill_w_mode=[-9,-2,-1],fill_w_0=[-8,np.nan])

        out_name=f'{conv_col}_above{self.th}'

        #then indicate if over or above threshold
        #self.subset.loc[self.subset[conv_col]<self.th,out_name]=0
        self.subset.loc[self.subset[conv_col]>=self.th,out_name]=1
        
        self.subset[conv_col].value_counts(dropna=False)

        #print(f'Binarised {self.subset[out_name].name}')
        #update out_name
        self.out_name=out_name
    
    def binarise_categorical(self):
        #print('Binarising categorical columns- life or death sentences will be set above threshold')
        
        #print('Starting the flat categorical sentences')
        #we need to do the flat sentences, which contain categorical info
        #6 the most common value within the subset that answered this question- going to be raplced with 0 anyway
        self.subset['single_sent_life']=copy.deepcopy(self.subset['V0401'].replace(to_replace=[-9,-2,-1,5,6,np.nan],value=0))
        #we can then replace the values that indicate a life or a death sentence with 997 for adding later
        self.subset['single_sent_life'].replace(to_replace=[1,2,3,4],value=1,inplace=True)
        self.one_if_any.append(self.subset['single_sent_life'].name)
        
        #print('Value count for categorical col 1')
        #print(self.subset['single_sent_life'].value_counts(dropna=False))
        
        #we can do the same for prisoners with multiple offences
        self.subset['multi_sent_life']=copy.deepcopy(self.subset['V0412'].replace(to_replace=[-9,-2,-1,5,6,np.nan],value=0))
        self.subset['multi_sent_life'].replace(to_replace=[1,2,3,4],value=1,inplace=True)
        self.one_if_any.append(self.subset['multi_sent_life'].name)

        #print('Value count for categorical col 2')
        #print(self.subset['multi_sent_life'].value_counts(dropna=False))

        #and multiple consecutive sentences of the same legnth
        #self.subset['multi2_sent_life']=copy.deepcopy(self.subset['V0439'].replace(to_replace=[-9,-2,-1,5,6,np.nan],value=0))
        #self.subset['multi2_sent_life'].replace(to_replace=[1,2,3,4],value=1,inplace=True)
        #self.one_if_any.append(self.subset['multi2_sent_life'].name)
        
    def make_one_if_any(self,newcol,test_val=1):
        
        temp_name='sum'
    
        print('One if any')
        print([col for col in self.one_if_any])
        
        #the binarised continuous columns and categoricl columns are included in the list
        self.subset[temp_name] = self.subset[list(self.one_if_any)].sum(axis=1)
        
        print(self.one_if_any)
        print('Check we are only gettings 1s')
        for name in self.subset[list(self.one_if_any)]:
            print(self.subset[name].value_counts(dropna=False))

        self.subset[self.target_name] = np.where(self.subset['sum']==test_val, 1,0)
        

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

