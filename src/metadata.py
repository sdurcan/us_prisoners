# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 17:00:03 2022

@author: siobh
"""

import pandas as pd
import numpy as np
#from pandas_profiling import ProfileReport
from pandas.api.types import infer_dtype
import pickle
from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
from pandas import read_csv
import seaborn as sn
import matplotlib.pyplot as plt
from src import load

#coefficient of varitation
cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100 

#see if column has a constant value
def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

def process_label(input_str):
    #takes in the label string from the metadata and splits into useful parts
    #returns a dictionary
    #delimiter is hyphen, only one split as we want to split on first hyphen which delimits
    #example input string: ='V0009 - DEMO1_Mo: Date of birth (mo) (suppressed)DEMO1. What is your date of birth?Taken from: Survey of Prison Inmates, United States, 2016.'
    #example R input string: RV0048: Temporarily suppressedTaken from: Survey of Prison Inmates, United States, 2016.
    #remove the variable name before the hyphen
    
    #there are a few recoded variables with a different description format
    if input_str.startswith('RV'):
        catlabel=input_str.split(':',1)[0]
        varname=input_str.split(':',1)[0]
        var_desc=input_str.split(':',1)[1].replace('Taken from: Survey of Prison Inmates, United States, 2016.','')
    
    else:
    
        varname=input_str.split('-',1)[0]
        alldesc=input_str.split('-',1)[1]

        #split out the category label from the description
        split_catlabel=alldesc.split(':',1)
        #print(split_catlabel)
        catlabel=split_catlabel[0]
        var_desc=split_catlabel[1].replace('Taken from: Survey of Prison Inmates, United States, 2016.','')

    #return a dictionary
    output={'varname':varname,'catlabel':catlabel,'var_desc':var_desc}

    return output

metadata=pd.read_csv(r'C:\Users\siobh\OneDrive\Masters\Dissertation\us_prisonsers\data\processing_config\variable_metadata_scraped.csv',header=0)
#convert variable metadata to dict
metadata_dict=metadata.to_dict(orient='index')

#create new empty dict
new_dict={}
bad_labels=['V1085']
count=0

#loop through and reorganise
for key in list(metadata_dict.keys()):
    #print('{}/{}'.format(count, len(metadata_dict.keys())))
    count=count+1
    #extract variable name
    varname=metadata_dict[key]['var name']
    print(varname)
    if varname not in bad_labels:
        #extract label
        label=metadata_dict[key]['label']
        #extract var type
        vartype=metadata_dict[key]['var type']

        #print(varname)
        #process label
        processed_label=process_label(label)
        #returns a dict
        catlabel=processed_label['catlabel']
        var_desc=processed_label['var_desc']

        new_dict[varname]={'label':label,'var type':vartype,'catlabel':catlabel,'var_desc':var_desc}


cols={}


#caution=pd.read_csv(r'C:\Users\siobh\OneDrive\Masters\Dissertation\dissertation/proceed with caution.csv')

bad_labels2=['V0772','V1085']

#caution=pd.read_csv(r'C:\Users\siobh\OneDrive\Masters\Dissertation\dissertation/proceed with caution.csv')
#caution_list=list(caution['variable'])

prisoners,subset=load.import_prisoners(subset=3)

for col in subset.columns:
    
    if col not in bad_labels2:
        #name of column for dict key lookups
        colname=col
        print(colname)
        #pandas series of columns for calculations
        col=prisoners[col]
       
        #set to_drop to false as default
        to_drop=False
        
        #get values from metadata file
        catlabel=new_dict[colname]['catlabel']
        official_dtype=new_dict[colname]['var type']
        var_desc=new_dict[colname]['var_desc']
        
        #descriptive
        dtype=infer_dtype(prisoners[colname])
        sum_nans=col.isna().sum()
        count=col.count()
        unique=list(pd.unique(col))
        constant= is_unique(col)
        
        #statistics
        maxval=col.max()
        meanval=''
        if pd.isna(col.mean())==False:
            meanval=round(col.mean())
        minval=col.min()
        std=col.std()
        #mode gives most common value
        #mode() produces a series, so convert to list and take first value
        if len(list(col.mode()))>0:
            mode=list(col.mode())[0] 
        #TO DO: most frequent values. this could be done with value_counts(), but useful if binarising?
        median=col.median()
        skew=col.skew()
        kurt=col.kurtosis()
        absdev=col.mad()
        #TO DO: fix cv function
        #cv=cv(np.array(col))
        
        #processing
        
        #TO DO: flag if likely categorical although numerical
        likely_categorical=False
        if len(unique)<30:
                if dtype=='integer':
                    likely_categorical=True
        
        supressed=False
        if len(unique)==1:
            if unique[0]==999:
                supressed=True
                to_drop=True
            elif unique[0]==9999:
                suprressed=True
                to_drop=True
               

        
        not_used=False
        
        if "VARIABLE NOT USED" in var_desc:
            not_used=True

        if "VARIABLE NOT USED" in catlabel:
            not_used=True
            
        #upcoded starting with upper and lower case in data desc
        upcoded_list=["Upcoded","upcoded"]
        upcoded=[word in var_desc for word in upcoded_list][0]
        
        #lower case not checked for because it's used to describe 'original offenses'
        #as oppose to the original question that was later recoded
        orig= 'Original' in var_desc
        
        #before or after admission/arrest
        #manually created wordlist
        after_keywords=["since admission","since arrest","after arrest"]
        after_admit= [word in var_desc for word in after_keywords][0]
        
        #manually created wordlist
        before_keywords=["before admission","before arrest","before arrest"]
        before_admit=[word in var_desc for word in before_keywords][0]

        #create list to capture dropped reason
        drop_reason=[]
                
        if supressed==True:
            to_drop==True
            drop_reason.append('supressed')
        
        if not_used==True:
            to_drop=True
            drop_reason.append('not_used')
            
        bgn= 'Begin Flag' in var_desc
        if bgn==True:
            to_drop=True
            drop_reason.append('begin flag')
        end= 'End Flag' in var_desc
        if end==True:
            to_drop=True
            drop_reason.append('end flag')
        
        rep_weight= 'replicate weight' in var_desc
        if rep_weight==True:
            to_drop=True
            drop_reason.append('replicate weight')

        rep_weight2= 'REPWT' in catlabel
        if rep_weight2==True:
            to_drop=True
            drop_reason.append('replicate weight')
        


        section=''
        binarise_flag=''
        
        #some questions have an item flag indicating
        #the validity of the response
        #in these cases, 0 means the question is skipped
        #could probably also get rid of these as upcoded data
        flag='Item Flag' in var_desc
        
        if flag==True:
            drop_reason.append('item flag')
            to_drop=True
            #change any 0 to na
            col.replace(to_replace=0, value= None)
            col.replace(to_replace='0', value= None)
            col.replace(to_replace=2, value= None)
            col.replace(to_replace='2', value= None)
            col.replace(to_replace=3, value= None)
            col.replace(to_replace='3', value= None)
            col.replace(to_replace=7, value= None)
            col.replace(to_replace='7', value= None)
            col.replace(to_replace=8, value= None)
            col.replace(to_replace='8', value= None)
                    
        #resum nans after updates
        resum_nans=col.isna().sum()
            
        #need to set section to a blank string because not all variables will have a section
        #and this will give a variable ref berfore assignment error when building dict
        section=''
        sections=['DEMO','CJ','SES','MH','PH','AU','DU','DTX','RV', 'TMR']
        for sname in sections:
            #TO DO: check column not in multiplesectionhits
            if sname in var_desc:
                section=sname
        
        if section=='TMR':
            to_drop=True
            drop_reason.append('TMR')

        #put in dictionary
        cols[colname]={'nans':sum_nans, 'resum nans':resum_nans, 'supressed':supressed,'to_drop':to_drop, 'drop_reason':drop_reason,'constant':constant,'count':count,'max':maxval,'min':minval,'mean':meanval, 'mode':mode,'median':median, 'unique':unique,'count unique':len(unique),'skew':skew,'kurtosis':kurt,'std':std,'absdev':absdev,'inferred_dtype':dtype,'catlabel':catlabel,'official_dtype':official_dtype,'var_desc':var_desc, 'bgn':bgn, 'end':end,'before admit':before_admit, 'after admit':after_admit,'upcoded':upcoded, 'orig':orig, 'section':section, 'binarise flag':binarise_flag, 'section':section, 'confirm drop':'','notes':'', 'confirm section':'', 'confirm type':official_dtype}
    