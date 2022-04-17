# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 18:10:21 2022

@author: siobh
"""
import pandas as pd
import pickle
from pandas.api.types import infer_dtype
import numpy as np


def drop_constants (dataframe, save=1):
    'takes a dataframe as input and removes columns with constant values'
    #print("Original number of variables is {}".format(dataframe.count(axis=1)))
    nunique=dataframe.columns[dataframe.nunique() <= 1]
    nunique_list=list(nunique)
    #print(nunique_list)
    #print('Count of columns with constant values is {}'.format(len(nunique_list)))
    df_out=dataframe.drop(nunique_list,axis=1)
    #print("With constant values removed, number of variables is {}".format(dataframe.count(axis=1)))

    if save==1:
      #save constants columns that we want to drop to file
      with open('constant_cols_to_drop.pkl', 'wb') as fid:
          pickle.dump(nunique_list, fid)
    return df_out

#cautions_path=r'C:\Users\siobh\OneDrive\Masters\Dissertation\dissertation\data\processing_config\proceed with caution.csv'

#cautions_path=r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisoners/data/processing_config/proceed with caution.csv'

cautions_path=r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/processing_config/proceed_with_caution.csv'
scraped_metadata_path=r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/processing_config/variable_metadata_scraped.csv'
exception_nans_path=r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/processing_config/98dkrf.csv'

#cautions_path=r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisoners/data/downloaded_package/DS0001/37692-0001-Data.tsv'

def show_cautions(path=cautions_path):
    caution=pd.read_csv(path)
    caution.head()
    return caution


class generate_metadata():
    
    def __init__(self,scraped_metadata="",cautions="",exception_nans=""):
        self.scraped_metadata=scraped_metadata
        self.cautions=cautions
        self.exception_nans=exception_nans
        
        if scraped_metadata=="":
            self.scraped_metadata=scraped_metadata_path
        if self.cautions=="":
            self.cautions=cautions_path
        if self.exception_nans=="":
            self.exception_nans=exception_nans_path
        
        #caution list    
        with open(self.cautions) as f:
            self.caution_list = [line.strip() for line in f]
    
        #File 98dkrf/excpetion_nans lists the variables where 98,99 should be replaced with na because 98 and 99 represented missing don't know or refused or a missing value
        none_vals2=[98,99,'98','99']
        with open(self.exception_nans) as f:
            none_indices2 = [line.strip() for line in f]
    
        self.setup=['V0001','V0001B','V0002','V0003','V0004','V0005','V0006','V0007']
        
        self.ignore=['V0772','V1085']
        
        self.sixnines=['V0055Y','V0056Y']
        
        #vals to make na for item flag values
        self.nonevals3=[0,'0',2,'2',3,'3','7',7,'8',8]
        
        #self.load_metadata()
        #self.setup_dict()
        #self.calculate_var_features()
        
    
    def load_metadata(self):
        #import variable metadata file
        self.metadata=pd.read_csv(self.scraped_metadata,header=0)
    
        #convert variable metadata to dict
        self.metadata_dict=self.metadata.to_dict(orient='index')
    
    def process_label(self,input_str):
        #takes in the label string from the metadata and splits into useful parts
        #returns a dictionary
        #delimiter is hyphen, only one split as we want to split on first hyphen which delimits
        #example input string: ='V0009 - DEMO1_Mo: Date of birth (mo) (suppressed)DEMO1. What is your date of birth?Taken from: Survey of Prison Inmates, United States, 2016.'
        #example R input string: RV0048: Temporarily suppressedTaken from: Survey of Prison Inmates, United States, 2016.
        #remove the variable name before the hyphen

        if type(input_str)==str:
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
    
    def setup_dict(self):

        #create new empty dict
        new_dict={}
        
        count=0
    
        #loop through and reorganise
        for key in list(self.metadata_dict.keys()):
            print('key is',key)
            #print('{}/{}'.format(count, len(metadata_dict.keys())))
            count=count+1
            #extract variable name
            varname=self.metadata_dict[key]['var name']
            if varname not in self.ignore:
                #extract label
                label=self.metadata_dict[key]['label']
                #extract var type
                vartype=self.metadata_dict[key]['var type']
        
                #print(varname)
                #process label
                processed_label=self.process_label(label)
                #returns a dict
                catlabel=processed_label['catlabel']
                var_desc=processed_label['var_desc']
        
                new_dict[varname]={'label':label,'var type':vartype,'catlabel':catlabel,'var_desc':var_desc}
            
        self.metadata_dict=new_dict
        
        
    def calculate_var_features():
        #call class

        cols={}
        
        #yse metadatadict so that we don't pick up ignored columns
        for col in self.metadata_dict.keys:
            
            if col not in self.ignore:
            #pass to class object
            #add to something stored in this class
                result=var_feature_calc(col,self.metadata[col],self).get_dict
                    

class col_feature_calc():
    
    def __init__(self,colname,col,metadata_generator):
        self.colname=colname
        self.col=col
        self.metadata_generator=metadata_generator
        self.cols={}
        self.metadata_dict=metadata_generator.metadata_dict
        self.data_in=metadata_generator.data
        
        #set to_drop to false as default
        to_drop=False
        
        #get values from metadata file
        self.catlabel=self.metadata_dict[colname]['catlabel']
        self.official_dtype=self.metadata_dict[colname]['var type']
        self.var_desc=self.metadata_dict[colname]['var_desc']
        
        self.sum_nans=col.isna().sum()
            
        #some questions have an item flag indicating the validity of the response
        #0 means the question is skipped
        self.flag='Item Flag' in self.var_desc
        
        if self.flag==True:
            self.drop_reason.append('item flag')
            self.to_drop=True
            #TO DO: update nan list
                     
            
        #resum nans after updates (for SOME of the columns, there should be an increase)
        self.resum_nans=col.isna().sum()
        
        #descriptive
        self.dtype=infer_dtype(data_in[colname])
        
        self.count=col.count()
        self.unique=list(pd.unique(col))
        self.constant= is_unique(col)
        
        #statistics
        self.maxval=col.max()
        self.meanval=''
        if pd.isna(col.mean())==False:
            self.meanval=round(col.mean())
        self.minval=col.min()
        self.std=col.std()
        #mode gives most common value
        #mode() produces a series, so convert to list and take first value
        if len(list(col.mode()))>0:
            self.mode=list(col.mode())[0] 
        #TO DO: most frequent values. this could be done with value_counts(), but useful if binarising?
        self.median=col.median()
        self.skew=col.skew()
        self.kurt=col.kurtosis()
        self.absdev=col.mad()
     
        #processing
        
        self.not_used=False
        
        if "VARIABLE NOT USED" in self.var_desc:
            self.not_used=True

        if "VARIABLE NOT USED" in self.catlabel:
            self.not_used=True
            
        #upcoded starting with upper and lower case in data desc
        upcoded_list=["Upcoded","upcoded"]
        self.upcoded=[word in self.var_desc for word in upcoded_list][0]
        
        #lower case not checked for because it's used to describe 'original offenses'
        #as oppose to the original question that was later recoded
        self.orig= 'Original' in self.var_desc
        
        #before or after admission/arrest
        #manually created wordlist
        after_keywords=["since admission","since arrest","after arrest"]
        self.after_admit= [word in self.var_desc for word in after_keywords][0]
        
        #manually created wordlist
        before_keywords=["before admission","before arrest","before arrest"]
        self.before_admit=[word in self.var_desc for word in before_keywords][0]

        #create list to capture dropped reason
        self.drop_reason=[]
                
        if self.supressed==True:
            self.to_drop==True
            self.drop_reason.append('supressed')
        
        if self.not_used==True:
            self.to_drop=True
            self.drop_reason.append('not_used')
            
        self.bgn= 'Begin Flag' in self.var_desc
        if self.bgn==True:
            self.to_drop=True
            self.drop_reason.append('begin flag')
            
        self.end= 'End Flag' in self.var_desc
        if self.end==True:
            self.to_drop=True
            self.drop_reason.append('end flag')
        
        self.rep_weight= 'replicate weight' in self.var_desc
        if self.rep_weight==True:
            self.to_drop=True
            self.drop_reason.append('replicate weight')

        self.rep_weight2= 'REPWT' in self.catlabel
        if self.rep_weight2==True:
            self.to_drop=True
            self.drop_reason.append('replicate weight')
        
        if self.colname in self.metadata_generator.setup:
            self.to_drop=True
            self.drop_reason.append('setup')
        
        self.caution_flag= colname in metadata_generator.caution_list

        self.section=''
        self.binarise_flag=''
            
        #need to set section to a blank string because not all variables will have a section
        #and this will give a variable ref berfore assignment error when building dict
        sections=['DEMO','CJ','SES','MH','PH','AU','DU','DTX','RV','TMR']
        for sname in sections:
            #TO DO: check column not in multiplesectionhits
            if sname in self.var_desc:
                self.section=sname
        
        if self.section=='TMR':
            self.to_drop=True
            self.drop_reason.append('TMR')
        
        def get_dict(self):
        
        #put in dictionary
         return dict({'nans':self.sum_nans, 'resum nans':self.resum_nans, 'supressed':self.supressed,'to_drop':self.to_drop,\
                       'drop_reason':self.drop_reason,'constant':self.constant,'count':self.count,'max':self.maxval,'min':self.minval,\
                           'mean':self.meanval, 'mode':self.mode,'median':self.median, 'unique':self.unique,'count unique':len(self.unique),\
                               'skew':self.skew,'kurtosis':self.kurt,'std':self.std,'absdev':self.absdev,'inferred_dtype':self.dtype,'catlabel':self.catlabel,\
                                   'official_dtype':self.official_dtype,'var_desc':self.var_desc, 'bgn':self.bgn, 'end':self.end,'before admit':self.before_admit,\
                                       'after admit':self.after_admit,'upcoded':self.upcoded, 'orig':self.orig, 'section':self.section, 'binarise flag':self.binarise_flag,\
                                           'confirm drop':'','notes':'', 'confirm section':'', 'confirm type':self.official_dtype,'caution flag':self.caution_flag})
        
            
        #see if column has a constant value
def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

