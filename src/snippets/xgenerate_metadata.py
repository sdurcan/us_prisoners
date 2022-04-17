# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 18:10:21 2022

@author: siobh
"""
import pandas as pd



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

#see if column has a constant value
def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()



def show_cautions(path=r'C:\Users\siobh\OneDrive\Masters\Dissertation\dissertation/proceed with caution.csv'):
    caution=pd.read_csv(path)
    caution.head()
    return caution


class generate_metadata(scraped_metadata="",cautions="",exception_nans=""):
    
    def __init__(self,scraped_metadata,cautions):
        if scraped_metadata=="":
            self.scraped_metadata=r'C:\Users\siobh\OneDrive\Masters\Dissertation\dissertation\data/prisoners_metadata.csv'
        if self.cautions=="":
            self.cautions=r'C:\Users\siobh\OneDrive\Masters\Dissertation\dissertation/proceed with caution.csv'
        if self.exception_nans=="":
            self.exception_nans=r'C:\Users\siobh\OneDrive\Masters\Dissertation\dissertation\data/98dkrf.csv'
        
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
        
        self.load_metadata()
        
    
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
        
        
    
        cols={}
        
        for col in self.metadata.columns:
            
            if col not in self.ignore:
                #name of column for dict key lookups
                colname=col
                #print(colname)
                #pandas series of columns for calculations
                col=metadata[col]
               
                #set to_drop to false as default
                to_drop=False
                
                #get values from metadata file
                catlabel=new_dict[colname]['catlabel']
                official_dtype=new_dict[colname]['var type']
                var_desc=new_dict[colname]['var_desc']
                
                sum_nans=col.isna().sum()
        
                #some questions have an item flag indicating
                #the validity of the response
                #in these cases, 0 means the question is skipped
                #could probably also get rid of these as upcoded data
                flag='Item Flag' in var_desc
                if flag==True:
                    drop_reason.append('item flag')
                    to_drop=True
                    pris3[colname].replace(to_replace=0,value=np.NaN,inplace=True)
                  
                if colname in none_indices1:
                    for val in none_vals1:
                        #print('The none val is {}'.format(val))
                        pris3[colname].replace(to_replace=val,value=np.NaN,inplace=True)
                
                if colname in none_indices2:
                    for val in none_vals2:
                        pris3[colname].replace(to_replace=val,value=np.NaN,inplace=True)
                
                #two columns have the value 999999 to showing missing data
                if colname in sixnines:
                    pris3[colname].replace(to_replace=999999,value=np.NaN,inplace=True)
                    
        
                #resum nans after updates (for SOME of the columns, there should be an increase)
                resum_nans=col.isna().sum()
                
                #descriptive
                dtype=infer_dtype(prisoners[colname])
                
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
                
                if colname in setup:
                    to_drop=True
                    drop_reason.append('setup')
                
                caution_flag= colname in caution_list
        
                section=''
                binarise_flag=''
                    
                #need to set section to a blank string because not all variables will have a section
                #and this will give a variable ref berfore assignment error when building dict
                section=''
                sections=['DEMO','CJ','SES','MH','PH','AU','DU','DTX','RV','TMR']
                for sname in sections:
                    #TO DO: check column not in multiplesectionhits
                    if sname in var_desc:
                        section=sname
                
                if section=='TMR':
                    to_drop=True
                    drop_reason.append('TMR')
                
                #put in dictionary
                cols[colname]={'nans':sum_nans, 'resum nans':resum_nans, 'supressed':supressed,'to_drop':to_drop,\
                               'drop_reason':drop_reason,'constant':constant,'count':count,'max':maxval,'min':minval,\
                                   'mean':meanval, 'mode':mode,'median':median, 'unique':unique,'count unique':len(unique),\
                                       'skew':skew,'kurtosis':kurt,'std':std,'absdev':absdev,'inferred_dtype':dtype,'catlabel':catlabel,\
                                           'official_dtype':official_dtype,'var_desc':var_desc, 'bgn':bgn, 'end':end,'before admit':before_admit,\
                                               'after admit':after_admit,'upcoded':upcoded, 'orig':orig, 'section':section, 'binarise flag':binarise_flag,\
                                                   'section':section, 'confirm drop':'','notes':'', 'confirm section':'', 'confirm type':official_dtype,'caution flag':caution_flag}

