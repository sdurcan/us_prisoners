# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 13:49:34 2022

@author: siobh
"""
import load
import pandas as pd
import numpy as np
import copy

class offense_transform:
    """Densifies offense data and returns subset with original cols removed"""
    
    def __init__(self,subset):
        self.subset=subset
        self.cols_in=copy.deepcopy(self.subset.columns)
        #self.keep_cols=[]
        self.offense_variables=load.offense_variables()
        self.flatten_colnames()
        self.set_single_offense()
        self.check_offense_data()
        self.collapse_offense_lists()
        self.set_lookup_for_ctrl_off()
        self.set_single_ctrl_off()
        self.encode_offense_codes()
        self.set_probation_parole_violation()
        self.drop_cols()
        #TODO: decide how to deal with secondary offences that are not the controlling offenses

    def flatten_colnames(self):
        #This isn't combined within other functions 
        #so that we can skip those functions
        #but still access a flat list of all colnames
        self.all_colnames=[]
        for inmate_type, info in self.offense_variables.items():
            codes=list(info['codes'].values())
            self.all_colnames=self.all_colnames+codes

    def check_offense_data(self):
        self.check_only_one_ctrl_type()

    def check_only_one_ctrl_type(self):
        """Checks in a list of lists that only one of the lists is populated"""

        codes=[]
        for inmate_type, info in self.offense_variables.items():
            type_codes=list(info['codes'].values())
            #print('Single',type_codes)
            codes.append(type_codes)
            #print('Cumulative',codes)
        self.check_one_per_group(self.subset,codes)
        
    @staticmethod
    def one_if_any(dataset,colnames):
        """Takes a dataset and a list of column names and returns a boolean col
        indicating of any of those cols had a non nan value on that row"""
        #check if any of these variables are populated 
        return dataset[colnames].any(axis=1).to_frame()
        
    @staticmethod
    def check_one_per_group(subset,listoflists):
        """Checks in a list of lists that only one of the lists is populated"""
        #sum over the codes to see if this is what we think it is
        count=0
        for varlist in listoflists: 
            #check if any of these variables are populated 
            any_in_group=offense_transform.one_if_any(subset,varlist)
            
            #TODO: would be better to append to empty dataframe
            if count==0:
                #first loop, so just assign series
                any_in=any_in_group
            else:
                any_in=pd.concat([any_in,any_in_group],axis=1)
            count=+1
        
        #then sum across any_in- should always be 1
        #this shows that only one of the sets has values- so 
        #particpants aren't answering for multiple prisoner types
        allowed={0,1}
        group_count=any_in.sum(axis=1)
        not_allowed=set(group_count.values)-allowed 
        if len(not_allowed)>0:
            print('Some rows have values in multiple controlling offence variable groups')

    def collapse_offense_lists(self):
        """Combines all the controlling offense variables into one set of variables"""
        
        #TODO: the value counts of the new columns should match the value count offenses

        #get a list of prisoner types as this will change depending on the subset
        pris_types=list(self.subset['V0063'].value_counts().index)
     
        self.new_code_cols={}
        self.new_type_cols={}        

        
        for pt in pris_types:

            #assumes five offenses are listed in the survey

            for item in range(1,6):
      
                code_colname=self.offense_variables[pt]['codes'][item]
                type_colname=self.offense_variables[pt]['off_types'][item]
                self.subset.loc[self.subset['V0063'] == pt,f'off_{item}_code'] = self.subset[code_colname]
                self.subset.loc[self.subset['V0063'] == pt, f'off_{item}_type'] = self.subset[type_colname]

                #add the new cols to the dict so that we can keep them when we drop

                self.new_code_cols[item]= f'off_{item}_code'
                self.new_type_cols[item]=f'off_{item}_type'


    def set_single_offense(self):
        """Creates a list of indexes of prisoners with multiple offenses
        and a list of single offenses. Informs how controlling offense is found"""

        #count across all offense columns
        #checks should warn if there is an error in the data where more than one
        #group has a value
        self.subset['count_offenses']=self.subset[self.all_colnames].count(axis=1)
                
        self.subset.loc[self.subset['count_offenses']==1,'single_offense']=1
        
    def set_lookup_for_ctrl_off(self):
        #TODO where the 'type' field is used to indicate parole or probation
        
        #Multiple controlling offenses is a different concept to multiple offenses
        #For a prisoner with multiple offenses they will indicate which offense recieved the longest sentence in the columns listed below
        #If the prisoner recieved one sentence for all offenses OR their longest offence covered two offenses
        #Then the controlling offense hierarchy is violent>property>drugs
        
        #V0428==1 : First Offense
        #V0429==2 : Second Offense
        #V0430==3 : Third Offense
        #V0431==4 : Fourth Offense
        #V0432==5 : Fifth Offense
        #V0433==6 : One sentence for all Offenses
        #V0434==7:  all sentences the same length
        #V0434 is all sentences are the same length
        self.ctrl_cols=['V0428','V0429','V0430','V0431','V0432']
        
        
        
        #in this version, the values in the ctrt_loc are 99 or the offence
        #looping through like this because calling replace on list of columns doesn't update original
        for col in self.ctrl_cols:
            self.subset[col].replace(99,np.nan,inplace=True)
     
        #count row-ise to get count of controlling offense
        #TODO: all prisoners with multiple offenses should have provided a controlling offense
        self.subset['ctrl_count']=self.subset[self.ctrl_cols].count(axis=1)

        #if ctrl_count is 0 and single_offenses is 1, make ctrl_count 1
        self.subset.loc[((self.subset['ctrl_count']==0) & (self.subset['single_offense']==1)),'ctrl_count']=1

        #there are some caetgorical rather than lookup answers 
        self.subset.loc[self.subset['V0433']==6,'ctrl_count']='one for all'
        self.subset.loc[self.subset['V0434']==7,'ctrl_count']='all same'
        #there are 9s in this col aswell, but they represent blank values
        self.subset.loc[self.subset['V0435'].isin([7,8]),'ctrl_count']='dk/ref'
        
        #TODO: if ctrl_count == 1 AND single_offense is 1, ctrl lookup is 1
        #TODO: check, for single offenses, the first offense is always filled
        self.subset.loc[(self.subset['ctrl_count']==1) & (self.subset['single_offense']==1),'ctrl_lookup']=1
        
        self.subset.loc[self.subset['V0433']==6,'ctrl_lookup']='one for all'
        self.subset.loc[self.subset['V0434']==7,'ctrl_lookup']='all same'
        self.subset.loc[self.subset['V0435'].isin([7,8]),'ctrl_lookup']='dk/ref'
        
        #TODO if ctrl_count is 1 and they have multiple offenses, use the sum of the ctrl offenses
        #because other columns are blank (as there is only one offense counted), the sum output will be the offence # tp look up 
        self.subset.loc[(self.subset['ctrl_count']==1) & (self.subset['single_offense']!=1) ,'ctrl_lookup']=self.subset[self.ctrl_cols].sum(axis=1)
        
        #The rows with na are then those people with multiple controlling offenses- we need to do the lookup for rows and the rows one for all and all same
        #TODO: all of the ctrl_counts 2,3,4 have na in ctrl_lookup, which is correct
        #TODO: what about the other nas in ctrl_lookup? why? 33 rows
        #r=op.subset.loc[ (op.subset['ctrl_lookup'].isna()) & ~op.subset['ctrl_count'].isin([2,3,4]) ]
        self.subset.loc[(self.subset['ctrl_lookup'].isna()) & (self.subset['ctrl_count'].isin([2,3,4])),',missing_check']='multiple'
        print('Ctrl_lookup')
        print(self.subset['ctrl_lookup'].value_counts(dropna=False))
    
    def set_lookup_for_ctrl_off_multi2(self):

        #self.subset.loc[self.subset.loc['ctrl_count'].isin([2,3,4]),'offense_details']={}
        #If controlling_offense is 'all_same' or 'one_for_all', then need to lookup
        
        #violent offense>property offense>drug offense>po offense>other offense


        #For prisoners with multiple offenses the offense
        #that recieved the longest sentence (controlling offense) is stored in

        #V0435==7,8 :DK/REF
        
        #MULTI- which offense recieved the longest sentence
        #V0428==1 : First Offense
        #V0429==2 : Second Offense
        #V0430==3 : Third Offense
        #V0431==4 : Fourth Offense
        #V0432==5 : Fifth Offense
        
        #Offense multi logic- if the above sum to more than 1 on a row, then we
        #need to work out the controlling offense so that we can get the offense code
        #1- violent offense
        #get violent offenses and pick one, then map to code
        
        #V0433 == 6 : One sentence for all offenses
        #V0434 == 7 : All sentences are the same length
        pass
        
    def set_single_ctrl_off(self):
        '''Populates ctrl_off and ctrl_type according to lookup value in ctrl_count'''
        #TODO: check dk-refs don't have offense populated anywhere?
        
        #dropping na because we don't want them and trying to slip over them using the list seems to cause errors      
        print(self.subset['ctrl_lookup'].value_counts())
        for index in self.subset['ctrl_lookup'].value_counts().index:
            #TODO: work out how to not need this line
            if index not in ['dk/ref','all same','one for all']:
                #get the column name from the new_code_cols dictionary
                code=self.new_code_cols[index]
                type_col=self.new_type_cols[index]
                self.subset.loc[self.subset['ctrl_lookup']==index,'ctrl_off_code']= self.subset[code]
                self.subset.loc[self.subset['ctrl_lookup']==index,'ctrl_off_type']= self.subset[type_col]
                #TODO: type should always be violent (1) : violent
                                                                                 

    def test_if_single_offence_not_first_listed():

        pass
    

    def encode_offense_codes(self):
        '''Creates a column for each offense code and populates it with how many times that offense
        appears for each prisoner'''

        #Parole offence code- 490
        #Probation violation-500
        #CJA6-CJA9- contain new offences when parole violators readmitted
        #Not anchored to controlling list
        #CJb1-CJB4 Inmate Type 10 and 12
        #CJB3
        #CJBType==2:probation violation, 1:parole violation, 
        
        
        #create a smaller subset to deal with
        #some offenses are held in variables that are not controlling offenses, so need to add them in
        self.other_offense_variables=['V0246','V0247','V0248','V0249','V0322','V0323','V0324','V0325','V0326']
        alloffs=self.all_colnames+self.other_offense_variables
        offense_cols_subset=self.subset[alloffs]
                
        #get a list of the codes across all offenses
        self.all_codes=[]
        for col_name in offense_cols_subset.columns:
            new=list(offense_cols_subset[col_name].unique())
            self.all_codes=self.all_codes+new
            self.all_codes=list(set(self.all_codes))
                
        self.all_codes=[code for code in self.all_codes if np.isnan(code)==False]
        
        self.code_list=[]
        for code in self.all_codes:
    
            #create a copy of values that are boolean wrt code
            c=copy.deepcopy(offense_cols_subset==code)
            print(c)
            self.subset[code]=copy.deepcopy(c.sum(axis=1))
            self.code_list.append(code)
   
    def set_probation_parole_violation(self):
        #460, 462- escape code
        #490- parole violation code
        #500 -probation violation code
        #460, 462 and 490 and 500 do not appear in the all_codes list. However, the values in V0240 indicate that there are probation and parole violators. Therefore, we need to add columns to capture this
        #CJ26 Type forr what offences were you incarcerated before you escaped 1 Parole 2 Probation 4 Escape V0214
        #CJ4 When you were arrested or charged with the offenses you are now serving time, were you on 1 Parole supervsion, 2 Probation, 3 Escape V0078
        #CJ7 When you were arrested or charged for the offenses for which you are now in prison, were you on 1 Parole 2 Probation 3 Escape V0081 
        #CJ11 Type For what offenses are you now in prison 1 Parole Violation 2 Probation V0108
        #CJ15 Type For what offenses are you awaiting trial/hearing 1 Parole 2 Probation  V0134
        #CJ17 When you were admitted to prison after being on escape, were you arrested or charged with new offenses 1 Yes 2 No V0160 
        #CJ19 For what new offenses were you sentenced? 1 Parole Violation 2 Probation Violation V0162
        #CJ23 type for what new offenses were you arrested or charged following your escape 1 Parole 2 Probation 4 Escape V0188
        #CJ26 Type For what offenses were you incarcerated before you escaped 1 Parole 2 Probation 4 Escape V0214
        #CJA1 Type For what offenses were you serving time in prison and then put on parole of post-release supervision 1 Parole 2 Probation 3 Actual offenses V0240
        #CJA7 Type For what new offenses were you arrested or charged 1 Parole 2 Probation V0288
        #CJB3 Type For what offenses were you on probation from a court? 1 Parole 2 Probation V0316
        #CJB6 Type for what new offenses were you sentenced? 1 Parole 2 Probation V0340
        #CJB9 Type For what new offenses were you arrested or charged? 1 Parole violation 2 Probation violation V0364
        self.parole_probation=['V0214','V0078','V0081','V0108','V0134','V0160','V0162','V0188','V0214','V0240','V0288','V0316','V0340']
        for variable in self.parole_probation:
            vals=self.subset[variable].value_counts(dropna=False).index
            drop=[val for val in vals if val not in [1,2]]
            self.subset[variable].replace(drop,0,inplace=True)

        parole=[]
        probation=[]
        for variable in self.parole_probation:
            self.subset.loc[self.subset[variable]==1, f'{variable}_probation']=1
            probation.append(f'{variable}_probation')
            self.subset.loc[self.subset[variable]==2, f'{variable}_parole']=1
            parole.append(f'{variable}_parole')

        self.subset['probation']=self.subset[probation].sum(axis=1)
        self.subset['parole']=self.subset[parole].sum(axis=1)
    
    def get_col_diff(self):
        l=set(self.subset.columns)-set(self.cols_in) 
        m=l-set(self.keep_list)
        return m

    def drop_cols(self):
        self.keep_list=list(self.new_code_cols.values())+list(self.new_type_cols.values())+['count_offenses','ctrl_off_code','ctrl_off_type','ctrl_count','arrest_date']+self.code_list+['parole','probation']
        self.to_drop=self.all_colnames+self.ctrl_cols+self.other_offense_variables
        diff=self.get_col_diff()
        final_drop=self.to_drop+list(diff)
        self.subset_out=self.subset
        self.subset_out.drop(final_drop,axis=1,inplace=True)
        print('These columns were created whilst transforming offense columns and are now being dropped')
        print(final_drop)
        #self.new_config_cols=set(self.subset_out.columns)-set(self.cols_in)

    def update_config(self,configdata):
        '''Updates to config dictionary stored under the dataset processor when passed a dictionary containing new variables'''
        new_config=pd.DataFrame.from_dict(configdata,orient='index')
        #print(new_config)
        self.config=pd.concat([self.config,new_config],axis=0)
        
        
    