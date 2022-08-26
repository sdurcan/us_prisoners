# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 13:49:34 2022

@author: siobh
"""
import os
os.chdir('C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers')
from src import load
import pandas as pd
import numpy as np
import copy
from src import sentence_transform
from src import offenses_config
from src import offense_transform


class offenses_transform():
    """Densifies offense data and returns dataset with original variables containing offense info removed"""
    
    def __init__(self,dp,ordinal=0):
        self.dataset=dp.dataset
        self.orig_cols=copy.deepcopy(self.dataset.columns)
        self.to_keep=[]
        self.dp=dp
        self.ordinal=ordinal
        
        self.load_inputs()
        self.transform()
        
        #TODO: put these back as static variables
        #sentence_transform.sentence_transformer.drop(self)
        #sentence_transform.sentence_transformer.update_config(self)
        
        #overwrite original dataset
        dp.dataset=self.dataset

        #TODO: decide how to deal with secondary offences that are not the controlling offenses
    
    def load_inputs(self):
        '''Sets attributes of offense_transformer'''
        #offense variables is a list of dicts. For each prisoner type, there is a dict with different offense variables
        
        self.current_off_cols=offenses_config.primary_offense_inputs_vars

        self.shared_info=self.dp.config.loc[self.dp.config['feeds_mutual_info']=='offense'].index
        
        #columns that contain the controlling offenses
        self.ctrl_offs=['V0428','V0429','V0430','V0431','V0432']

        #columns that contain information about the offenses previously on parole or probation for
        self.parole_probation=['V0214','V0078','V0081','V0108','V0134','V0160','V0162','V0188','V0214','V0240','V0288','V0316','V0340']
        
        self.configdata=offenses_config.offense_configdata
        
        #change to ordinal encoding
        if self.ordinal==1:
            for var, config in self.configdata.keys():
                config['enc_scale']='ordinal'
        
        for key in self.configdata:
            self.to_keep.append(key)
    
    def transform(self):
        self.replace_99_nan()
        self.set_ctrl_count()
        self.set_offence_count()
        self.set_ctrl_apply()
        self.set_ctrl_lookup_1()
        self.set_offense_counts()
        self.set_type_counts()
        self.collapse_offense_lists()
        self.set_off_lookup_one_violent_offense()
        
        
        
    def replace_99_nan(self):
        for col in self.current_off_cols:
            self.dataset[col].replace(99,np.nan,inplace=True)

        for col in self.ctrl_offs:
           self.dataset[col].replace(99,np.nan,inplace=True)
    
    def set_ctrl_count(self):
        #count how many of the variables holding controlling offense info have a value
        #WARNING: these will only be populated where there is more than one offense
        #And more than one controlling offense
        #And the sentence was not the same sentence for all offense
        #And the sentence was not one sentence for all offenses
        self.dataset['ctrl_count']=self[self.ctrl_offs].count(axis=1)
    
    def set_offense_count(self):
        #count row-ise to get count of offenses
        self.dataset['count_offenses']= self.dataset.count(axis=1)
    
    def set_ctrl_apply(self):
        ###SETTING HOW THE CONTROLLING OFFENSE HAS BEEN APPLIED
        #this value should be one hot encoded, not ordinal
        #to be used when providing counts of offense codes and offense types

        self.dataset.loc[self.dataset['count_offenses']==1,'ctrl_apply']='single_offense'

        self.dataset.loc[ ((self.dataset['ctrl_count']==1) & (self.dataset['count_offenses']>1)   ), 'ctrl_apply']='one_of_n'

        #see=temp[['ctrl_apply','ctrl_count','count_offenses']+current_off_cols+ctrl_offs]

        self.dataset.loc[self.dataset['V0433']==6,'ctrl_apply']='one_sentence_all'
        self.dataset.loc[self.dataset['V0434']==7,'ctrl_apply']='same_length_all'
        self.dataset.loc[self.dataset['V0435'].isin([7,8]),'ctrl_apply']='dk/ref'

        self.dataset.loc[ self.dataset['ctrl_count'].isin([2,3,4]), 'ctrl_apply']='2<n-1'
        #there are 33 entries where the controlling offence or controlling offence type doesn't appear to be coded correctly. 
        self.dataset['ctrl_apply'].replace(np.nan,'misc',inplace=True)
    
    def set_ctrl_lookup_1(self):
        #WHERE THERE IS ONLY ONE OFFENSE, ctrl lookup is offese 1
        #if ctrl_count is 0 and single_offenses is 1, make ctrl_count 1
        #ctrl_count will be 0 where there was only one offense because controlling offense does not need to be specified if only 1 offense
        self.dataset.loc[((self.dataset['ctrl_count']==0) & (self.dataset['count_offenses']==1)),'ctrl_lookup']=1

        #then set control lookup to 1- expect that surveyor has used first field to capture offense if there is a single offense
        self.dataset.loc[(self.dataset['ctrl_count']==1) & (self.dataset['count_offenses']==1),'ctrl_lookup']=1
        #however, if ctrl_count is 1 but there are multiple offenses, use the sum of the ctrl offenses
        #because other columns are blank (as there is only one offense counted), the sum output will be the offence # to look up 
        #WARNING: this only works because of how the columns are encoded in the originl dataset. 
        #In the N columns capturing controlling offenses, the column for offense i will be in column Ni and contain i
        #Therefore, if offense two is the controlling offense and controlling offenses are captured in columns v1...v5
        #Then, v2 will have a value of 2
        self.dataset.loc[(self.dataset['ctrl_count']==1) & (self.dataset['count_offenses']>1) ,'ctrl_lookup']=self.dataset[self.ctrl_offs].sum(axis=1)
    
    def set_offense_counts(self):
        #get all of the offence codes so that we can then do counts for them

        all_current_off_codes=[]
        for col in self.current_off_cols:
            all_current_off_codes=all_current_off_codes+self.dataset[col].replace(np.nan,0).unique().tolist()

        all_current_off_codes=set(all_current_off_codes)
        all_current_off_codes=[code for code in all_current_off_codes if code not in [0.0,999.0]]


        #create a dict where keys are index and vals are an inner dict of varname:val
        adict=self.dataset[self.current_off_cols].replace(np.nan,0).to_dict(orient='index')

        #create empty subdict for each index in a new dictionary
        new_rows={}
        for key, values in adict.items():
            new_rows[key]={}

        #then populate 0 for each offense count
        for key in new_rows.keys():
            for off_code in all_current_off_codes:
                new_rows[key][off_code]=0
            
        #then go through and populate with counts

        #go to each index
        for index in adict:
            #get a list of unique offense codes across columns
            aset=set(adict[index].values())
            #update the dictionary for each of these
            for value in aset:
                if value not in  [0.0,999.0]:
                    new_rows[index][value]=new_rows[index][value]+1

        #now need to turn back to df adn join on index
        temp_df=pd.DataFrame.from_dict(new_rows,orient='index')
        self.dataset=self.dataset.join(temp_df)

    def set_type_counts(self):
        #then we need to do the same thing with offence type columns
        #from here, whereever there is only 1 in the count for violent offense, this can be populated as the controlling offense
        off_type_cols=self.current_off_cols=list(type11['off_types'].values())+list(type8['off_types'].values())+list(type3['off_types'].values())

        off_types=[1.0, 4.0, 2.0, 3.0, 0.0, 5.0]

        bdict=self.dataset[off_type_cols].replace(np.nan,6666).to_dict(orient='index')

        #create empty subdict for each index
        tnew_rows={}
        for key, values in bdict.items():
            tnew_rows[key]={}

        #then populate 0 for each offense count
        for key in tnew_rows.keys():
            for off_type in off_types:
                tnew_rows[key][off_type]=0
            
        #then go through and populate with counts
        #almost there- need to get rid of nan

        #go to each index
        for index in bdict:
            #get a list of offense codes across columns
            bset=bdict[index].values()
            #update the dictionary for each of these
            #each valuewill be a violent offense type
            for value in bset:
                if value not in  [0.0,999.0,6666]:
                    #increase the count
                    tnew_rows[index][value]=tnew_rows[index][value]+1
                    
         
        temp_df=pd.DataFrame.from_dict(tnew_rows,orient='index')
        temp_df.drop(0,axis=1,inplace=True)
        self.dataset=self.dataset.join(temp_df)
    
    def collapse_offense_lists(self):
        """Combines all the controlling offense variables into one set of variables"""
        
        #TODO: the value counts of the new columns should match the value count offenses

        #get a list of prisoner types as this will change depending on the dataset
        pris_types=list(self.dataset['V0063'].value_counts().index)
     
        self.new_code_cols={}
        self.new_type_cols={}        

        
        for pt in pris_types:

            #assumes five offenses are listed in the survey

            for item in range(1,6):
      
                code_colname=self.primary_offense_inputs_vars[pt]['codes'][item]
                type_colname=self.primary_offense_inputs_vars[pt]['off_types'][item]
                self.dataset.loc[self.dataset['V0063'] == pt,f'off_{item}_code'] = self.dataset[code_colname]
                self.dataset.loc[self.dataset['V0063'] == pt, f'off_{item}_type'] = self.dataset[type_colname]

                #add the new cols to the dict so that we can keep them when we drop

                self.new_code_cols[item]= f'off_{item}_code'
                self.new_type_cols[item]=f'off_{item}_type'
                self.to_keep.append( f'off_{item}_code')
                self.to_keep.append(f'off_{item}_type')

    
    def set_off_lookup_one_violent_offense(self):
        #where the offense can be determined
        #go through each off type col and see if has a value of 1 for violent offense
        #if it does, then that code col becomes the lookup value

        for count, col in enumerate(self.new_type_cols.values()):
            self.dataset.loc[ ( (self.dataset['ctrl_apply'].isin(['one_sentence_all', 'misc','same_length_all','2<n-1','dk/ref'])) & (self.dataset[1.0]==1) & (self.dataset[col]==1.0)),'ctrl_lookup' ]=count+1
       
    def xtransform(self):
        self.flatten_colnames()
        self.set_single_offense()
        #self.check_offense_data()
        self.collapse_offense_lists()
        self.set_lookup_for_ctrl_off()
        self.set_single_ctrl_off()
        #self.encode_offense_codes()
        self.set_probation_parole_violation()
        

    def flatten_colnames(self):
        '''Creates a flat list of all columns that contain offense codes. 'This isn't combined within other functions 
        so that we can skip those functions'''
        #but still access a flat list of all colnames
        self.all_offense_colnames=[]
        for inmate_type, info in self.primary_offenses.items():
            codes=list(info['codes'].values())
            self.all_offense_colnames=self.all_offense_colnames+codes

    def check_offense_data(self):
        '''Runs checks on the offense calculations'''
        self.check_only_one_ctrl_type()

    def check_only_one_ctrl_type(self):
        """Checks in a list of lists that only one of the lists is populated"""
        codes=[]
        for inmate_type, info in self.primary_offense_inputs_vars.items():
            type_codes=list(info['codes'].values())
            codes.append(type_codes)
        self.check_one_per_group(self.dataset,codes)
        
    @staticmethod
    def one_if_any(dataset,colnames):
        """Takes a dataset and a list of column names and returns a boolean col
        indicating of any of those cols had a non nan value on that row"""
        #check if any of these variables are populated 
        return dataset[colnames].any(axis=1).to_frame()
        
    @staticmethod
    def check_one_per_group(dataset,listoflists):
        """Checks in a list of lists that only one of the lists is populated"""
        #sum over the codes to see if this is what we think it is
        count=0
        for varlist in listoflists: 
            #check if any of these variables are populated 
            any_in_group=offense_transform.one_if_any(dataset,varlist)
            
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




    def set_single_offense(self):
        """Creates a list of indexes of prisoners with multiple offenses
        and a list of single offenses. Informs how controlling offense is found"""

        #count across all offense columns
        #checks in offense_transform.check_one_per_group
        #should warn if there is an error in the data where more than one
        #group has a value
        self.dataset['count_offenses']=self.dataset[self.all_offense_colnames].count(axis=1)
                
        self.dataset.loc[self.dataset['count_offenses']==1,'single_offense']=1
        
    def set_lookup_for_ctrl_off(self, encode_categ_ctrl_count=2):
        '''Creates an intermediate lookup column in order to locate the controlling offense column. 
        When a prisoner has multiple offenses, the controlling offense is the offense with the longest sentence. 
        This will be indicated by the values in the below columns: 

        #V0428==1 : First Offense
        #V0429==2 : Second Offense
        #V0430==3 : Third Offense
        #V0431==4 : Fourth Offense
        #V0432==5 : Fifth Offense
        #V0433==6 : One sentence for all Offenses
        #V0434==7:  all sentences the same length
        #V0434 is all sentences are the same length
        
        If the prisoner recieved one sentence for all offenses OR their longest offence covered two offenses
        Then the controlling offense hierarchy is violent>property>drugs
        
        WARNING: Multiple controlling offenses is a different concept to multiple offenses. A prisoner can have
        multiple offenses but only one controlling offense. 
        
        '''
        #in this version, the values in the ctrl_cols are 99 or the offence
        #looping through like this because calling replace on list of columns doesn't update original
        #need nan values (rather than 0) so that count will work
        for col in self.ctrl_cols:
            self.dataset[col].replace(99,np.nan,inplace=True)
     
        #count row-ise to get count of controlling offense
        #TODO: check all prisoners with multiple offenses have provided a controlling offense
        self.dataset['ctrl_count']=self.dataset[self.ctrl_cols].count(axis=1)
        
        
        ##WHERE THERE IS ONE CONTROLLING OFFENSE
        #if ctrl_count is 0 and single_offenses is 1, make ctrl_count 1
        #ctrl_count is expected to be 0 because controlling offense does not need to be specified if only 1 offense
        self.dataset.loc[((self.dataset['ctrl_count']==0) & (self.dataset['single_offense']==1)),'ctrl_count']=1
        #POPULATE CONTROLLING OFFENSE LOOKUP BASED ON COUNT OF CONTROLLING OFFENSES
        #TODO: check, for single offenses, the first offense is always filled
        self.dataset.loc[(self.dataset['ctrl_count']==1) & (self.dataset['single_offense']==1),'ctrl_lookup']=1
        #if ctrl_count is 1 and they have multiple offenses, use the sum of the ctrl offenses
        #because other columns are blank (as there is only one offense counted), the sum output will be the offence # to look up 
        self.dataset.loc[(self.dataset['ctrl_count']==1) & (self.dataset['single_offense']!=1) ,'ctrl_lookup']=self.dataset[self.ctrl_cols].sum(axis=1)
        
        
        ##NO CONTROLLING OFFENSE- ONE SENTENCE OR ALL THE SAME
        #in instances where the prisoner recieved one sentence for all offenses or each offense recieved the same sentence
        #the controlling offense is captured differently.
        
        
        self.dataset.loc[self.dataset['V0433']==6,'ctrl_apply']='one_sentence_all'
        self.dataset.loc[self.dataset['V0434']==7,'ctrl_apply']='same_length_all'
        self.dataset.loc[self.dataset['V0435'].isin([7,8]),'ctrl_apply']='dk/ref'
        self.dataset.loc[self.dataset['count_offenses']==1,'ctrl_apply']='single'
        print('ctrl lookup before 2- n-1 offs')
        print(self.dataset['ctrl_apply'].value_counts(dropna=False))
        
        ###WHERE THERE ARE 2-4 controlling offenses
        #The rows with na are then those people with multiple controlling offenses- we need to do the lookup for rows and the rows one for all and all same
        #TODO: all of the ctrl_counts 2,3,4 have na in ctrl_lookup, which is correct
        #TODO: what about the other nas in ctrl_lookup? why? 33 rows
        #r=op.dataset.loc[ (op.dataset['ctrl_lookup'].isna()) & ~op.dataset['ctrl_count'].isin([2,3,4]) ]
        self.dataset.loc[(self.dataset['ctrl_lookup'].isna()) & (self.dataset['ctrl_count'].isin([2,3,4])),',missing_check']='multiple'
        
        '''
        if encode_categ_ctrl_count==1:
            
            self.dataset.loc[self.dataset['V0433']==6,'ctrl_count']='one for all'
            self.dataset.loc[self.dataset['V0434']==7,'ctrl_count']='all same'
            #there are 9s in this col aswell, but they represent blank values
            self.dataset.loc[self.dataset['V0435'].isin([7,8]),'ctrl_count']='dk/ref'
        
        elif encode_categ_ctrl_count==2:
            #set the control count to 5
            self.dataset.loc[self.dataset['V0433']==6,'ctrl_count']=5
            self.dataset.loc[self.dataset['V0434']==7,'ctrl_count']=5

        else:
            #create new column one_for_all_all_for same
            self.dataset.loc[self.dataset['V0433']==6,'one_for_all']=1
            self.dataset.loc[self.dataset['V0434']==7,'all_same']=1
            #there are 9s in this col aswell, but they represent blank values
            self.dataset.loc[self.dataset['V0435'].isin([7,8]),'dk_ref']=1
        '''
  
    def set_lookup_for_ctrl_off_multi2(self):

        #self.dataset.loc[self.dataset.loc['ctrl_count'].isin([2,3,4]),'offense_details']={}
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
        
        for index in self.dataset['ctrl_lookup'].value_counts().index:
            #TODO: work out how to not need this line
            if index not in ['dk/ref','all same','one for all']:
                #get the column name from the new_code_cols dictionary
                code=self.new_code_cols[index]
                type_col=self.new_type_cols[index]
                self.dataset.loc[self.dataset['ctrl_lookup']==index,'ctrl_off_code']= self.dataset[code]
                self.dataset.loc[self.dataset['ctrl_lookup']==index,'ctrl_off_type']= self.dataset[type_col]
                #TODO: type should always be violent (1) : violent
                                                                                  

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
        
        
        #create a smaller dataset to deal with
        #some offenses are held in variables that are not controlling offenses, so need to add them in
        self.other_offense_variables=['V0246','V0247','V0248','V0249','V0322','V0323','V0324','V0325','V0326']
        alloffs=self.all_colnames+self.other_offense_variables
        offense_cols_dataset=self.dataset[alloffs]
                
        #get a list of the codes across all offenses
        self.all_codes=[]
        for col_name in offense_cols_dataset.columns:
            new=list(offense_cols_dataset[col_name].unique())
            self.all_codes=self.all_codes+new
            self.all_codes=list(set(self.all_codes))
                
        self.all_codes=[code for code in self.all_codes if np.isnan(code)==False]
        
        self.code_list=[]
        for code in self.all_codes:
    
            #create a copy of values that are boolean wrt code
            c=copy.deepcopy(offense_cols_dataset==code)
            self.dataset[code]=copy.deepcopy(c.sum(axis=1))
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
       
        for variable in self.parole_probation:
            vals=self.dataset[variable].value_counts(dropna=False).index
            drop=[val for val in vals if val not in [1,2]]
            self.dataset[variable].replace(drop,0,inplace=True)

        parole=[]
        probation=[]
        for variable in self.parole_probation:
            self.dataset.loc[self.dataset[variable]==1, f'{variable}_probation']=1
            probation.append(f'{variable}_probation')
            self.dataset.loc[self.dataset[variable]==2, f'{variable}_parole']=1
            parole.append(f'{variable}_parole')

        self.dataset['probation']=self.dataset[probation].sum(axis=1)
        self.dataset['parole']=self.dataset[parole].sum(axis=1)
        
        self.to_keep.append('probation')
        self.to_keep.append('parole')
    
    '''
    def get_col_diff(self):
        l=set(self.dataset.columns)-set(self.cols_in) 
        m=l-set(self.keep_list)
        return m

    def drop_cols(self):
        self.keep_list=list(self.new_code_cols.values())+list(self.new_type_cols.values())+['count_offenses','ctrl_off_code','ctrl_off_type','ctrl_count','arrest_date']+self.code_list+['parole','probation']
        self.to_drop=self.all_colnames+self.ctrl_cols+self.other_offense_variables
        diff=self.get_col_diff()
        final_drop=self.to_drop+list(diff)
        self.dataset_out=self.dataset
        self.dataset_out.drop(final_drop,axis=1,inplace=True)
        print('These columns were created whilst transforming offense columns and are now being dropped')
        print(final_drop)
        #self.new_config_cols=set(self.dataset_out.columns)-set(self.cols_in)
    '''

        
        
    