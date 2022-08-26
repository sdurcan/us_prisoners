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
    
    def __init__(self,dp,low_freq_code_counts=0):
        self.low_freq_code_counts=low_freq_code_counts
        self.dataset=copy.deepcopy(dp.dataset)
        self.orig_cols=copy.deepcopy(self.dataset.columns)
        self.to_keep=[]
        self.dp=dp
        self.load_inputs()
        self.transform()

        
        #TODO: put these back as static variables
        sentence_transform.sentence_transformer.drop(self)
        sentence_transform.sentence_transformer.update_config(self)
        self.dataset['ctrl_count'].value_counts(dropna=False)
        #overwrite original dataset
        dp.dataset=copy.deepcopy(self.dataset)
    
    def load_inputs(self):
        '''Sets attributes of offense_transformer'''
        #offense variables is a list of dicts. For each prisoner type, there is a dict with different offense variables
        
        self.primary_offenses=offenses_config.primary_offense_inputs_vars

        self.shared_info=self.dp.config.loc[self.dp.config['feeds_mutual_info']=='offense'].index
        
        #columns that contain the controlling offenses
        self.ctrl_offs=['V0428','V0429','V0430','V0431','V0432']

        #columns that contain information about the offenses previously on parole or probation for
        self.parole_probation=['V0214','V0078','V0081','V0108','V0134','V0160','V0162','V0188','V0240','V0288','V0316','V0340']
        
        self.configdata=offenses_config.offense_configdata
                
        #don't need to append anything to 'to keep' if defined in config data
        for key in self.configdata:
            self.to_keep.append(key)
    
    def transform(self):
        self.flatten_colnames()
        self.replace_99_nan()
        self.set_ctrl_count()
        self.set_offense_count()
        self.set_ctrl_apply()

        self.set_offense_code_counts()
        self.set_type_counts()
        print('Collapsing offense lists')
        self.collapse_offense_lists()

        print('Setting ctrl lookup 1')
        self.set_ctrl_lookup_1()

        print('Looking up controlling offense')
        self.lookup_ctrl_offense()
        
        print('Finding further controlling offenses')
        self.set_ctrl_lookup_one_violent_offense()
        self.set_ctrl_lookup_multiple_violent_off_same_code()
 
        print('Setting parole and probation violation')
        self.set_probation_parole_violation()
        print('Searching for violent type')
        self.set_violent_type()
        print('Getting parole and probation offense counts')
        self.collapse_probation_parole_offense_lists()
        
        print('Dropping low frequency offense codes')
        self.drop_low_freq_code_counts()


    def flatten_colnames(self):
        '''Creates a flat list of all columns that contain offense codes. 'This isn't combined within other functions 
        so that we can skip those functions'''
        #but still access a flat list of all colnames
        self.current_off_cols=[]
        for inmate_type, info in self.primary_offenses.items():
            codes=list(info['codes'].values())
            self.current_off_cols=self.current_off_cols+codes
        
        self.current_type_cols=[]
        for inmate_type, info in self.primary_offenses.items():
            types=list(info['off_types'].values())
            self.current_type_cols=self.current_type_cols+types
        
    def replace_99_nan(self):
        for col in self.current_off_cols:
            self.dataset[col].replace(99,np.nan,inplace=True)

        for col in self.ctrl_offs:
           self.dataset[col].replace(99,np.nan,inplace=True)
    
    def set_ctrl_count(self):
        #count how many of the variables holding controlling offense lookup have a value
        #WARNING: these will only be populated where:
        #There is more than one offense 
        #And more than one controlling offense
        #And the sentence was not the same sentence for all offense
        #And the sentence was not one sentence for all offenses
        self.dataset['ctrl_count']=self.dataset[self.ctrl_offs].count(axis=1)
    
    def set_offense_count(self):
        #count row-ise to get count of offenses
        self.dataset['count_offenses']= self.dataset[self.current_off_cols].count(axis=1)
    
    def set_ctrl_apply(self):
        print('Setting control apply type')
        ###SETTING HOW THE CONTROLLING OFFENSE HAS BEEN APPLIED
        #this value should be one hot encoded, not ordinal
        #to be used when providing counts of offense codes and offense types

        self.dataset.loc[self.dataset['count_offenses']==1,'ctrl_apply']='single_offense'

        self.dataset.loc[ ((self.dataset['ctrl_count']==1) & (self.dataset['count_offenses']>1)   ), 'ctrl_apply']='one_of_n'

        self.dataset.loc[self.dataset['V0433']==6,'ctrl_apply']='one_sentence_all'
        self.dataset.loc[self.dataset['V0434']==7,'ctrl_apply']='same_length_all'
        self.dataset.loc[self.dataset['V0435'].isin([7,8,-9]),'ctrl_apply']='dk/ref/issue'

        self.dataset.loc[ self.dataset['ctrl_count'].isin([2,3,4]), 'ctrl_apply']='2<n-1'
        #there are 33 entries where the controlling offence or controlling offence type doesn't appear to be coded correctly. 
        self.dataset['ctrl_apply'].replace(np.nan,'misc',inplace=True)
    
    def set_ctrl_lookup_1(self):
        print('Setting control lookup1')
        '''Creates an intermediate lookup column in order to locate the controlling offense column. 
        This will be correct only where there is one controlling offense

        #V0428==1 : First Offense
        #V0429==2 : Second Offense
        #V0430==3 : Third Offense
        #V0431==4 : Fourth Offense
        #V0432==5 : Fifth Offense
        #V0433==6 : One sentence for all Offenses
        #V0434==7:  all sentences the same length
        #V0434 is all sentences are the same length
              
        WARNING: Multiple controlling offenses is a different concept to multiple offenses. A prisoner can have
        multiple offenses but only one controlling offense. 
        
        '''
        
        #WHERE THERE IS ONLY ONE OFFENSE, ctrl lookup is offense 1
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
    
    def set_offense_code_counts(self):
        #get all of the offence codes so that we can then do counts for them
        print('Setting offense code counts')


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
        self.new_offense_count_cols=temp_df.columns
        self.dataset=self.dataset.join(temp_df)
        
        for col in self.new_offense_count_cols:
            self.configdata[col]={'enc_scale':'one_hot','description':'count of offense code','protected_characteristic':0,'include_violent_sent_predictor':1}
            self.to_keep.append(col)
            
    def set_type_counts(self):

        #then we need to do the same thing with offence type columns
        #from here, whereever there is only 1 in the count for violent offense, this can be populated as the controlling offense
        off_type_cols=self.current_type_cols

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
        
        self.new_type_count_cols=temp_df.columns
        
     
        self.dataset=self.dataset.join(temp_df)
    
    def collapse_offense_lists(self):
        """Combines all the controlling offense variables into one set of variables. This is then used to lookup the controlling offense"""
        self.dataset=self.dataset.copy()
        #TODO: the value counts of the new columns should match the value count offenses

        #get a list of prisoner types as this will change depending on the dataset
        pris_types=list(self.dataset['V0063'].value_counts().index)
     
        self.new_code_cols={}
        self.new_type_cols={}        

        
        for pt in pris_types:

            #assumes five offenses are listed in the survey

            for item in range(1,6):
      
                code_colname=self.primary_offenses[pt]['codes'][item]
                type_colname=self.primary_offenses[pt]['off_types'][item]
                self.dataset.loc[self.dataset['V0063'] == pt,f'off_{item}_code'] = self.dataset[code_colname]
                self.dataset.loc[self.dataset['V0063'] == pt, f'off_{item}_type'] = self.dataset[type_colname]
                self.dataset=self.dataset.copy()
                #add the new cols to the dict so that we can keep them when we drop

                self.new_code_cols[item]= f'off_{item}_code'
                self.new_type_cols[item]=f'off_{item}_type'
                self.to_keep.append( f'off_{item}_code')
                self.to_keep.append(f'off_{item}_type')

    
    def set_ctrl_lookup_one_violent_offense(self):
        #violent offense>property offense>drug offense>po offense>other offens  
        #Offense multi logic- if the above sum to more than 1 on a row, then we
        #need to work out the controlling offense so that we can get the offense code
        #1- violent offense
        #get violent offenses and pick one, then map to code
        #where the offense can be determined
        #go through each off type col and see if has a value of 1 for violent offense
        #if it does, then that code col becomes the lookup value

        for count, col in enumerate(self.new_type_cols.values()):
          self.dataset=self.dataset.copy()
          self.dataset.loc[ ( (self.dataset['ctrl_apply'].isin(['one_sentence_all', 'misc','same_length_all','2<n-1','dk/ref'])) & (self.dataset[1.0]==1) & (self.dataset[col]==1.0)),'ctrl_lookup' ]=count+1
       
    
    def set_ctrl_lookup_multiple_violent_off_same_code(self):
        self.dataset=self.dataset.copy()
        #if there are only two violent offenses and they are all the same, then we can use this offense code for the control lookup
        self.dataset.loc[ ( ( self.dataset['ctrl_off_code']==6666) & ( self.dataset['off_1_code']==  self.dataset['off_2_code']) & ( self.dataset[1.0]==2)),'ctrl_lookup']=1
        #we can do the same if there are 3 violent offenses that are all the same
        self.dataset.loc[ ( ( self.dataset['ctrl_off_code']==6666) & ( self.dataset['off_1_code']==  self.dataset['off_2_code']) & ( self.dataset['off_2_code']==  self.dataset['off_3_code']) & ( self.dataset[1.0]==3)),'ctrl_lookup']=1
        #the same if there are four violent offenses
        self.dataset.loc[ ( ( self.dataset['ctrl_off_code']==6666) & ( self.dataset['off_1_code']==  self.dataset['off_2_code']) & ( self.dataset['off_2_code']==  self.dataset['off_3_code']) &  ( self.dataset['off_3_code']==  self.dataset['off_4_code'])& ( self.dataset[1.0]==4)),'ctrl_lookup']=1
        #and five- but there are none with 5 offenss
    
    def lookup_ctrl_offense(self):
        '''Populates ctrl_off and ctrl_type according to lookup value in ctrl_count'''
        for index in  self.dataset['ctrl_lookup'].value_counts().index:
            #TODO: work out how to not need this line
            if index != 999.0:
                #get the column name from the new_code_cols dictionary
                code=self.new_code_cols[index]
                type_col=self.new_type_cols[index]
                self.dataset=self.dataset.copy()
                self.dataset.loc[ self.dataset['ctrl_lookup']==index,'ctrl_off_code']= self.dataset[code]
                self.dataset.loc[ self.dataset['ctrl_lookup']==index,'ctrl_off_type']=  self.dataset[type_col]
    
    def rename_type_counts(self):
        off_mapping={1.0:'violent_current_count',2.0:'property_current_count',3.0:'drug_current_count',4.0:'public_order_current_count',5.0:'other_current_count'}
        self.dataset.rename(columns=off_mapping,inplace=True)                                                                              
   
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
        
        self.dataset=self.dataset.copy()
        for variable in self.parole_probation:
            vals=self.dataset[variable].value_counts(dropna=False).index
            #crude method of dropping dk, ref and actual offenses given
            #in the probaiton and parole variables, 1 is always probation and 2 is always parole
            drop=[val for val in vals if val not in [1,2]]
            #fill all the other values with 0
            self.dataset[variable].replace(drop,0,inplace=True)

        #then create new columns that count the number of parole and probation offenses
        parole=[]
        probation=[]
        
        for variable in self.parole_probation:
            #self.dataset=self.dataset.copy()
            self.dataset.loc[self.dataset[variable]==1, f'{variable}_probation']=1
            probation.append(f'{variable}_probation')
            self.dataset.loc[self.dataset[variable]==2, f'{variable}_parole']=1
            parole.append(f'{variable}_parole')


        self.dataset['probation']=self.dataset[probation].sum(axis=1)
        self.dataset['parole']=self.dataset[parole].sum(axis=1)
        
        self.to_keep.append('probation')
        self.to_keep.append('parole')
        print('parole value counts')
        print(self.dataset['parole'].value_counts(dropna=False))
        print('probation value counts')
        print(self.dataset['probation'].value_counts(dropna=False))
        
    def collapse_probation_parole_offense_lists(self):
        """Combines all the controlling offense variables into one set of variables. This is then used to lookup the controlling offense"""
                
        #get all of the offence codes so that we can then do counts for them
        all_previous_off_codes=[]
        for col in self.parole_probation:
            all_previous_off_codes=all_previous_off_codes+self.dataset[col].replace(np.nan,0).unique().tolist()

        all_previous_off_codes=set(all_previous_off_codes)
        all_previous_off_codes=[code for code in all_previous_off_codes if code not in [0.0,999.0]]

        #create a dict where keys are index and vals are an inner dict of varname:val
        cdict=self.dataset[self.parole_probation].replace(np.nan,0).to_dict(orient='index')

        #create empty subdict for each index in a new dictionary
        new_rows={}
        for key, values in cdict.items():
            new_rows[key]={}

        #then populate 0 for each offense count
        for key in new_rows.keys():
            for off_code in all_previous_off_codes:
                new_rows[key][off_code]=0
            
        #then go through and populate with counts

        #go to each index
        for index in cdict:
            #get a list of unique offense codes across columns
            aset=set(cdict[index].values())
            #update the dictionary for each of these
            for value in aset:
                if value not in  [0.0,999.0]:
                    new_rows[index][value]=new_rows[index][value]+1

        #now need to turn back to df adn join on index
        temp_df=pd.DataFrame.from_dict(new_rows,orient='index')

        #replace columns
        temp_df.columns=[f'{col}_pp' for col in temp_df.columns]

        self.new_previous_offense_count_cols=temp_df.columns

        print([col for col in temp_df.columns if col in self.dataset.columns])
        self.dataset=self.dataset.join(temp_df)

        for col in self.new_previous_offense_count_cols:
            self.configdata[col]={'enc_scale':'one_hot','description':'count of previous offense code (parole and probation offenses)','protected_characteristic':0,'include_violent_sent_predictor':1}
            self.to_keep.append(col)
       

    def set_violent_type(self):
        #in theory, as the subset is only looking at violent crimes, violent type should always be populated
        #in some cases it is not. There is no flag or warning on these varaibles. 
        #527 with no violent type
        #468 rows where the data is missing
        #59 where the answer given was either -1 or -2
        #the lookup table isn't provided in the user guide
        
        #we can probably determine a violent type where we have a controlling offense code?
        #l=prepped.loc [ (prepped['v']>0) & (prepped[V0069].sum(axis=1)<1)]
        #l1['sum']=l1.sum(axis=1)
        #it seems like most of these only have one off code? so we can look this up?
        #but it may be that we've dropped the other columns relevant to this by dropping low frequency codes
        #where is ctrl_apply single or one of n AND has a blank violent type? this is easy to correcg
        
       
        #print(t.head())

        #t.to_csv('check.csv')
        #this gves 381 rows- so it worth doing
        
        violent_type_mapping={90: 3.0,
         70: 2.0,
         120: 3.0,
         10: 1.0,
         80: 2.0,
         180: 3.0,
         92: 3.0,
         50: 2.0,
         100: 3.0,
         860: 3.0,
         140: 3.0,
         150: 3.0,
         40: 3.0,
         60: 2.0,
         91: 3.0,
         170: 3.0,
         11: 1.0,
         14: 1.0,
         30: 1.0,
         51: 2.0,
         71: 2.0,
         12: 1.0,
         20: 1.0,
         640: 3.0,
         81: 2.0,
         15: 1.0,
         121: 3.0,
         41: 3.0,
         110: 2.0,
         13: 1.0,
         480: 3.0,
         565: 3.0}
        
            
        for code, vtype in violent_type_mapping.items():
            self.dataset.loc[ ((~self.dataset['V0069'].isin([1.0,2.0,3.0])) & self.dataset['ctrl_apply'].isin(['single_offense','one_of_n'])),'V0069']=vtype
        
        #t=self.dataset.loc[ ((~self.dataset['V0069'].isin([1.0,2.0,3.0])) & self.dataset['ctrl_apply'].isin(['single_offense','one_of_n'])),['ctrl_off_type','ctrl_off_code']]
    
    def drop_low_freq_code_counts(self):
        high_freq_codes=[90.0,10.0,120.0,70.0,480.0,11.0,40.0,190.0,50.0,180.0]
        #combine non high freq code count
        low_freq=[col for col in self.new_offense_count_cols if col not in high_freq_codes]
        self.dataset['low_freq_codes_sum']=self.dataset[low_freq].sum(axis=0)
        self.dataset.drop(low_freq, axis=1,inplace=True)
        self.to_keep.append('low_freq_codes_sum')
        
        pp_high_freq=['120_pp','90_pp','190_pp','480_pp','250_pp','410_pp','370_pp','220_pp','565_pp']
        low_freq_pp=[col for col in self.new_previous_offense_count_cols if col not in pp_high_freq]
        self.dataset['pp_low_freq_codes_sum']=self.dataset[low_freq_pp].sum(axis=0)
        self.dataset['pp_low_freq_codes_sum'].replace(np.nan,0)
        self.dataset.drop(low_freq_pp, axis=1,inplace=True)
        self.to_keep.append('pp_low_freq_codes_sum')
        
        high_freq_ctrl=[10.0,90.0,120.0,70.0,50.0,11.0,30.0,80.0,40.0,13.0,60.0,180.0]
        
        
        self.dataset.loc[(~self.dataset['ctrl_off_code'].isin(high_freq_ctrl)),'ctrl_off_code']=998.0
        


        
        
    