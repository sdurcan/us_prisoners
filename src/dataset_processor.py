import os
import copy
import pandas as pd
from src import var_processor as vp
from sklearn.model_selection import train_test_split
#from src import sentence_creator as sc
#from sentence_transform as sentence_transform
from src import load
import numpy as np
import warnings
from src import offenses_config
from src import offense_transform
from src import sentence_transform
from src import victim_injuries_config


class dataset_processor:
    '''
    Collapses data, dervies now columns and applies one hot encoding and scaling to a dataset. 
    '''
    
    def __init__(self,dataset,config=[]):
        
        #dataset will usually be a subset of the prisoner data
        #config will be a dataframe of variables and how to treat them
        self.dataset_in=copy.deepcopy(dataset)
        #this attribute is updated throughout the processing
        self.dataset=dataset
        if config==[]:
            #returns a dataframe to pass into dataset_processor.encode_and_scale()
            self.config=load.import_starter_config()

    def set_sentence(self,th, print_th=1,inc_max_range=1, impute=1):
        '''Calls sentence_transform class. Updates self.dataset with a new dataframe containing sentence legnth variable
        Drops original sentence length variables from dataframe
        '''
        #don't need to worry about standardisation and normalisation here
        self.st=sentence_transform.sentence_transformer(self,th,inc_max_range)
        #get back the subset with the new column
        #the name for the column created as the target column is stored on the setnence_transfom object
        self.sent_th_name=self.st.sent_th_name
        if print_th==1:
            print('Sentence threshold variable is',self.sent_th_name, 'and distribution is')
            print(self.dataset[self.sent_th_name].value_counts(dropna=False))
        
    def set_offenses(self,low_freq_code_counts=0):
        
        self.ot=offense_transform.offenses_transform(self,low_freq_code_counts)
        self.dataset['ctrl_count']=self.dataset['ctrl_count'].astype(str)
        
    def set_offender_race(self):
        '''Creates binary variable victim_white'''
        
        #V1951-Spanish/Hispanic origin
        #V1952- White
        #V1953-Black/African American
        #V1954- American Indian/Alaska Native
        #V1955- Asian
        #V1957- Anyting else
        
        #replace don't know and refuse responses with most common value (1:male) and make skips their own category
        #replace 2 (not white) with 0
        self.dataset['offender_white']=copy.deepcopy(self.dataset['V1952'].replace([-9,-2,-1,np.nan,2],[1,1,1,-8,0]))
        
        self.dataset.drop(['V1951','V1953','V1954','V1955','V1957','V1952'],axis=1)
        
        
        #There are other columns describing how others would see the offenders race
        #Drop these apart from white for now- can experiment with them later
        #V0923- if others would describe offenders race as white
        
        #V0922- if others would describe offender as hispanic
        #V0924- if others would describe offenders race as Black or African American
        #V0924- if others would describe offender as American Indian or Alaska Native
        #V0926- if others would descrice race as Asian
        #V0927- if others would describe as Native Hawaian
        #V0928- if others would describe as something else
        self.dataset.drop(['V0923','V0924','V0925','V0926','V0927','V0928'],axis=1)
        new_config={'offender_white':{'enc_scale':'one_hot','description':'offender white or not','protected_characteristic':1,'include_violent_sent_predictor':1}}
        self.update_config(new_config)
        
        print('Offender race value counts')
        print(self.dataset['offender_white'].value_counts(dropna=False))
        
    @staticmethod
    def make_one_if_any(dataset,colnames,newname):
        # the binarised continuous columns and categoricl columns are included in the list
        sum_rows = dataset[colnames].sum(axis=1)

    def set_victim_injuries(self):
        
        #load the dictionary to input into the collapse binary function
        victim_injuries=victim_injuries_config.victim_injuries
        self.collapse_encode_binary(victim_injuries['to_collapse'])

        #create a config item for each need variable
        new_config={}
        for gname, colgroup in victim_injuries['to_collapse'].items():
            new_config[gname]=victim_injuries['config']

        shared_info=self.config.loc[self.config['feeds_mutual_info']=='victim_injuries'].index
        #the survey routing means that if an offenders crime is classed as murder or rape
        #victim died or rape_sexual_abuse fields are skipped.
        #'rape_sexual_assault' V0587==1, V0495==1
        self.dataset.loc[ ((self.dataset['V0587']==1) | (self.dataset['V0495']==1 )),'rape_sexual_assault']=1
        # 'victim_died'  V0587==2, V0495==2
        self.dataset.loc[ ((self.dataset['V0587']==2) | (self.dataset['V0495']==2 )),'victim_died']=1
        
        #drop the columns
        to_drop=[col for col in shared_info]
        self.dataset.drop(to_drop, axis=1,inplace=True)        
        #pass this to update config
        self.update_config(new_config)
    
    def set_victim_age(self):
        #V0555- 1 is under 12 years,2 is 12 to 17. 661 values where one of these is true. 232 where no answer has been given
        #V0490- victim age for single victim. 1 and 2. 1784 where this is true 284 dk/ref
        #collapse these columns to indicate if any victim was child
        self.dataset.loc[ ( (self.dataset['V0555']==1) | (self.dataset['V0490']==1)), 'victim_below_12']=1
        
        #drop columns because they contain the same info and we need to reduce dimensionality
        self.dataset.drop(['V0555','V0490'],inplace=True,axis=1)
        new_config={'victim_below_12':{'enc_scale':'one_hot','description':'victim below 12','protected_characteristic':0,'include_violent_sent_predictor':1}}
        self.update_config(new_config)

        
    #take a subset of the datframe with the relevant columns
    def collapse_encode_binary(self,collapser_input,negval=99):
        '''Takes a dict where one key is 'to_collapse' and value is a dict where keys are names for new categorical value
        and values are the columns to count. Also contains config for new column''' 
        
        for newcol, colgroup in collapser_input.items():
            #first replace negative values with np.nan so that count works
            for col in colgroup:
                self.dataset[col].replace(99,np.nan,inplace=True)
            self.dataset.loc [ self.dataset[colgroup].count(axis=1)>=1, newcol] = 1
    
    def set_offender_sex(self):
        #V1211- sex assigned at birth
        #V1212- self described sex identity
        #if self described sex identity is male, put as male (1), otherwise not male (0)
        self.dataset['offender_male']=copy.deepcopy(self.dataset['V1212'])
        self.dataset['offender_male'].replace([-9,-8,-2,-1,np.nan],[1,1,1,1,-8])
        self.dataset.loc [self.dataset['offender_male']!=1,'offender_male']=0
        print('Offender male value counts')
        print(self.dataset['offender_male'].value_counts())
        #we need to add this to the config
        new_config={'offender_male':{'enc_scale':'one_hot','description':'offender male or not male','protected_characteristic':1,'include_violent_sent_predictor':1}}
        self.update_config(new_config)

    def set_protected_attr(self):
        self.set_victim_race()
        self.set_offender_race()
        self.set_victim_sex()
        self.set_offender_sex()
    
    def set_victim_race(self):
        """Creates a variable called victim_white that could be used as a target variable. 
        Where there are multiple victims, the value is set to 1 if most victims where white"""

        ###Victim Race- Multiple Victims
        #V0543- any hispanic victims
        #V0545- any white victims
        #V0546- any black or african american victims
        #V0547- any victims american indian or alskan native
        #V0548- any asian victims
        #V0549- any hawaain or other pacific islander
        #V0550- don't know/ ref victim race
        
        #V0552- race of most victims
        ##1- most victims white
        ##2- most Black or African American
        ##3- most American Indian or Alaska Native
        ##4- most Asian
        ##5- most native hawaiin
        
        #V0481- hispanic victim (single victim)
        #V0482- white victim (single victim)
        #V0484- american indian or alaska native victim (single victim)
        #V0485- asian victim (single victim)
        #V0486- native hawaiin or pacific islander victim (single victim)
        #V0487- victim race dk/ref (single victim)

        #single victim white
        single_victim_white=self.dataset_in['V0482']==1
        #multiple victims mainly white
        multi_victims_white=self.dataset_in['V0552']==1
        white=pd.concat([single_victim_white,multi_victims_white],axis=1).sum(axis=1)
        white.name='victim_white'
        
        #could look at 'any victims' if needed later- for now, drop the cols
        any_victims_of_race=['V0543','V0545','V0546','V0547','V0548','V0549','V0550']
        self.dataset.drop(any_victims_of_race,inplace=True,axis=1)
        
        self.dataset=self.dataset.join(white)
        print('Victim white value counts')
        print(self.dataset['victim_white'].value_counts(dropna=False))
        
        #we need to add this to the config
        new_config={'victim_white':{'enc_scale':'one_hot','description':'Victim white or non white','protected_characteristic':0,'include_violent_sent_predictor':1}}
        self.update_config(new_config)
   
    def set_victim_sex(self):
        '''Creates a binary variable 'victim_male' that could be used as a target variable in a machine learning model'''
        #V0489- sex of victim (single victim)
        #V0554- sex of victims (multiple victims) majority male or female victims (multiple victims, both male and female)
        
        #SINGLE VICTIM
        #replace don't know and refuse answers with most common value (male) and make skips own category
        self.dataset['V0489']=self.dataset['V0489'].replace([-9,-2,-1,np.nan],[1,1,1,-8])
        self.dataset['V0553']=self.dataset['V0553'].replace([-9,-2,-1,np.nan],[1,1,1,-8])
        self.dataset['V0554']=self.dataset['V0554'].replace([-9,-2,-1,np.nan],[1,1,1,-8])
        
        #binarise as male or not male
        self.dataset.loc[ self.dataset['V0553']==1, 'victim_male']=1
        self.dataset.loc[ self.dataset['V0553']==3, 'victim_male']=1
        self.dataset.loc[ self.dataset['V0489']==1, 'victim_male']=1    
        self.dataset.loc[ self.dataset['victim_male']!=1, 'victim_male']=0
        print('Victim male value counts')
        print(self.dataset['victim_male'].value_counts(dropna=False))
        
        self.dataset.drop(['V0489','V0584','V0554'],axis=1)
        
        new_config={'victim_male':{'enc_scale':'one_hot','description':'victim male','protected_characteristic':0,'include_violent_sent_predictor':1}}
        self.update_config(new_config)

    def set_victim_relationship(self):
        '''Collasses victim relationship: 
            -spouse or ex-spouse (including boyfriend or girlfiend)
            -child (including stephcild) (does not indicate age legally considered adult, but relationship as offspring or step offspring of offender)
            -victim_other_well_known
            -victim_stranger
            -victim_sight_only
            -victim_casual_acq
        '''
        #For multiple victims, there could be a positive answer in multiple categories of relationship to victims
        #therefore each category has it's own column in the dataset
        
        #create an encoded column for victim victim spouse
        #so need to have a column for each grouping- so do ohe as part of pre processing
        #V0493 Single victim relationship at time of crime
        #1: spouse, 2: ex-spouse, 3:parent/step-parent, 4:own child, 5:stepchild, 6: sibling or step sibling, 7: other relative
        #8: boyfriend/girlfriend, 9: ex boyfriend/girlfriend, 10: friend/ex friend, 11:other
        #V0561 (spouse),#V0562 (ex spouse), V0569 (ex boyfriend or girlfriend), V0568 (boyfriend or girlfriend)
        self.dataset.loc [ ( (self.dataset['V0493'].isin([1,2,8,9])) | (self.dataset['V0561']==1) |  (self.dataset['V0562']==7) | (self.dataset['V0569']==9) | (self.dataset['V0568']==8) ),'victim_spouse']=1
        
        #create an ecoded column for victim being child (not necessarily under 18)
        #multiple victims: V0564 (child), V0565 (step child)
        #V0493, 5 (stepchild), 4 (own child)
        self.dataset.loc [( (self.dataset['V0493'].isin([4,5])) | (self.dataset['V0564']==4) |  (self.dataset['V0565']==5) ) ,'victim_child']=1
        
        #create an encoded column for other relationship
        #V0563, V0566, V0567, V0571
        self.dataset.loc [ ((self.dataset['V0493'].isin([6,7,10,11])) | (self.dataset['V0563']==3) |  (self.dataset['V0566']==6) | (self.dataset['V0567']==7) | (self.dataset['V0571']==11))   ,'victim_other_well_known']=1
        
        #Multiple victims
        #V0557=1 Knew All Victims, V0560=1- knew victims well 2- Some Known 3- All Strangers
        #V0558=1 Knew Victims by sight only
        #V0559=1 Knew Victims as casual acquantainces
        
        #Single Victim9
        #V0492 How well did you know the victims. 1=Sight Only, 2=Casual Acquaintance, 3=Well Known
        #V0491 Did you know the victim 1=Knew, 2=Stranger
        self.dataset.loc [ ((self.dataset['V0491']==2) | (self.dataset['V0557'].isin([2,3]))) ,'victim_stranger']=1
        
        #victim known by sight only    
        self.dataset.loc [ ((self.dataset['V0492']==1)| (self.dataset['V0558']==1)) ,'victim_sight_only']=1

        #victim casual acquaintance
        self.dataset.loc [((self.dataset['V0492']==2)| (self.dataset['V0559']==1 )),'victim_casualacq']=1
        
        
        #define config dictionary
        victim_relationship_config={'victim_spouse':{'enc_scale':'one_hot','description':'victim was a spouse or ex','protected_characteristic':0,'include_violent_sent_predictor':1,'derived':1},\
                                    'victim_child':{'enc_scale':'one_hot','description':'victim was a child or step-child of offender','protected_characteristic':0,'include_violent_sent_predictor':1,'derived':1},\
                                    'victim_other_known':{'enc_scale':'one_hot','description':'victim well known but not spouse or child','protected_characteristic':0,'include_violent_sent_predictor':1,'derived':1},\
                                    'victim_stranger':{'enc_scale':'one_hot','description':'victim was a stranger','protected_characteristic':0,'include_violent_sent_predictor':1,'derived':1},\
                                    'victim_sight_only':{'enc_scale':'one_hot','description':'victim only known by sight','protected_characteristic':0,'include_violent_sent_predictor':1,'derived':1},\
                                    'victim_casualacq':{'enc_scale':'one_hot','description':'victim was a casual acquaintance','protected_characteristic':0,'include_violent_sent_predictor':1,'derived':1}  }
        
        #update config
        self.update_config(victim_relationship_config)
        
        #drop original columns
        #TODO: drop columns marked as feeds: 'victim_relationship'
        self.dataset.drop(['V0491','V0492','V0493','V0557','V0558','V0559','V0561','V0562','V0563','V0566','V0567','V0571','V0569','V0568','V0493','V0564','V0565'],axis=1)
      
    def process(self,enc=1, scale=1, impute=1):
        '''Takes in a dataset after transformers have densified columns and applies either one_hot encoding
        or scaling.Builds dataset_out attribute attribute '''

        #print('Config type at encode and scale',type(self.config))
        count=0
        for colname in self.dataset.columns:
            if colname in self.config.index:
            

                if self.config['include_violent_sent_predictor'][colname]==1:
                    print('Encoding/Scaling',colname)                   
                    #cell var processor
                    #initiates object
                    #pass the dataset stored in the dataset_processor class
                    #this will have been updated with new columns in the transformation steps
                    processed_var=vp.var_processor(self.dataset, colname, self.config, enc=enc, scale=scale, impute=impute)

                    #if this is the first variable in the list, initiate encoded dataframe
                    if count==0:
                        #use the variable dataset_out
                        #we need to turn it into a dataframe to use .join later
                        self.dataset_out=pd.DataFrame(processed_var.output_col)
                        #print(colname, 'first')
                        count=1
                        
                    else:
                                            
                        self.dataset_out=self.dataset_out.join(processed_var.output_col)
                                        

    def check_output(self,colname):
        #the columns should sum to 1
        newcols= dataset_processor.encoded_colnames(self.dataset,colname)
        if self.check_rows_sum_1(self.dataset,newcols)==False:
            print(colname,'encoded columns do not sum to 1')

    
    def update_config(self,configdata):
        '''Updates to config dictionary stored under the dataset processor when passed a dictionary containing new variables'''
        new_config=pd.DataFrame.from_dict(configdata,orient='index')
        #print(new_config)
        self.config=pd.concat([self.config,new_config],axis=0)


