# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:32:24 2022

@author: siobh
"""

#CJ11 Inmate Type 3
#No criminal justice status-sentenced
#NCRP codes V0114-V018
type3={'codes':{1:'V0114',2:'V0115',3:'V0116',4:'V0117',5:'V0118'},'off_types':{1:'V0124',2:'V0125',3:'V0126',4:'V0127',5:'V0128'},'type':'V0108'}
#Offence type V0124-V0128
#self.CJ11_type=['V0124','V0125','V0126','V0127','V0128']

#CJA3-CJA5- inmate type 8
#Parole violators, new offenses
#CJA4Type (V0264) ==1 :parole violation, 2: probation violation
type8={'codes':{1:'V0270',2:'V0271',3:'V0272',4:'V0273',5:'V0274'},'off_types':{1:'V0280',2:'V0281',3:'V0282',4:'V0283',5:'V0284'},'type':'V0264'}

#Inmate Type 11
#V0340 CJB6 TYPE= 2: probation violation, 1=parole violation, 
type11={'codes':{1:'V0346',2:'V0347',3:'V0348',4:'V0349',5:'V0350'},'off_types':{1:'V0356',2:'V0357',3:'V0358',4:'V0359',5:'V0360'},'type':'V0340'}

current_off_cols=list(type11['codes'].values())+list(type8['codes'].values())+list(type3['codes'].values())

ctrl_offs=['V0428','V0429','V0430','V0431','V0432']



for col in current_off_cols:
    temp[col].replace(99,np.nan,inplace=True)

for col in ctrl_offs:
    temp[col].replace(99,np.nan,inplace=True)
    
 
#count how many of the variables holding controlling offense info have a value
#WARNING: these will only be populated where there is more than one offense
#And more than one controlling offense
#And the sentence was not the same sentence for all offense
#And the sentence was not one sentence for all offenses
temp['ctrl_count']=temp[ctrl_offs].count(axis=1)
 
#count row-ise to get count of offenses
temp['count_offenses']=temp[current_off_cols].count(axis=1)

###SETTING HOW THE CONTROLLING OFFENSE HAS BEEN APPLIED
#this value should be one hot encoded, not ordinal
#to be used when providing counts of offense codes and offense types

temp.loc[temp['count_offenses']==1,'ctrl_apply']='single_offense'

temp.loc[ ((temp['ctrl_count']==1) & (temp['count_offenses']>1)   ), 'ctrl_apply']='one_of_n'

#see=temp[['ctrl_apply','ctrl_count','count_offenses']+current_off_cols+ctrl_offs]

temp.loc[temp['V0433']==6,'ctrl_apply']='one_sentence_all'
temp.loc[temp['V0434']==7,'ctrl_apply']='same_length_all'
temp.loc[temp['V0435'].isin([7,8]),'ctrl_apply']='dk/ref'

temp.loc[ temp['ctrl_count'].isin([2,3,4]), 'ctrl_apply']='2<n-1'
#there are 33 entries where the controlling offence or controlling offence type doesn't appear to be coded correctly. 
temp['ctrl_apply'].replace(np.nan,'misc',inplace=True)


#WHERE THERE IS ONLY ONE OFFENSE, ctrl lookup is offese 1
#if ctrl_count is 0 and single_offenses is 1, make ctrl_count 1
#ctrl_count will be 0 where there was only one offense because controlling offense does not need to be specified if only 1 offense
temp.loc[((temp['ctrl_count']==0) & (temp['count_offenses']==1)),'ctrl_lookup']=1


see=temp[['ctrl_lookup','ctrl_apply','ctrl_count','count_offenses']+current_off_cols+ctrl_offs]

#then set control lookup to 1- expect that surveyor has used first field to capture offense if there is a single offense
temp.loc[(temp['ctrl_count']==1) & (temp['count_offenses']==1),'ctrl_lookup']=1
#however, if ctrl_count is 1 but there are multiple offenses, use the sum of the ctrl offenses
#because other columns are blank (as there is only one offense counted), the sum output will be the offence # to look up 
#WARNING: this only works because of how the columns are encoded in the originl dataset. 
#In the N columns capturing controlling offenses, the column for offense i will be in column Ni and contain i
#Therefore, if offense two is the controlling offense and controlling offenses are captured in columns v1...v5
#Then, v2 will have a value of 2
temp.loc[(temp['ctrl_count']==1) & (temp['count_offenses']>1) ,'ctrl_lookup']=temp[ctrl_offs].sum(axis=1)


##MULTIPLE CONTROLLING OFFENSES- ONE SENTENCE FOR ALL, ALL THE SAME or 2-4 CONTROLLING OFFENSES
#in instances where the prisoner recieved one sentence for all offenses or each offense recieved the same sentence
#the controlling offense is captured differently.
#Where people have multiple controlling offenses and only one violent offense, the controlling offense will be that violent offense
#Where people have multiple controlling offenses and two or more violent offenses, it is unclear how to determine which offense data was collected about
violent_off_codes=[10,11,12,13,14,15,16,20,21,22,30,31,32,40,41,42,50,51,52,60,61,62,70,71,72,80,81,82,90,91,92,110,100,101,102,111,112,120,121,122,130,131,132,140,141,142,160,161,162,170,171,172,180]


#get all of the offence codes so that we can then do counts for them

all_current_off_codes=[]
for col in current_off_cols:
    all_current_off_codes=all_current_off_codes+temp[col].replace(np.nan,0).unique().tolist()

all_current_off_codes=set(all_current_off_codes)
all_current_off_codes=[code for code in all_current_off_codes if code not in [0.0,999.0]]


#create a dict where keys are index and vals are an inner dict of varname:val


adict=temp[current_off_cols].replace(np.nan,0).to_dict(orient='index')

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
test2=pd.DataFrame.from_dict(new_rows,orient='index')
temp=temp.join(test2)
to_see=[list(temp.columns[-115:])+current_off_cols+ctrl_offs]


for item in to_see:
    SEE=[thing for thing in item]
    
see=temp[SEE]



high_freq_codes=[90,10,120,70,480,11,40,190,50,180]

#then we need to do the same thing with offence type columns
#from here, whereever there is only 1 in the count for violent offense, this can be populated as the controlling offense
off_type_cols=current_off_cols=list(type11['off_types'].values())+list(type8['off_types'].values())+list(type3['off_types'].values())

off_types=[1.0, 4.0, 2.0, 3.0, 0.0, 5.0]

bdict=temp[off_type_cols].replace(np.nan,6666).to_dict(orient='index')

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
            
 
test3=pd.DataFrame.from_dict(tnew_rows,orient='index')
test3.drop(0,axis=1,inplace=True)
temp=temp.join(test3)

see=temp[SEE+off_types+'ctrl_off_1']

#where there are multiple controlling offenses and more than one violent offense
#we need to do some more work to determine violent offense
find_ctrl=see.loc [(~see['ctrl_apply'].isin(['single_offense','one_of_n'])) & (see[1.0]>1)]



violent_off_codes=[10,11,12,13,14,15,16,20,21,22,30,31,32,40,41,42,50,51,52,60,61,62,70,71,72,80,81,82,90,91,92,110,100,101,102,111,112,120,121,122,130,131,132,140,141,142,160,161,162,170,171,172,180]

c=set_ctrl_violent.columns
d=[col for col in c if col in violent_off_codes]


type3={'codes':{1:'V0114',2:'V0115',3:'V0116',4:'V0117',5:'V0118'},'off_types':{1:'V0124',2:'V0125',3:'V0126',4:'V0127',5:'V0128'},'type':'V0108'}
#Offence type V0124-V0128
#self.CJ11_type=['V0124','V0125','V0126','V0127','V0128']

#CJA3-CJA5- inmate type 8
#Parole violators, new offenses
#CJA4Type (V0264) ==1 :parole violation, 2: probation violation
type8={'codes':{1:'V0270',2:'V0271',3:'V0272',4:'V0273',5:'V0274'},'off_types':{1:'V0280',2:'V0281',3:'V0282',4:'V0283',5:'V0284'},'type':'V0264'}

#Inmate Type 11
#V0340 CJB6 TYPE= 2: probation violation, 1=parole violation, 
type11={'codes':{1:'V0346',2:'V0347',3:'V0348',4:'V0349',5:'V0350'},'off_types':{1:'V0356',2:'V0357',3:'V0358',4:'V0359',5:'V0360'},'type':'V0340'}
  
primary_offense_inputs_vars={3:type3,8:type8,11:type11}


"""Combines all the controlling offense variables into one set of variables"""

#TODO: the value counts of the new columns should match the value count offenses

#get a list of prisoner types as this will change depending on the dataset
pris_types=list(temp['V0063'].value_counts().index)
 
new_code_cols={}
new_type_cols={}        


for pt in pris_types:

    #assumes five offenses are listed in the survey

    for item in range(1,6):
  
        code_colname=primary_offense_inputs_vars[pt]['codes'][item]
        type_colname=primary_offense_inputs_vars[pt]['off_types'][item]
        temp.loc[temp['V0063'] == pt,f'off_{item}_code'] = temp[code_colname]
        temp.loc[temp['V0063'] == pt, f'off_{item}_type'] = temp[type_colname]

        #add the new cols to the dict so that we can keep them when we drop

        new_code_cols[item]= f'off_{item}_code'
        new_type_cols[item]=f'off_{item}_type'

see=temp[SEE+off_types+list(new_type_cols.values())+list(new_code_cols.values())]




#where the offense can be determined
#go through each off type col and see if has a value of 1 for violent offense
#if it does, then that code col becomes the lookup value

for count, col in enumerate(new_type_cols.values()):
    temp.loc[ ( (temp['ctrl_apply'].isin(['one_sentence_all', 'misc','same_length_all','2<n-1','dk/ref'])) & (temp[1.0]==1) & (temp[col]==1.0)),'ctrl_lookup' ]=count+1

see=temp[SEE+off_types+list(new_type_cols.values())+list(new_code_cols.values())]


for index in temp['ctrl_lookup'].value_counts().index:
    #TODO: work out how to not need this line
    if index != 999.0:
        #get the column name from the new_code_cols dictionary
        code=new_code_cols[index]
        type_col=new_type_cols[index]
        temp.loc[temp['ctrl_lookup']==index,'ctrl_off_code']=temp[code]
        temp.loc[temp['ctrl_lookup']==index,'ctrl_off_type']= temp[type_col]

#if there are only two violent offenses and they are all the same, then we can use this offense code for the control lookup
temp.loc[ ( (temp['ctrl_off_code']==6666) & (temp['off_1_code']== temp['off_2_code']) & (temp[1.0]==2)),'ctrl_lookup']=1
#we can do the same if there are 3 violent offenses that are all the same
temp.loc[ ( (temp['ctrl_off_code']==6666) & (temp['off_1_code']== temp['off_2_code']) & (temp['off_2_code']== temp['off_3_code']) & (temp[1.0]==3)),'ctrl_lookup']=1
#the same if there are four violent offenses
temp.loc[ ( (temp['ctrl_off_code']==6666) & (temp['off_1_code']== temp['off_2_code']) & (temp['off_2_code']== temp['off_3_code']) &  (temp['off_3_code']== temp['off_4_code'])& (temp[1.0]==4)),'ctrl_lookup']=1
#and five- but there are none with 5 offenss



#loose the 33 rows where there is a misc value

temp2=copy(temp)
temp.drop( (temp.loc[temp['ctrl_apply']=='misc'].index),axis=0,inplace=True)

#drop control off type- should always be 1.0

