# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:25:11 2022

@author: siobh
"""

import pickle 

# VO402- years sentenced to prison, single offence
# V0403- months sentenced to prison, single offence
# V0404- days sentenced to prison, single offence
single_flat = {'d': 'V0404', 'm': 'V0403', 'y': 'V0402'}

# V0405- min years sentenced to prison, single offence
# V0407- min months sentenced to prison, single offence
# V0409- min days sentenced to prison, single offence
single_range_min = {'d': 'V0409', 'm': 'V0407', 'y': 'V0405'}

# V0406- max years sentenced to prison, single offence
# V0408- max months sentenced to prison, single offence
# V0410-max days sentenced to prison, single offence
single_range_max = {'d': 'V0410', 'm': 'V0408', 'y': 'V0406'}

# V0413- max years, multi flat
# V0414- max months, multi flat
# V0415- max days, multi flat
multi_flat = {'d': 'V0415', 'm': 'V0414', 'y': 'V0413'}

# V0417, max sentence min years, multi range
# V04019- max sentence min months, multi range
# V0421-max sentence min days, multi range
multi_range_min = {'d': 'V0421', 'm': 'V0419', 'y': 'V0417'}

# V0418- max sentence max years, multi range
# V0420- max sentence max months, multi range
# V0422- max sentence max days, multi range
multi_range_max = {'d': 'V0422', 'm': 'V0420', 'y': 'V0418'}

# V0440- years, multi 2 flat
# V0441- months, multi 2 flat
# V0442- days, multi 2 flat
multi2_range_flat = {'d': 'V0442', 'm': 'V0441', 'y': 'V0440'}

# V0444- min years, multi 2 range
# V0446- min months, multi 2 range
# V0448- min days, multi 2 range
multi2_range_min = {'d': 'V0448', 'm': 'V0446', 'y': 'V0444'}

# V0445- max years, multi 2 range
# V0447- max months- multi 2 range
# V0449- max days, multi 2 range
multi2_range_max = {'d': 'V0449', 'm': 'V0447', 'y': 'V0445'}
# TODO : check that only one of the converted year columns is populated for each row

dmys = {'single_range_min': single_range_min, 'single_range_max': single_range_max, 'single_flat': single_flat,
             'multi_flat': multi_flat, 'multi_range_min': multi_range_min, 'multi_range_max': multi_range_max,
             'multi2_range_flat': multi2_range_flat, 'multi2_range_min': multi2_range_min, 'multi2_range_max': multi2_range_max}



'''
with open("sentence_length_variables","wb") as f:
    pickle.dump(dmys,f )

#CJ14- Inmate Type 1
#NCRP Codes V0140-V0144
#Offence type V0150-V0154

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
  
offense_variables={3:type3,8:type8,11:type11}

other_offense_variables=['V0246','V0247','V0248','V0249','V0322','V0323','V0324','V0325','V0326']

#Parole violators- original offenses?
#V0246
#V0247
#V0248
#V0249

#Probation violators original offense
#V0322
#V0323
#V0324
#V0325
#V0326

#Other fields containing offences of other inmate types
#CJ8A- Inmate Type 2
## name of offenses being held for. V0083-V0087 supressed
##NCRP code  V0088-V0089- populated
#Offense type V0098-V0102

#self.CJ8=['V0088','V0089','V0090','V0091','V0092']

#CJ23-CJ26 Inmate Types 4 and 6
#CJ26-Type6 and 4- offense_single is 'escape' if CJ26 Type=6 and it's 'parole violation' if CJ26TYPE=1,
#probation violation if CJ26 Type=2
      
#Cj17-Cj22- Inmate Type 5
#Type 5-CJ19
#If CJ19=4: parole violation, 1:probation violation, 

#CJA1-CJA2. Inmate Types 7 and 9
#Type 7: If CJA1 type= 1:parole violation, 2: probation violation
#Type 9
with open("offense_variables","wb") as f:
    pickle.dump(offense_variables,f )


offenses_config={'ctrl_off_code':{'enc_scale':'one_hot','description':'Controlling offense code collapsed','protected_characteristic':0,'include_violent_sent_predictor':1}, \
'count_offenses':{'enc_scale':'scale','description':'Count of all primary offenses','protected_characteristic':0,'include_violent_sent_predcitor':1}, \
'off_1_code': {'enc_scale':'one_hot','description':'primary offense 1 NCRP code','protected_characteristic':0,'include_violent_sent_predcitor':1},\
'off_1_type': {'enc_scale':'one_hot','description':'primary offense 1 NCRP type','protected_characteristic':0,'include_violent_sent_predcitor':1},\
'off_2_code': {'enc_scale':'one_hot','description':'primary offense 2 NCRP code','protected_characteristic':0,'include_violent_sent_predcitor':1},\
'off_2_type':{'enc_scale':'one_hot','description':'primary offense 2 NCRP type','protected_characteristic':0,'include_violent_sent_predcitor':1},\
'off_3_code':{'enc_scale':'one_hot','description':'primary offense 3 NCRP code','protected_characteristic':0,'include_violent_sent_predcitor':1},\
'off_3_type':{'enc_scale':'one_hot','description':'primary offense 3 NCRP type','protected_characteristic':0,'include_violent_sent_predcitor':1},\
'off_4_code':{'enc_scale':'one_hot','description':'primary offense 4 NCRP code','protected_characteristic':0,'include_violent_sent_predcitor':1},\
'off_4_type':{'enc_scale':'one_hot','description':'primary offense 4 NCRP type','protected_characteristic':0,'include_violent_sent_predcitor':1},\
'off_5_code':{'enc_scale':'one_hot','description':'primary offense 5 NCRP code','protected_characteristic':0,'include_violent_sent_predcitor':1},\
'off_5_type':{'enc_scale':'one_hot','description':'primary offense 5 NCRP type','protected_characteristic':0,'include_violent_sent_predcitor':1},\
}

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

parole_probation=['V0214','V0078','V0081','V0108','V0134','V0160','V0162','V0188','V0214','V0240','V0288','V0316','V0340']
for variable in parole_probation:
    vals=prisoners[variable].value_counts(dropna=False).index
    drop=[val for val in vals if val not in [1,2]]
    prisoners[variable].replace(drop,0,inplace=True)

parole=[]
probation=[]
for variable in parole_probation:
    prisoners.loc[prisoners[variable]==1, f'{variable}_probation']=1
    probation.append(f'{variable}_probation')
    prisoners.loc[prisoners[variable]==2, f'{variable}_parole']=1
    parole.append(f'{variable}_parole')

prisoners['probation']=prisoners[probation].sum(axis=1)
prisoners['parole']=prisoners[parole].sum(axis=1)
'''