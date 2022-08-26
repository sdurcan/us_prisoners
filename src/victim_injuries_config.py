# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 16:46:12 2022

@author: siobh
"""

#what do do about DK/REF injuries
#check child victim fields
#V0620 not physically injured
#check murder and rape is represtented accurately
#V0590 rape is sexual assault injurity
#V0593 teeth knoched out

#Murder, Rape, Other
#single victim- V0495
#multiple victims- V0587

#Victim died
#the values are 99 or 0
#multiple victims- other violent offense V0589
#single victims- other violent offense V0497

#Rape or sexual abuse
#multiple victims- other violent offense V0590

victim_injuries={\

#variables close together because child and step child
#injuries will appear in multiple sections
#only two values for rape sexual assault
#only two values for teeth knocked out or chipped

#Single victim, rape
##Single victim, other violent offense
##Multiple victims, rape
##Multiple victims, other violent offense

#V0069- violent type. 1=Rape, 2=Murder, 3=Other
#V0495 classify offense as Rape, Murder, Other
#V0587 classify offense as Rape, Murder, Other

'config':{'enc_scale':'one_hot','description':'victim injuries collapsed','protected_characteristic':0,'include_violent_sent_predictor':1},

'to_collapse':{'broken_bones': ['V0593','V0615','V0501','V0523'],\

'bruises_swelling': ['V0597','V0618','V0505','V0526'],\
    
'gunshot_bullet':['V0592','V0614','V0500','V0522'],\

'internal_injuries':['V0595','V0616','V0503','V0524'],\
    
'knife_stab':['V0591','V0613','V0499','V0521'],\
    
'other_injuries':['V0598','V0619','V0506','V0527'],\
    
'knocked_unconscious':['V0596','V0617','V0504','V0525'],\
    
'victim_died':['V0589','V0497'],\
    
'not_physically_injured':['V0620','V0528'],\
    
'rape_sexual_assault':['V0590','V0498','V0510','V0602'],\
    
'teeth_chipped':['V0594','V0502'] }  }

'''
utils.name_and_pickle(victim_injuries,'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/processing_config','victim_injuries',ext='pkl')

#SINGLE VICTIM
#this code shows that, if violent_tpe in variable V096 is 2 (rape), then rape is not captured as a victim injury
#not raped
#prisoners.loc[prisoners['V0495']==1,'V0498'].value_counts(dropna=False)
#not injured
#prisoners.loc[prisoners['V0495']==1,'V0496'].value_counts(dropna=False)
#murder
#violent offense murder but victim not 'injured'
#prisoners.loc[prisoners['V0495']==2,'V0496'].value_counts(dropna=False)
#murdered but didn't die
#prisoners.loc[prisoners['V0495']==2,'V0497'].value_counts(dropna=False)

##MULTIPLE VICTIMS
#raped but not hurt
#prisoners.loc[prisoners['V0587']==1,'V0588'].value_counts(dropna=False)
#murdered but not hurt
#prisoners.loc[prisoners['V0587']==2,'V0588'].value_counts(dropna=False)
#raped but not raped or sexually abused
#prisoners.loc[prisoners['V0587']==1,'V0590'].value_counts(dropna=False)
#murdered but didn't die
#prisoners.loc[prisoners['V0587']==2,'V0509'].value_counts(dropna=False)

#Update injuries
#'rape_sexual_assault' V0587==1, V0495==1
# 'victim_died'  V0587==2, V0495==2

def victim_injuries(path=""):
    
    if path=="":
        path=r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/processing_config/victim_injuries.pkl'
    
    victim_injuries=pickle.load( open( path, "rb" ) )
    return victim_injuries
'''

