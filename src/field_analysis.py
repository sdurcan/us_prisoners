# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 22:58:30 2022

@author: siobh
"""
#V055Y < 41.5
#off code 10
#V0560 Yes, knew the victims well
#off2 type 1
#rape sexual assault
#victim child
#victim died
#ctrl off code 70
#V0479- how many people helped with the offense
#off 3 type
#V0557- knew victims, all known
#V0917- date of first admission to prison
#V0262-Year arrested or charged with CJA1 offense - Parole violators
#V0362 Year arrested or charged with CJB6 offense - Probation violators, new offenses
#V0461 sex offenders treatment
#V0460 does sentence include drug or alcohol treatment

#sklearn.metrics.classification_report(y_true, y_pred

#Get previous best decision tree map back

#Create dataset with:
#Flag for previous probation and parole violation (remove year)
#Arrest year
#Offence types
#Controlling offense type (just one)
#Victim child (consolidate info from single and multi victims)
#Victim died
##V0888 if anyone that was shot died
#Sex offenders treatment
#Drug or alcohol treatment
#V0459-1- yes, sentence includes mandatory drug testing
#V0495- type of violent offense if one victim, V0587- type of violent offense if more than 1 victim, and V0069 should be both? 
#there are some without a violent type given?
#t=subset.loc[((~subset['V0495'].isin([-1,-2,1.0,2.0,3.0,4.0,5.0])) & (~subset['V0587'].isin([-1,-2,1.0,2.0,3.0,4.0,5.0]))),['V0069','V0587','V0495']]
#10,70,90,120

#ctrl off 70 (sexual assault-other)
#ctrl off 90 (armed robbery)
#ctrl off 120 (aggravated assault)

#count for off 10, 90, 120, 70

#V0461- sex offenders included in sentence
#V0888 if anyone that was shot died
#V0887-1 Yes, someone was shot during offense
#V0891-2- No, weapon was not used to get away

#V0779- number of weapons carried during offense


#violent type analysis
#in theory, as the subset is only looking at violent crimes, violent type should always be populated
#in some cases it is not. There is no flag or warning on these varaibles. 
#527 with no violent type
#468 rows where the data is missing
#59 where the answer given was either -1 or -2

#victim age analysis
#V0555- 1 is under 12 years,2 is 12 to 17. 661 values where one of these is true. 232 where no answer has been given
#V0490- victim age for single victim. 1 and 2. 1784 where this is true 284 dk/ref
#V0480 indicates if there was single or multiple victims

'''
IF CONTROLLING_TYPE = 1 AND THERE IS A MATCH OF LITERALS WITH
“RAPE” FROM LOOKUP TABLE THEN VIOLENT_TYPE = 1 (RAPE)
IF CONTROLLING_TYPE = 1 AND THERE IS A MATCH OF LITERALS WITH
“MURDER”, “MANSLAUGHTER”, OR “HOMICIDE” FROM LOOKUP TABLE
THEN VIOLENT_TYPE = 2 (MURDER/MANSLAUGHTER/HOMOCIDE)
IF CONTROLLING_TYPE = 1 AND THERE IS NO MATCH OF LITERALS WITH
“RAPE”, “MURDER”, “MANSLAUGHTER”, OR “HOMICIDE” FROM LOOKUP
TABLE THEN VIOLENT_TYPE = EMPTY
'''


#67 offenses for those with one for all or all one
pp=['parole','probation','V0246','V0247','V0248','V0249','V0322','V0323','V0324','V0325','V0326']
pp_offs=subset[pp]
pp_offs['parole'].replace([1,2],'Parole',inplace=True)
pp_offs['probation'].replace([1,2],'Probation',inplace=True)

#create df
#need to drop na
df=pd.DataFrame(dp.dataset['off_1_code'].value_counts())
for thing in ['off_2_code','off_3_code','off_4_code','off_5_code','V0427','V0248','V0249','V0322','V0323','V0324','V0325','V0326',]:
    df=df.join(subset[thing].value_counts())
df['sum']=df.sum(axis=1)


#plotting distribution of offenses
import matplotlib.pyplot as plt
#df where 'sum' is count of offenses and index is offense codes
#sort dataframe by frequency
df.sort_values('sum',inplace=True, ascending=False)
y=df['sum']
x=df.index.astype(int).astype(str)
fig, ax = plt.subplots(figsize=(30, 10))
plt.xlabel('Offense code', labelpad=20)
plt.ylabel('Frequency')
plt.title('Offense code distribution - no controlling offence type')
ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
plt.savefig('Offense code distribution - parole and probation violators original offenses',dpi=200)


#for the most common offense types, count how many offenses are in current set of oddenses
