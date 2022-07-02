# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:28:33 2022

@author: siobh
"""


#dataset is a dataframe
#offense_columns is a list of strings corresponding to column names in the dataset

#create a list of all the codes that appear across all offense columns
all_possible_offense_codes=[]
for colname in all_possible_offense_codes.values():
    for value in dataset[colname].values():
        if value not in all_possible_offense_codes:
            all_possible_offense_codes.append(value)


#create a copy subset of the dataframe with just the offense columns
offense_cols_subset=dataset[offense_columns]

#convert to dictionary- quicker to loop through than df
offense_cols_dict=offense_cols_subset.to_dict(orient='index')

#create an empty dictionary to hold the counts and append back onto the main dataframe
all_offense_counts={}

#look at each row in the dataframe (converted into a dict) one by one
for row,variables in offense_cols_dict.items():
    
    #create a dict with all offense code as key and value as 0 (starting count)
    #considered using get(code,0) rather than prepopulating keys and vals...
    #but think different vals across dicts would create alignment issues...
    #when appending back onto dataset df
    this_row_offense_counts={code:0 for code in all_possible_offense_codes}
    
    #then go through each offense column
    for column in offense_columns:
        #find the code stored in this column for this row
        code=offense_cols_dict[row,column]
        #increment count by 1
        this_row_offense_counts[code]=this_row_offense_offense_counts[code]+1
    
    #once all columns have been counted, store counts in dictionary
    all_offense_counts[row]=this_row_offense_counts

#once all rows have been counted, turn into a dataframe 
offense_counts_cols=pf.DataFrame.from_dict(all_offense_counts,orient=index)
#join to the original dataframe
dataset.join(offense_counts)
#drop the sparsely populated offense_columns
dataset.drop(offense_columns,axis=1)

#No criminal justice status-sentenced
#V0114
#V0115
#V0116
#V0117
#V0118
#Parole violators- original offenses?
#V0246
#V0247
#V0248
#V0249
#Parole violators, new offenses
#V0270
#V0270
#V0271
#V0274
#Probation violators original offense
#V0322
#V0323
#V0324
#V0325
#V0326
#Probation violators new offense
#V0346-V0350