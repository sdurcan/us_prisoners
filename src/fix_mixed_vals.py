# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:35:51 2022

@author: siobh
"""

from pandas.api.types import infer_dtype

#UTILS FOR TURNING MIXED TYPE COLUMNS INTO INTEGERS AND FLOATS

def check_float(potential_float):
    try:
        float(potential_float)
        #Try to convert argument into a float
        return True
    except ValueError:
        return False

def make_num(val):
    #if potential float, convert to float
    if check_float(val)==True:
        return float(val)
    #if potential integer, convert to float (avoiding mixed types)
    elif val.isdigit()==True:
        return float(val)
    else:
        #return column, dtype and val
        
        #print('{},{}'.format(val,type(val)))
        return(val)

def fix_mixed_vals(df_series):
    """takes a dataframe as input and standardises columns with mixed values"""
    #columns_fixed=0
    #test to see if mixed types
    if infer_dtype(df_series) in ["mixed-integer-float","mixed-integer","mixed"]:
        #turn these values into ints or floats where possible
        df_series=df_series.apply(make_num)
        #columns_fixed=columns_fixed+1
    #print("Columns with fixe mixed vals applied {}".format(columns_fixed))
    return df_series

def apply_fix_mixed_vals(dataframe):
    'applies standardise_mixed vals to all columns in a dataframe'
    out=dataframe.apply(fix_mixed_vals, axis='columns')
    return out

#STRING TOGETHER PROCESSING UTILS IN PIPE
def process (dataframe,filename=""):
    'use df pipe to chain together processing'
    processed_df=dataframe.pipe(apply_fix_mixed_vals)
    return processed_df