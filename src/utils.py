# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:31:58 2022

@author: siobh
"""

import os, re
import pickle
import statsmodels.api as sm
import pandas as pd

def increment_savename(fdir,prefix="test_name",ext='txt'):

    #increment the savename by 1 based on what is already in the folder
    #input path must be folder with files following prefix+num.extension pattern
    
    fdir=fdir
    ext=ext
    
    current_files=[os.path.join(fdir, _) for _ in os.listdir(fdir) if _.endswith(ext)]
    
    num_list = [0]
    
    for f in current_files:
    
        i = os.path.splitext(f)[0]
    
        try:
    
            num = re.findall('[0-9]+$', i)[0]
    
            num_list.append(int(num))
    
        except IndexError:
    
            pass
    
    num_list = sorted(num_list)
    
    new_num = num_list[-1]+1
    
    save_name = f"{prefix}{new_num}.{ext}"
    
    print(save_name)
    return save_name

def name_and_pickle(item,fdir,prefix,ext='pkl'):
    
    save_name=increment_savename(fdir,prefix,ext)
    
    save_path=fdir+save_name
    print(save_path)
    
    with open(save_path,"wb") as f:
        pickle.dump(item,f )
    
    return save_path


def calculate_vif(data):
    vif_df = pd.DataFrame(columns = ['Var', 'Vif'])
    x_var_names = data.columns
    for i in range(0, x_var_names.shape[0]):
        y = data[x_var_names[i]]
        x = data[x_var_names.drop([x_var_names[i]])]
        r_squared = sm.OLS(y,x).fit().rsquared
        vif = round(1/(1-r_squared),2)
        vif_df.loc[i] = [x_var_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis = 0, ascending=False, inplace=False)