# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:31:58 2022

@author: siobh
"""

import os, re
import pickle


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

'''
alist=[0,1,2,3,4,5]
fdir='C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/output/test/'
prefix="a_test"
name_and_pickle(alist,fdir,prefix)

#to_open=open('C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/output/test/a_test3.pkl','rb')
#opening=pickle.load(to_open)
#print(opening)
'''