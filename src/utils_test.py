# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 13:53:56 2022

@author: siobh
"""
from src import utils as u
import pickle

alist=[0,1,2,3,4,5]
fdir='C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/output/test/'
prefix="a_test"
save_loc=u.name_and_pickle(alist,fdir,prefix)

to_open=open(save_loc,'rb')
opening=pickle.load(to_open)
print(opening)