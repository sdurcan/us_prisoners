# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 22:12:53 2022

@author: siobh
"""

import pandas as pd
for item in md.config.index:
    print(item)
    if md.config['treatment_violent_lr'][item]=='one_hot':
        vc=subset1[item].value_counts(dropna=False)
        if count==0:
            review=pd.DataFrame([vc])
        #if count==1:
            #review=pd.merge(vc,review,left_index=True, right_index=True)
        else:
            review=review.join(vc,review)