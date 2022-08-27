# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 17:20:19 2022

@author: siobh
"""


to_plot=encoded_violent_subset[to_plot]
for column_name in to_plot:
    to_plot[column_name].value_counts().plot(kind='bar')
    plt.show()
    
to_plot=['sentence_above_25yrs','offender_white-x0_1','offender_male-x0_1.0','victim_white-x0_1','victim_male-x0_1.0']
df=encoded_violent_subset[to_plot]
size = math.ceil(df.shape[1]** (1/2))
fig = plt.figure()

for i, col in enumerate(df.columns):
    fig.add_subplot(size, size, i + 1)
    df[col].value_counts().sort_index().plot(kind="bar", ax=plt.gca(), title=col, rot=0)

fig.tight_layout()

violent[['V0482','V0552','V0545']].head(10)
encoded_violent_subset['victim_white-x0_1'].head(10)