# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 19:59:50 2022

@author: siobh
"""


#plot column distribution over increasing value
def plot_sentence_length_dist(series):
    vc=series.value_counts(dropna=False)
    vc.sort_index(inplace=True)
    vc.plot.bar(figsize=(110,5))
    plt.show()

def corr_cont_vars_w_target(config,output_dataset,treatment_col='treatment_violent_lr',target='combined_offence_length'):
    #variables config is a dataframe
    corr_vars=config[[treatment_col] == 'cont_wnans'].index.to_list()
    corr_vars.append(output_dataset[target])
    new = dataset_out[corr_vars].copy()
    #new.head()
    new.corr()
    plt.matshow(new.corr())

def plot_manual_cut(series,bin_list=""):
    #cut_labels = ['<1', '<5', '<10', '<15','<20','<25','<30','>30']
    #cut_labels = ['<1', '<5', '<10', '<15','<20','<25','30','35','40']
    
    if bin_list=="":
        bin_list=[-0.001, 365,730,1095,1460,1825,2190,2555, 2920,3285,3650,4015,4380,4745,5110, 5475,5840,6205,6570,6935,7300,7665,8030,8395,8760,9125,9490,9855,10220,10585,10950,12775,14600,328500]

    manual_cut= pd.cut(series, bins=cut_bins, labels=False)
    manual_cut.value_counts().sort_index().plot(kind='bar')


def plot_qcut(series,n):
    #qcut will calculate the size of each bin in order to make sure the distribution of data in the bins is equal
    #all bins will have (roughly) the same number of observations but the bin range will vary.
    qcut=pd.qcut(series, q=n)
    qcut.value_counts().sort_index().plot(kind='bar')

def corr(dataframe,filename='corr.png'):
    #produce correlations
    import seaborn as sn
    corr=dataframe.corr()
    corrmap=sn.heatmap(corr, annot=False)
    corrmap.figure.savefig(filename)


def profile_data_to_file(dataframe,filename="report.html"):
    profile=dataframe.profile_report(title=filename)
    #profile = ProfileReport(processed_prisoners, title="prisoner_without_corr_7",correlations=None,interactions=None,html={'style',:'full_width': True}})
    profile.to_file(output_file=filename)
    #return pn.pane.HTML(profile.to_html())

def corr2(dataset):
    df=dataset
    c = df.corr().abs()
    s = c.unstack()
    so = s.sort_values(kind="quicksort",ascending=False)
    return so

#subset3 will contain all: ['V0401','V0402','V0405','V0406','V0413','V0417','V0418']
#subset2 will not contain V0405,V0406,V0417,V0418
#subset1 will not contain V0401, V0402, V0405,V0406,V0417,V0418

#V0402:FLAT SINGLE the number of years for prisoners being sentence for a single crime with a flat sentence an a specific number of years
#V0413:FLAT MULTI the length of the max sentence if it's a flat sentence for prisoners being sentenced for multiple crimes

#V0401: the type of flat sentence for inmates being sentenced for a single crime with a flat sentence- indicates life and death
#V0412: the type of flat sentence for inmates being sentenced for a multiple crimes with a flat sentence-- indicates life and death
#1 (Life), 2 (Life Plus Additional Years), 3 (Life Without Parole), 4 (Death), -9 (Missed),-2 (Refusal),-1 (Don't know)-8 (Skipped)

#V0405 INDETERMINATE SINGLE contains the minimum number of years for a single sentence if it's a range
#V0406 INDETERMINATE SINGLE contains the maximum number of years for a single sentence if it's a range

#V0417 INDETERMINATE MULTI contains minimum number of years for the longest sentence they are being sentenced for
#V0418 INDETERMINATE MULTI contains the max number of years for the longest sentence they are being sentenced for


#code to analyse sentence length threshold distrbutions
#save subset3 somewhere
#do this for each of the 'year' fields- making a copy of each one
#subset 3 is 10248 in length
#add a line to replace nan and dk_refs with 0
import pandas as pd
import numpy as np
subset=subset3
sentence_years=['V0402','V0413','V0405','V0406','V0417','V0418']
for index,var in enumerate(sentence_years):
    if var in subset.columns:
        vc=subset[var].value_counts(bins=[0,10,np.inf])
        if index==0:
            out=pd.DataFrame(vc)
        else:
            out=out.join(vc)
    
    
    out['05-06']=out['V0405']
    out.drop(columns=['V0405','V0406','V0407','V0418'])
    out.sort_index().plot(kind='bar')
            
    
    
#need to create an 'or' columns for 05/06 and 17/18
copy_402.replace(to_replace=[-9,-2,-1,np.nan],value=0)
cut=pd.cut(copy_402,bins=[0,10,np.inf])
cut.value_counts().sort_index().plot(kind='bar')



#get the 'above threshold' counts from the flat categorical fields
copy_401.replace(to_replace=[-9,-2,-1,5,6,np.nan],value=0)
copy_401.value_counts(dropna=False)

def one_if_either_is_x(subset,col1,col2,new_name,test_val=1,new_val=1):
    #creates a new pandas column with a '1' if either column satisfies logical test
    #df.loc[df[‘column’] condition, ‘new column name’] = ‘value if condition is met’
    
    
    #subset[new_name]=np.where()
    cond=((col1|col2)==test_val)
    subset[new_name] = np.where(cond, new_val, 0)

    def get_truth(col, relate, test_val):
        ops = {'>': operator.gt,
               '<': operator.lt,
               '>=': operator.ge,
               '<=': operator.le,
               '==': operator.eq}
        return ops[relate](col, test_val)

    def one_if_either(self,col1,col2,new_name,operator='==',test_val=1,new_val=1):

        cond=self.get_truth(col1,operator,test_val) | self.get_truth(col2,operator,test_val) 

        self.subset[new_name] = np.where(cond, new_val, 0)