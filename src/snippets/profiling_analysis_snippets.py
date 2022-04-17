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