import copy
import pandas as pd
from src import var_processor as vp
from sklearn.model_selection import train_test_split
from src import sentence_creator as sc

class dataset_processor:
    
    def __init__(self,dataset,config,target="sentence_length",bi=True,th=10,multi_cat=False,auto_cats=0,bin_edges=[], continuous=False,subset_num=3):
        
        #dataset will usually be a subset of the prisoner data
        #config will be a dataframe of variables and how to treat them
        #target is the variable to be predicted. This will inform which 'calc' function is called
        #bi is wether the target is to be binary
        #th is the threshold if the target is to be binary
        #multi_cat is if multiple categories will be used
        #auto_cats is populated if multiple categories is true and the categories are to be
        #generated using qcut
        #bin_edges is populated if multi_cat is true and grouping is to be set mnaully
        #continuous is true if the target variable is to be continuous

        self.dataset_in=copy.deepcopy(dataset)
        self.config=config
        print('Config type at initialisation',type(self.config))
        self.target=target
        self.bi=bi
        self.th=th
        self.multi_cat=multi_cat
        self.auto_cats=auto_cats
        self.bin_edges=bin_edges
        self.continuous=continuous
        self.subset_num=subset_num


        '''
        print('dataset processor before encode dataset in type',type(self.dataset_in))
        print(self.dataset_in.isna().sum().sum())
        print(len(self.dataset_in.index))
        print(self.dataset_in.index[0:10])
        '''

        #self.calc_sentence()
        #self.encode_and_scale()

        '''
        print('dataset processor after encode',type(self.dataset_out))
        print(self.dataset_out.isna().sum().sum())
        print(self.dataset_out.index[0:10])
        '''
    
    def calc_target(self):
        if self.target=='sentence_length':
            self.calc_sentence()

    def calc_sentence(self):
        #call sentence creator object
        #update self.dataset_out with a new dataframe that has sentence length calculated and dropped other vars
        #don't need to worry about standardisation and normalisation here
        update=sc.sentence_creator(self.dataset_in,self.subset_num,self.bi, self.th)
        self.dataset_out=update.subset
        
        return #sc.sentence_creator.new_cols#print names of new columns that have been created
    
    def encode_and_scale(self,stand=0,norm=0):
        #print('Config type at encode and scale',type(self.config))
        for colname in self.config.index:
            #print(colname)                   
            #cell var processor
            #initiates object
            processed_var=vp.var_processor(self.dataset_in, colname, self.config)
            
            #see how many nans there are in this column
            if processed_var.output_col.isna().sum().sum()>0:
                if processed_var.treatment!="cont_wnans":
                    print(processed_var.output_col.isna().sum().sum())
                    print(colname)

            #if this is the first variable in the list, initiate encoded dataframe
            if self.config.index[0]==processed_var.colname:
                self.dataset_out=processed_var.output_col
            
            else:
                self.dataset_out=self.dataset_out.join(processed_var.output_col)
        
        self.join_target()
        
    
    def join_target(self):
        
        #then add the target cols that were created
        #these won't have appeared on the config list and so won't have been processed and added
    
        if 'above_thr_inc_life' in list(self.dataset_out.columns):
            print('its in there')
        else:
            print('It not in there')
            
        self.dataset_out=self.dataset_out.join(self.dataset_in['above_thr_inc_life'])

        if 'above_thr_inc_life' in list(self.dataset_out.columns):
            print('its in there')
        else:
            print('It not in there')

                                               
                                               
                                               
#print('Compare index above thr series with dataset out')
#print(list(self.dataset_in['above_thr_inc_life'].index[0:10]))
#print(list(self.dataset_out.index)[0:10])
#self.dataset_out=self.dataset_out.join(self.dataset_in['above_thr_inc_life'], how = 'left', lsuffix='left', rsuffix='right')

