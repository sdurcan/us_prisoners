import copy
import pandas as pd
from src import var_processor as vp
from sklearn.model_selection import train_test_split

class dataset_processor:
    
    def __init__(self,dataset,config,pred_classes=2,stand=0,norm=0,model="lr",ytype="cont", target="sentence_length",train_size=0.8):


        
        self.train_size=train_size
        self.model=model
        self.ytype=ytype
        self.target=target
        self.norm=norm
        self.dataset_in=copy.deepcopy(dataset)
        self.pred_classes=pred_classes

        '''
        print('dataset processor before encode dataset in type',type(self.dataset_in))
        print(self.dataset_in.isna().sum().sum())
        print(len(self.dataset_in.index))
        print(self.dataset_in.index[0:10])
        '''

        #this needs to be a dframe
        self.config=config
        self.stand=stand
        #call encode method upon initialising
        self.encode()

        '''
        print('dataset processor after encode',type(self.dataset_out))
        print(self.dataset_out.isna().sum().sum())
        print(self.dataset_out.index[0:10])
        '''
    
    def encode(self):
        for colname in self.config.index:
            #print(colname)                   
            #cell var processor
            #initiates object
            processed_var=vp.var_processor(self.dataset_in, colname, self.config,stand=self.stand,norm=self.norm)
            
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

    def calc_sentence_length(self):
        #calculate sentence lengths for the prisoner dataset by multiplying days, years, months
        #because the sentence length columns are continuous, they won't have new colnames like ohe
        
        #copy frame to see if this reduces errors
        self.dataset_out=self.dataset_out.copy()

        #single offence single/flat sentence (years: V0402, months: V0403,days: V0404)
        self.dataset_out['single_offence_length']=(self.dataset_out['V0402']*365)+(self.dataset_out['V0403']*30.5)+(self.dataset_out['V0404'])

        #multiple offences (years: V0413, months: V0414, days: V0415)
        self.dataset_out['multiple_offence_length']=(self.dataset_out['V0413']*365)+(self.dataset_out['V0414']*30.5)+(self.dataset_out['V0415'])
        #test['multiple_offence_length']=(test['V0413']*365)+(test['V0414']*30.5)+(test['V0415'])
        
        #combine both of the above into a continuous value
        self.dataset_out['combined_offence_length']=(self.dataset_out['single_offence_length']+self.dataset_out['multiple_offence_length'])

        #binary or multiclass sentence length
        self.dataset_out['binary_sentence_length']=pd.qcut(self.dataset_out['combined_offence_length'], q=self.pred_classes, labels=[i for i in range(self.pred_classes)])

        
    def get_xy_binary(self):
        self.y_train=self.train['binary_sentence_length']
        #drop the target variable and the variables its derived from
        self.x_train=self.train.drop(columns=["binary_sentence_length","combined_offence_length","multiple_offence_length","single_offence_length","V0402","V0403","V0404","V0413","V0414","V0415"],axis=1)
        
        self.y_test=self.test['binary_sentence_length']
        #drop the target variable and the variables its derived from
        self.x_test=self.test.drop(columns=["binary_sentence_length","combined_offence_length","multiple_offence_length","single_offence_length","V0402","V0403","V0404","V0413","V0414","V0415"],axis=1)

    def get_xy_cont(self):

        self.y_train=self.train['combined_offence_length']
        #drop the target variable and the variables its derived from
        self.x_train=self.train.drop(columns=["binary_sentence_length","combined_offence_length","multiple_offence_length","single_offence_length","V0402","V0403","V0404","V0413","V0414","V0415"],axis=1)
        
        self.y_test=self.test['combined_offence_length']
        #drop the target variable and the variables its derived from
        self.x_test=self.test.drop(columns=["binary_sentence_length","combined_offence_length","multiple_offence_length","single_offence_length","V0402","V0403","V0404","V0413","V0414","V0415"],axis=1)


    def split(self):
        #split into training and test data before dropping predicted variables
        self.test, self.train = train_test_split(self.dataset_out, train_size=self.train_size)        
        #call get_xy to split out data
        if self.ytype=="binary":
            self.get_xy_binary()
            
        elif self.ytype=="cont":
            self.get_xy_cont()
        
        else:
            raise ValueError('Unknown ytype')