# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 12:59:51 2022

@author: siobh
"""

from src import model
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle

#import survey of prison inmates
#important to keep the empty string set to na, otherwise something goes wrong with data types, which impacts value counts and more
prisoners=pd.read_csv(r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/downloaded_package/DS0001/37692-0001-Data.tsv',sep="\t",keep_default_na=False,na_values=[' '])
#import variables config
#violent_variables=pd.read_csv(r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/violent_variables.csv', index_col=0)
violent_variables=r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/data/processing_config/violent_variables.csv'



#filter to top 3 prisoner types and violent crime
#doing this before normalising data types in order to reduce size of dataframe
#V0401==6;  single crime, single or flat sentence with specified amount of time
#V0412==6; multiple crimes,single or flat sentence with specified amount of time
subset1=prisoners[(prisoners['V0062']==1) & (prisoners['V0063'].isin([3,11,8]))& ((prisoners['V0401']==6) | (prisoners['V0412']==6))]
#print('Index length',len(subset1.index))

md=model.model(subset1,violent_variables,model='mlp',pred_classes=2,stand=1,norm=0,ytype="binary", target="sentence_length",train_size=0.2)

#serialise the data for use in colab
pickle.dump(md.dataset_out,open('C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/output/processed_data.csv','wb'))

x_train,y_train,x_test,y_test=md.x_train,md.y_train,md.x_test,md.y_test

save_loc='C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/output/mlp_peram_sweep_results/results3.pkl'


#mlp params
'''
activations=['relu','tanh','identity', 'logistic']
solvers=['adam','sgd','lbfgs']
learning_rates=['constant','invscaling','adaptive']
alphas = np.logspace(-1, 1, 5)
'''

activations=['identity', 'logistic']
solvers=['lbfgs']
learning_rates=['constant','invscaling','adaptive']
alphas = np.logspace(-1, 1, 5)

count=0
mlp_sweep_results={}
for solver in solvers:
    for activation in activations:
        for learning_rate in learning_rates:
            for alpha in alphas:

                mlp = MLPClassifier(max_iter=100000,solver=solver,activation=activation,learning_rate=learning_rate, alpha=alpha).fit(x_train, y_train)
                y_pred=mlp.predict(x_test)
                train_score=mlp.score(x_train,y_train)
                test_score=mlp.score(x_test,y_test)
                print(activation, solver, learning_rate, alpha)
                print(train_score)
                print(test_score)
                #self.params=self.mlp.get_params()
                #self.train_proba=self.mlp.predict_proba(self.x_train)
                #self.test_proba=self.mlp.predict_proba(self.x_test)
                mlp_sweep_results[count]={}
                
                mlp_sweep_results[count]['solver']=solver
                mlp_sweep_results[count]['activation']=activation
                mlp_sweep_results[count]['learning rate']=learning_rate
                mlp_sweep_results[count]['alpha']=alpha
                count=+1
                #pickle results
                pickle.dump( mlp_sweep_results, open( save_loc, "wb" ) )


#relu adam invscaling 3.1622776601683795
'''
tanh adam adaptive 3.1622776601683795
0.7360905044510386
0.706973293768546
identity adam constant 0.1
0.7640949554896143
0.7188427299703264
'''