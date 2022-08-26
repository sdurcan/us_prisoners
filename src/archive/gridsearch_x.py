# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:41:49 2022

@author: siobh
"""

#logistic regression
#mlp
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_classif,VarianceThreshold, mutual_info_classif, chi2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import warnings
import copy as copy
import pickle
from sklearn.feature_selection import SelectKBest
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plot
warnings.filterwarnings("ignore")

'''
#mlp perams
mlps={'classifier':[MLPClassifier(verbose=True,max_iter=100000,verbose=True, early_stopping=True)],
'classifier__activation':['identity', 'logistic', 'tanh', 'relu'],
'classifier__solver':['lbfgs', 'sgd', 'adam'],
'classidier__alpha':[0.0001,0.001,0.01,0.1],
'classifier__tol':[0.01, 0.001,0.0001,0.00001]}
'''
#load the encoded subset

#r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/
prepped_subset=pd.read_pickle(r'C:/Users/siobh/OneDrive/Masters/Dissertation/us_prisonsers/output/prepped_nopp_offcounts.csv')

#set and drop target column the dataset processor
#column name is dynamic based on the threshold
target='sentence_above_20yrs'

#drop any columns containing information about the target
X=prepped_subset.drop(labels=target,axis=1)
#there is a col with int name 0 messing things up
#X.drop(0,inplace=True,axis=1)
X.columns = X.columns.astype(str)
y=copy.deepcopy(prepped_subset[target])

var_thr = VarianceThreshold(threshold = 0.05) #Removing both constant and quasi-constant
var_thr.fit(X)
var_thr.get_support()
concol = [column for column in X.columns 
          if column not in X.columns[var_thr.get_support()]]
X.drop(concol,axis=1,inplace=True)

#we are doing cv, so keeping a small held-out test set to check final classifier on
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=1)

'''
#initialise pipeline with a selector and a classifier
pipe = Pipeline([('selector', SelectKBest()),
                 ('classifier', LogisticRegression(verbose=True, max_iter=100000,n_jobs=8))])


#logistic regression with Kbest
parameters =  {'classifier__penalty':['l1', 'l2', 'elasticnet'],
                 'classifier__tol':[0.0001,0.00001],
                 'classifier__solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                 'selector__k':[5,55,100,150,200,250,300,350,400],
                 'selector__score_func':[chi2,mutual_info_classif]}
'''
'''
#logistic regression with variance threshold
parameters =  {'classifier__penalty':['l1', 'l2', 'elasticnet'],
                 'classifier__tol':[0.001,0.0001,0.00001],
                 'classifier__solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                 'selector__threshold':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}

#logistic regression with select from decision tree model
pipe = Pipeline([('selector', SelectFromModel(DecisionTreeClassifier())),
                 ('classifier', LogisticRegression(verbose=True, max_iter=100000,n_jobs=8))])
parameters =  {'classifier__penalty':['l1', 'l2', 'elasticnet'],
                 'classifier__tol':[0.001,0.0001,0.00001],
                 'classifier__solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

#mlp with kbest
pipe = Pipeline([('selector', SelectKBest()),
                 ('classifier', MLPClassifier(max_iter=100000,verbose=True, early_stopping=True))])

parameters={

'classifier__activation':['identity', 'logistic', 'tanh', 'relu'],
'classifier__solver':['lbfgs', 'sgd', 'adam'],
'classifier__alpha':[0.0001,0.001,0.01,0.1],
'classifier__tol':[0.001,0.0001,0.00001],
'selector__k':[100,200,300,400,500,600],
'selector__score_func':[chi2,mutual_info_classif]}
'''

#Decision tree with Kbest
pipe=Pipeline([('selector',SelectKBest()),('classifier',DecisionTreeClassifier())])
parameters={'classifier__max_depth':[5,10,20,30],
            'classifier__criterion':['gini','entropy'],
            'classifier__max_leaf_nodes':[10,20,30],
            'classifier__min_samples_split':[10,20,30],
            'classifier__min_samples_leaf':[1,10,20,30],
            'selector__k':[5,10,20,30,40,50,75,100],
            'selector__score_func':[mutual_info_classif, chi2]}

grid = GridSearchCV(pipe, parameters, cv=5,verbose=3,n_jobs=8).fit(X, y)
result_df = pd.DataFrame.from_dict(grid.cv_results_, orient='columns')
print(grid.best_estimator_)
print(grid.best_score_)
result_df.to_csv('results_mlpgridsearch_round4')
#print(grid.best_estimator_.named_steps['classifier'].feature_names_in_)

#see the features that have come out
#cols=grid.best_estimator_.named_steps['selector'].get_support(indices=True)

#features_df_new = X.iloc[:,cols]

result_df = pd.DataFrame.from_dict(grid.cv_results_, orient='columns')
#print(result_df.columns)
 
print('Training set score: ' + str(grid.score(X_train, y_train)))
print('Test set score: ' + str(grid.score(X_test, y_test)))

#This DataFrame is very valuable as it shows us the scores for different parameters. 
#The column with the mean_test_score is the average of the scores on the test set for all the folds during cross-validation. 
#The DataFrame may be too big to visualize manually, hence, it is always a good idea to plot the results. 
#Let’s see how n_neighbors affect the performance for different scalers and for different values of p.

'''
#changing the value in col changes the value of the dataset
plt=sns.relplot(data=result_df,
	kind='line',
	x='param_selector__k',
	y='mean_test_score',
	hue='param_classifier__solver',
	col='param_classifier__penalty')

    
plt.savefig('tpol_conf.png', dpi=400)
'''

#plot feature importance 

#df['Mutual_Info_Rank'] = df['mutual_info_score'].rank(ascending = 1)
xfs=SelectKBest(score_func=mutual_info_classif,k='all')

selector=SelectKBest(k=50,score_func=mutual_info_classif)
fs.fit(X,y)
zipped=zip(X.columns, fs.scores_)

#feature selection
fs=SelectKBest(score_func=chi2,k='all')
fs.fit(X_train,y_train)
feature_names=X.columns
scores=fs.scores_
zipped = zip(feature_names, scores)
chi_df=pd.DataFrame(zipped, columns=["feature", "chi2_score"])
chi_df = chi_df.sort_values("feature", ascending=False)
chi_df['chi2_Rank'] = chi_df['chi2_score'].rank(ascending = 1)



#fsx=fs.transform(X)
#classifier
xclf=DecisionTreeClassifier(max_depth=10, max_leaf_nodes=30,
                        min_samples_split=10)

1clf=DecisionTreeClassifier(criterion='entropy', max_depth=5,max_leaf_nodes=20, min_samples_leaf=10,min_samples_split=30)


    


model=clf.fit(X,y)
coefs = clf.feature_importances_
feature_names = clf.feature_names_in_
zipped = zip(feature_names, coefs)
df = pd.DataFrame(zipped, columns=["feature", "value"])
# Sort the features by the absolute value of their coefficient
df["abs_value"] = df["value"].apply(lambda x: abs(x))
df["colors"] = df["value"].apply(lambda x: "green" if x > 0 else "red")
df = df.sort_values("abs_value", ascending=False)

fig, ax = plt.subplots(1, 1, figsize=(43, 7))
sns.barplot(x="feature",
            y="value",
            data=df.head(150),
           palette=df.head(150)["colors"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=20)
ax.set_title("Top 20 Features", fontsize=25)
ax.set_ylabel("Coef", fontsize=22)
ax.set_xlabel("Feature Name", fontsize=22)


#show relative feature importance
fs=grid.best_estimator_.named_steps['selector']
feature_importance = abs(fs.scores_)
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
featfig = plot.figure(figsize=(50, 50),dpi=500)
featax = featfig.add_subplot(1, 1, 1)
featax.barh(pos, feature_importance[sorted_idx], align='center')
featax.set_yticks(pos)
featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=8)
featax.set_xlabel('Relative Feature Importance')
plot.savefig('feature importance')
#plt.tight_layout()   
plot.show()

#sklearn inprocessing compatible with decision trees and logistic regression
#https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.GerryFairClassifier.html
#maybe this one
#https://aif360.readthedocs.io/en/stable/modules/generated/aif360.algorithms.inprocessing.ExponentiatedGradientReduction.html

#from sklearn import tree
#tree.plot_tree(elf)

#tree.plot_tree(grid.best_estimator_.named_steps['classifier'])

#DecisionTree
#MLP
#Logistic Regression
#K nearest neighbours

#VarianceThreshold
#SelectKBest
#SelectFromModel OR
#RFE (estimators and nfeatures to select)

#Dataset variations
#Try these 48 variations against a decision tree and logistic regression dataset that has performed okay on current version of dataset
#Scaling MinMax, StandardScaler, Robust [0,1,2]
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

# Drop all of the offense codes apart from violent type and count offenses? [1,0]
# This could be implemented by removing all columns containing off code from prepped subset
# Drop all data about year of arrest and age [1,0]
#RV002 is age bracket - ordinal, V0917 is date first admitted to prison (year), V0055Y is arrest year
## Ordinal encoding applied vs not applied to year ranges/large variation continuous vars [0,1]
# Ordinal encoding applied to offenses [1,0]


#Imputing- show that there aren't many missing values and so this isn't being explored

#some column s have a constant value after encoding
#for code columns, this will be where offenders do not have a 4th or 5th offence listed
constant_cols=['V0572-x0_8',
 'V0902',
 'off_1_code-x0_360.0',
 'off_1_code-x0_482.0',
 'off_2_code-x0_42.0',
 'off_2_code-x0_192.0',
 'off_2_code-x0_260.0',
 'off_2_code-x0_271.0',
 'off_2_code-x0_481.0',
 'off_2_code-x0_570.0',
 'off_2_code-x0_810.0',
 'off_3_code-x0_20.0',
 'off_3_code-x0_380.0',
 'off_4_code-x0_42.0',
 'off_4_code-x0_340.0',
 'off_4_code-x0_342.0',
 'off_4_code-x0_347.0',
 'off_4_code-x0_400.0',
 'off_4_code-x0_560.0',
 'off_4_code-x0_580.0',
 'off_4_code-x0_601.0',
 'off_4_code-x0_630.0',
 'off_5_code-x0_91.0',
 'off_5_code-x0_130.0',
 'off_5_code-x0_290.0',
 'off_5_code-x0_320.0',
 'off_5_code-x0_410.0',
 'off_5_code-x0_550.0',
 'off_5_code-x0_565.0',
 'off_5_code-x0_601.0',
 'ctrl_off_code-x0_42.0',
 'ctrl_off_code-x0_190.0',
 'ctrl_off_code-x0_565.0']


'''
#align the scores to the feature names
zipped=zip(fs.feature_names_in_,fs_scores)

#get the top k features
df = pd.DataFrame(zipped, columns=["feature", "value"])
df = df.sort_values("value", ascending=False)

#plot the decision tree
plot.figure(figsize=(30, 30),dpi=400) # Resize figure
tree.plot_tree(clf, filled=True)
plot.savefig('test')
plot.show()
'''


selector=SelectKBest(k=5,score_func=mutual_info_classif)
clf=DecisionTreeClassifier(criterion='gini', max_depth=10,
                        max_leaf_nodes=30, min_samples_leaf=30,
                        min_samples_split=20)
descriptive_dtree_plot(selector,clf,X_train,y_train,X_test,y_test,'best_k5_dtree_plot_after_round4.png')

def descriptive_dtree_plot(selector,clf,X_train,y_train,X_test,y_test,title):
    #selector is a feature selector object
    #clf is a classifier
    selector.fit(X_train,y_train)
    cols = selector.get_support(indices=True)
    X_train_new = X_train.iloc[:,cols]
    X_test_new = X_test.iloc[:,cols]

    
    #then we can give tthese to the classifier
    clf.fit(X_train_new,y_train)
    print('Score on training set',clf.score(X_train_new,y_train))
    print('Score on test set',clf.score(X_test_new,y_test))

    from sklearn import tree
    plot.figure(dpi=200, figsize=(40,40))
    tree.plot_tree(clf, filled=True, feature_names=X_train_new.columns,class_names=['below','above'])
    plot.savefig(title)
    plot.show()
    #once classifier is trained, can get a table
    dtree_scores=clf.feature_importances_
    feature_names=X_train_new.columns
    zipped=zip(feature_names,dtree_scores)
    dtree_df=pd.DataFrame(zipped,columns=['feature','dtree_score'])
    dtree_df.sort_values('dtree_score',inplace=True,ascending=False)
    
    return dtree_df



selector=SelectKBest(k=50,score_func=mutual_info_classif)
selector.fit(X_train,y_train)
cols = selector.get_support(indices=True)
X_train_new = X_train.iloc[:,cols]
X_test_new = X_test.iloc[:,cols]
#clf is a trained logistic regression classifier
clf.fit(X_train_new,y_train)
feature_names=clf.feature_names_in_
lgclf_feature_scores=clf.coef_[0]
zipped=zip(feature_names,lgclf_feature_scores)
lgclf_df=pd.DataFrame(zipped,columns=['feature','lgclf_coef'])



clf=DecisionTreeClassifier(criterion='entropy', max_depth=50,max_leaf_nodes=20, min_samples_leaf=10,min_samples_split=30)
selector.fit(X_train,y_train)

cols = selector.get_support(indices=True)
xclf=DecisionTreeClassifier(max_depth=10, max_leaf_nodes=40,
                       min_samples_leaf=10,
                       min_samples_split=10)           


X_train_new = X_train.iloc[:,cols]
X_test_new = X_test.iloc[:,cols]

clf.fit(X_train_new,y_train)

from sklearn import tree
plot.figure(dpi=200, figsize=(40,40))
tree.plot_tree(clf, filled=True, feature_names=X_train_new.columns,class_names=['below','above'])
plot.savefig('best_dtree_plot_after_off_count')
plot.show()

dtree_scores=clf.feature_importances_
feature_names=X_train_new.columns
zipped=zip(feature_names,dtree_scores)
dtree_df=pd.DataFrame(zipped,columns=['feature','dtree_score'])