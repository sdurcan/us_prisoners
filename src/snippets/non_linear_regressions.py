# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 17:46:53 2022

@author: siobh
"""

#RF Regressor model
model_rf = RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=100)
model_rf.fit(x_train, y_train) 
pred_train_rf= model_rf.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_rf)))
print(r2_score(y_train, pred_train_rf))

pred_test_rf = model_rf.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_rf)))
print(r2_score(y_test, pred_test_rf))


#####

#Decision tree

dtree = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=3)

dtree.fit(x_train, y_train)
'''
#classification
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

y_train = np.asarray(y_train)
clf.fit(x_train, y_train)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
              solver='lbfgs')
'''

# Code Lines 1 to 4: Fit the regression tree 'dtree1' and 'dtree2' 
dtree1 = DecisionTreeRegressor(max_depth=20)
dtree2 = DecisionTreeRegressor(max_depth=15)
dtree1.fit(x_train, y_train)
dtree2.fit(x_train, y_train)

# Code Lines 5 to 6: Predict on training data
tr1 = dtree1.predict(x_train)
tr2 = dtree2.predict(x_train) 

#Code Lines 7 to 8: Predict on testing data
y1 = dtree1.predict(x_test)
y2 = dtree2.predict(x_test) 

###

# Print RMSE and R-squared value for regression tree 'dtree1' on training data
print('RMSE training data',np.sqrt(mean_squared_error(y_train,tr1))) 
print('R-squared traininig data',r2_score(y_train, tr1))

# Print RMSE and R-squared value for regression tree 'dtree1' on testing data
print('RSME test data',np.sqrt(mean_squared_error(y_test,y1))) 
print('R squared test data',r2_score(y_test, y1)) 

#https://www.pluralsight.com/guides/non-linear-regression-trees-scikit-learn
# Print RMSE and R-squared value for regression tree 'dtree2' on training data
print(np.sqrt(mean_squared_error(y_train,tr2))) 
print(r2_score(y_train, tr2))

# Print RMSE and R-squared value for regression tree 'dtree2' on testing data
print(np.sqrt(mean_squared_error(y_test,y2))) 
print(r2_score(y_test, y2)) 

###

#RF Regressor model
model_rf = RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=100)
model_rf.fit(x_train, y_train) 
pred_train_rf= model_rf.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_rf)))
print(r2_score(y_train, pred_train_rf))

pred_test_rf = model_rf.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_rf)))
print(r2_score(y_test, pred_test_rf))

###
Support vector regression

X=x_train
y=y_train

# Fit regression model
svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_lin = SVR(kernel="linear", C=100, gamma="auto")
svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)

# Look at the results
lw = 2

svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ["RBF", "Linear", "Polynomial"]
model_color = ["m", "c", "g"]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(
        X,
        svr.fit(X, y).predict(X),
        color=model_color[ix],
        lw=lw,
        label="{} model".format(kernel_label[ix]),
    )
    axes[ix].scatter(
        X[svr.support_],
        y[svr.support_],
        facecolor="none",
        edgecolor=model_color[ix],
        s=50,
        label="{} support vectors".format(kernel_label[ix]),
    )
    axes[ix].scatter(
        X[np.setdiff1d(np.arange(len(X)), svr.support_)],
        y[np.setdiff1d(np.arange(len(X)), svr.support_)],
        facecolor="none",
        edgecolor="k",
        s=50,
        label="other training data",
    )
    axes[ix].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=1,
        fancybox=True,
        shadow=True,
    )

fig.text(0.5, 0.04, "data", ha="center", va="center")
fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()