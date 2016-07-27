# -*- coding: utf-8 -*-
"""
Created on Tue Jun 07 18:46:38 2016

@author: kush
"""

ipython notebook --pylab=inline

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

#Import the dataset using PANDAS
df= pd.read_excel("C:\\Users\\kush\\Desktop\\sales.xlsx")

test_df= pd.read_excel("C:\\Users\\kush\\Desktop\\Test_sales.xlsx")

# Check Data types of the variables
df.dtypes

#Check the brief about data i.e. it gives you CENTRAL TENDENCY stats about the data. mean,median,std,quartile
df.describe()

# For this data there are no Missing values



#To check the outliers the best way is to visualize it. Following are some ways to show the data distributions
new_df= df[['vegetable Name','last day sales','last to last day sales','last week sales','last to last week sales','returns','sales']]

#AS the dataset is small Seperating the data for each vegetable to get a better view
coriander_data= new_df.loc[df['vegetable Name']== 'Coriander']  
veg_cor=sns.swarmplot(data=coriander_data)

cucumber_data= new_df.loc[df['vegetable Name']== 'Cucumber (Indian)']  
veg_cucu=sns.swarmplot(data=cucumber_data)

peas_data=new_df.loc[df['vegetable Name']== 'Green peas'] 
veg_peas=sns.swarmplot(data=peas_data)
#You can observe that green_peas has a outlier values. better to remove them. 

potato_data= new_df.loc[df['vegetable Name']== 'Potato']
veg_pota=sns.swarmplot(data=potato_data)

tomato_data= new_df.loc[df['vegetable Name']== 'Tomato (hybrid)']  
veg_tomato=sns.swarmplot(data=tomato_data)


#Now to get the Idea about variables, we need to see how much they are collineared amongst them
df.corr()
new_df.corr()


#goupby can give you the idea what was the sales in last week and last days. You can calculate the difference and tell weather sales are doing good or not from the last week.
new_df.groupby('vegetable Name').sum()

# Data Exploration. 
#For the given dataset its better to add variables to make our model as strong as possible
new_df['Total Previous Sales']= df['last week sales'] + df['last to last week sales']
new_df['last two days sales']= df['last day sales'] + df['last to last day sales']
new_df['New Sales']= df['sales']-df['returns']



predictors=['Total Previous Sales','last two days sales','last day sales','last to last day sales','last to last week sales','last week sales']
                                                            	

#feature selection
from sklearn.feature_selection import  SelectKBest,f_classif
selecter= SelectKBest(f_classif,k='all')
selecter.fit(new_df[predictors],new_df['New Sales'])
pkivalue= -np.log10(selecter.pvalues_)
#from pkivalue you can see that almost all the feature have equal importance except "last to last week sales'




new_predictors=['Total Previous Sales','last two days sales']

from sklearn.linear_model import LinearRegression
algo= LinearRegression()

from sklearn.cross_validation import KFold

kf= KFold(df.shape[0],n_folds=3,random_state=1)

predictions=[]

for train,test in kf:
    training_predictors= df[predictors].iloc[train,:]
    training_target= df['New Sales'].iloc[train]
    algo.fit(training_predictors,training_target)
    test_predictions= algo.predict(df[predictors].iloc[test,:])
    predictions.append(test_predictions)
    
predictions= np.concatenate(predictions)

kf1= KFold(new_df.shape[0],n_folds=5,random_state=1)

predictions1=[]
for train1,test1 in kf1:
    training_predictors1= new_df[new_predictors].iloc[train,:]
    training_target1= new_df['New Sales'].iloc[train]
    algo.fit(training_predictors1,training_target1)
    test_predictions1= algo.predict(new_df[new_predictors].iloc[test,:])
    predictions1.append(test_predictions1)
    
predictions1= np.concatenate(predictions1)


kf2= KFold(new_df.shape[0],n_folds=10,random_state=1)

predictions2=[]
for train2,test2 in kf2:
    training_predictors2= new_df[new_predictors].iloc[train,:]
    training_target2= new_df['New Sales'].iloc[train]
    algo.fit(training_predictors2,training_target2)
    test_predictions2= algo.predict(new_df[new_predictors].iloc[test,:])
    predictions2.append(test_predictions2)
    
predictions2= np.concatenate(predictions2)



# AS you can observe from the values of prediction1 and predictions2, they both are same. Hence n_folds=5 is enough for our validation testing.

#Case1
# ... When "last two days sales" is independent variable
import statsmodels.api as sm
import statsmodels.formula.api as smf
y=new_df['New Sales']
X= new_df['last two days sales']
X= sm.add_constant(X)
est= sm.OLS(y,X).fit()
est.summary()
est.params


#CASE2
#... when "Total Previous Sales" is independent variable
Z=new_df['Total Previous Sales']
Z= sm.add_constant(Z)
est2= sm.OLS(y,Z).fit()
est2.summary()
est2.params

#CASE3
#... when we have more than one Independent variable
W=new_df[['last two days sales','Total Previous Sales']]
W=sm.add_constant(W)
est3=sm.OLS(y,W).fit()
est3.summary()

#Case4:
Q= new_df['last day sales']
Q=sm.add_constant(Q)
est4= sm.OLS(y,Q).fit()
est4.summary()

#Case5:
O= new_df['last to last day sales']
O=sm.add_constant(O)
est5= sm.OLS(y,O).fit()
est5.summary()


#Predict the values for Case1
X_test=test_df['last two days sales']
X_test= sm.add_constant(X_test)
test_df['New sales']=est.predict(X_test)


#Predict the values for Case2
X_test=test_df['last two days sales']
X_test= sm.add_constant(X_test)
test_df['New sales']=est2.predict(X_test)

#Predict the values for Case3
X_test=test_df[['last two days sales','Total Previous Sales']]
X_test= sm.add_constant(X_test)
test_df['New sales']=est3.predict(X_test)



