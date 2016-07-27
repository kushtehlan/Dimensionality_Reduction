# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 02:33:52 2016

@author: kush
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode
df= pd.read_csv("G:\\Datasets\\Analytics Vidya(FINTRO)\\Train_pjb2QcD.csv")

df.describe()

   
#Filling na vallues
df['Applicant_City_PIN'].mode()
df['Applicant_City_PIN'].fillna(mode(df['Applicant_City_PIN']).mode[0],inplace= True)
df['Applicant_City_PIN']= df['Applicant_City_PIN'].astype(int)

df['Manager_Grade'].unique()
df['Manager_Grade'].mode()
df['Manager_Grade'].fillna(mode(df['Manager_Grade']).mode[0],inplace= True)
df['Manager_Grade']= df['Manager_Grade'].astype(int)

df['Manager_Num_Application'].describe()
df['Manager_Num_Application'].unique()
df['Manager_Num_Application']=df['Manager_Num_Application'].fillna(df['Manager_Num_Application'].mean())
df['Manager_Num_Application']=df['Manager_Num_Application'].astype(int)

df['Manager_Num_Coded'].plot()
df['Manager_Num_Coded'].mode()
df['Manager_Num_Coded'].unique()
df['Manager_Num_Coded'].median()
df['Manager_Num_Coded'].fillna(mode(df['Manager_Num_Coded']).mode[0],inplace= True)
df['Manager_Num_Coded']=df['Manager_Num_Coded'].astype(int)
sns.distplot(df['Manager_Num_Coded'])

df['Manager_Business'].plot()
df['Manager_Business'].median()
df['Manager_Business'].mean()
df['Manager_Business']=df['Manager_Business'].fillna(df['Manager_Business'].median())
sns.distplot(df['Manager_Business'])
df['Manager_Business']=df['Manager_Business'].astype(int)

df['Manager_Business2'].plot()
df['Manager_Business2'].median()
df['Manager_Business'].mean()
df['Manager_Business2']=df['Manager_Business2'].fillna(df['Manager_Business2'].median())
df['Manager_Business2']=df['Manager_Business2'].astype(int)

df['Manager_Num_Products2'].mean()
df['Manager_Num_Products2'].median()
df['Manager_Num_Products2']=df['Manager_Num_Products2'].fillna(df['Manager_Num_Products2'].mean())
df['Manager_Num_Products2']= df['Manager_Num_Products2'].astype(int)

df['Manager_Num_Products'].unique()
df['Manager_Num_Products'].mean()
df['Manager_Num_Products'].median()
df['Manager_Num_Products']=df['Manager_Num_Products2'].fillna(df['Manager_Num_Products'].mean())
df['Manager_Num_Products']= df['Manager_Num_Products'].astype(int)

df['Application_Receipt_Date'].value_counts()
df['Application_Receipt_Date'].isnull().sum()

df['Applicant_BirthDate'].isnull().sum()
df['Applicant_BirthDate'].mode()    
df['Applicant_BirthDate'].fillna(mode(df['Applicant_BirthDate']).mode[0],inplace= True)

df['Manager_DOJ'].isnull().sum()
df['Manager_DOJ'].unique()
df['Manager_DOJ'].mode()
df['Manager_DOJ'].fillna(mode(df['Manager_DOJ']).mode[0],inplace= True)

df['Manager_Grade'].unique()

df['Application_Receipt_Date']= pd.to_datetime(df['Application_Receipt_Date'])
df['Applicant_BirthDate']= pd.to_datetime(df['Applicant_BirthDate'])

applicant_date= df['Application_Receipt_Date'].dt.year.astype(int)
applicant_dob= df['Applicant_BirthDate'].dt.year.astype(int)

df['Age']= applicant_date - applicant_dob

applicant_predicors=['Age','Applicant_Gender','Applicant_Marital_Status','Applicant_Occupation','Applicant_Qualification']


var_mod = ['Applicant_Qualification','Applicant_Occupation','Applicant_Marital_Status','Applicant_Gender','Age']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in var_mod:
    df[i]=le.fit_transform(df[i])
    
    
from sklearn.feature_selection import  SelectKBest,f_classif
selecter= SelectKBest(f_classif,k='all')
selecter.fit(df[applicant_predicors],df['Business_Sourced'])
pkivalue= -np.log10(selecter.pvalues_)
print (pkivalue)
#from here we can conclude that  Age,Applicant_Qualification are most important features

sns.distplot(pkivalue)
 
var_mod2= ['Manager_Joining_Designation','Manager_Current_Designation','Manager_Grade','Manager_Status','Manager_Gender','Manager_Num_Application','Manager_Num_Coded','Manager_Business','Manager_Business2','Manager_Num_Products','Manager_Num_Products2']

for i in var_mod2:
    df[i]=le.fit_transform(df[i])
    selecter2= SelectKBest(f_classif,k='all')
selecter2.fit(df[var_mod2],df['Business_Sourced'])
pkivalue2= -np.log10(selecter2.pvalues_)
print (pkivalue2)

df[['Applicant_Occupation','Applicant_Qualification','Age']].corr()
df[['Applicant_Occupation','Applicant_Qualification']].cov()

sns.distplot(df['Applicant_Qualification'])
sns.boxplot(df['Applicant_Qualification'])

plt.scatter(df['Applicant_Qualification'],df['Business_Sourced'])
plt.scatter(df['Applicant_Occupation'],df['Business_Sourced'])
plt.scatter(df['Manager_Num_Products2'],df['Business_Sourced'])
plt.scatter(df['Age'],df['Business_Sourced'])
tt=[[df['Manager_Business2'],df['Business_Sourced']]]
sns.stripplot(x=df['Manager_Business2'], y=df['Business_Sourced'], data=df)
plt.boxplot(df['Applicant_Occupation'],df['Applicant_Qualification'])




df.hist(column="Applicant_Qualification",figsize=(8,8),  color="blue")

df.boxplot(column='Manager_Business',figsize=(8,8))

sns.boxplot(x=["Manager_Business2"], y=["Manager_Business"],  data=df);


kush= ['Age','Applicant_Qualification']


from sklearn import cross_validation
from sklearn.cross_validation import KFold
#alg = LogisticRegression()
kf = KFold(df.shape[0], n_folds=3)

from sklearn.ensemble import RandomForestClassifier
alg_random = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)

predictions = []
for train, test in kf:
    train_predictors = (df[kush].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = df["Business_Sourced"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg_random.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg_random.predict(df[kush].iloc[test,:])
    predictions.append(test_predictions)    

predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0


from sklearn.metrics import roc_curve,auc,roc_auc_score
false_positive_rate, true_positive_rate, thresholds = roc_curve(df['Business_Sourced'], predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

