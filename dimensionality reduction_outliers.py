# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:09:21 2016

@author: kush
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import seaborn as sn

sn.set(color_codes=True)
machine_1= pd.read_csv("G:\\Datasets\\7z assignment\\Train\\machine1.csv")
machine_1['X'].unique()

machine_1.describe()

def nullvalues(x):
    return sum(x.isnull())
    
machine_1.apply(nullvalues)



machine_1.mean()
machine_1.corr()
machine_1.cov()
machine_1.var()
machine_1.std()

#variance-covariance matrix

#distributions
sn.pairplot(machine_1)
sn.distplot(machine_1['A'])
sn.distplot(machine_1['B'])
sn.distplot(machine_1['D'])
sn.distplot(machine_1['E'])
sn.distplot(machine_1['J'])
sn.distplot(machine_1['L'])
sn.distplot(machine_1['K'])
sn.distplot(machine_1['R'])
sn.distplot(machine_1['X'])

plt.scatter(machine_1['A'],machine_1['X'])
plt.scatter(machine_1['B'],machine_1['X'])
plt.scatter(machine_1['C'],machine_1['X'])


#this answer states that there is multicollinearity  in the problem
import statsmodels.api as sm
import statsmodels.formula.api as smf
A= machine_1['A']
X= machine_1['X']
B= machine_1['B']
C=machine_1['C']
D=machine_1['D']
E=machine_1['E']
F=machine_1['F']
formula= 'A ~ B * C * D'
lm= smf.ols(formula,data=machine_1['X']).fit()
lm.summary()


#feature selection on the basis on variance_
from sklearn import preprocessing
x = machine_1.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df.var()


#feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
pre= machine_1[['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W',]]
clf= ExtraTreesClassifier().fit(pre,machine_1['X'])
clf.feature_importances_
model= SelectFromModel(clf, prefit=True)
new=model.transform(pre)
new.shape()



# Recursive Feature Elimination
from sklearn.feature_selection import RFE
model = ExtraTreesClassifier()
rfe = RFE(model)
rfe = rfe.fit(pre,machine_1['X'])
# summarize the selection of the attributes
machine_model= rfe.transform(pre)
print(rfe.support_)
print(rfe.ranking_)
#after comparing output of two models, its been concluded that Recursive Feature Elimination gives better results

machine_2= pd.read_csv("G:\\Datasets\\7z assignment\\Train\\machine2.csv")
machine_2.isnull().sum()
predictors_2= machine_2[['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W',]]
clf2= ExtraTreesClassifier().fit(predictors_2,machine_2['X'])
clf2.feature_importances_
model_2= SelectFromModel(clf2, prefit=True)
new_model2=model_2.transform(predictors_2)


df_machine1= pd.DataFrame(new)
df_machine1.loc[:,'X']= pd.Series(machine_1['X'],index=df_machine1.index)

df_machine2= pd.DataFrame(new_model2)
df_machine2.loc[:,'X']= pd.Series(machine_2['X'],index= df_machine2.index)

machine_3= pd.read_csv("G:\\Datasets\\7z assignment\\Test\\machine3.csv")

predictors_3= machine_3[['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W',]]

model_3 = ExtraTreesClassifier()
rfe = RFE(model_3,8)
rfe = rfe.fit(predictors_3,machine_3['X'])
# summarize the selection of the attributes
model_3= rfe.transform(predictors_3)

df_machine3= pd.DataFrame(model_3)
df_machine3.loc[:,'X']= pd.Series(machine_3['X'],index= df_machine3.index)
df_machine3.isnull().sum()

df_machine1.to_csv('G:\\Datasets\\7z assignment\\Test\\machine1_dataset.csv')
df_machine2.to_csv('G:\\Datasets\\7z assignment\\Test\\machine2_dataset.csv')

new_df_machine_1= pd.read_csv('G:\\Datasets\\7z assignment\\Test\\machine1_dataset.csv')

new_df_machine_2= pd.read_csv('G:\\Datasets\\7z assignment\\Test\\machine2_dataset.csv')

prediction=[]
from sklearn.ensemble import RandomForestClassifier
forest= RandomForestClassifier(n_estimators=100)
forest= forest.fit(new_df_machine_1,new_df_machine_1['X'])

forest_2= RandomForestClassifier(n_estimators=100)
forest_2= forest_2.fit(new_df_machine_2,new_df_machine_2['X']) 

prediction=forest.predict(df_machine3)


machine_4= pd.read_csv("G:\\Datasets\\7z assignment\\Test\\machine4.csv")
machine_4.isnull().sum()
predictors_4= machine_4[['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W',]]
clf4= ExtraTreesClassifier().fit(predictors_4,machine_4['X'])
clf4.feature_importances_
model_4= SelectFromModel(clf4, prefit=True)
new_model4=model_4.transform(predictors_4)

df_machine4= pd.DataFrame(new_model4)
df_machine4.loc[:,'X']= pd.Series(machine_4['X'],index= df_machine4.index)
df_machine4.isnull().sum()

prediction_model_2= forest_2.predict(df_machine4)





clf3= ExtraTreesClassifier().fit(predictors_3,machine_3['X'])
clf3.feature_importances_
model_3= SelectFromModel(clf3, prefit=True)
new_model3=model_3.transform(predictors_3)

X_3=machine_3['X']
df_new_3= pd.DataFrame(new_model_3)
df_new_3.loc[:,'X']= pd.Series(X_3,index=df_new_3.index)
df_new_3.loc[:,'Y']= pd.Series(index=df_new_3.index)



df_new_3.isnull().sum()
df_new_3['X'].mode()
from scipy.stats import mode
df_new_3['X'].fillna(mode(df_new_3['X']).mode[0],inplace = True)


df_new_3=pd.read_csv('G:\\Datasets\\7z assignment\\Test\\machine3_answer.csv')
df_new_3=df_new_3.drop("Unnamed: 0",axis=1)

df_new_3.dtypes()

df_new_3['Y']= df_new_3['Y'].astype(float)
df_new_3['Y']=df_new_3['Y'].values

from sklearn import  svm
svm_clf= svm.SVC(decision_function_shape='ovo')
svm_clf.fit(new_df_machine_1[new_predictor_machine1],new_df_machine_1['X'])

X = [[0], [1], [2], [3]]
Y= [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes: 4*3/2 = 6
6
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes
4
