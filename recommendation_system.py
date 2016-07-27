# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 23:00:38 2016

@author: kush
"""

import pandas as pd
import numpy as np

df= pd.read_excel("C:\\Users\\kush\\Desktop\\MovieReco_rawData.xlsx")

df.dtypes

df.describe()

def nullvalues(x):
    return sum(x.isnull())
df['language'].unique()    
    
df.apply(nullvalues)
#So there are only two NULL values in language


df[['language','Genre']].mode()

#if you see the ouput of mode values of 'language'.. its hindi. But if you see the actors they all are from Tamil industry.
#You cannot choose mode imputation to fill out values in language blindly. So i am using "tam" to fill out the missing values of language
#its simply two values so i just entered in excel sheet only.




#SKLEARN works on Integers only, so its must to convert our columns into nuumeric dataTypes
var_mod = ['Genre','language','Director']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in var_mod:
    df[i]=le.fit_transform(df[i])


#We divide the dataset into test and training. But in here if you divide the dataset, test data may not have all the Genre types which are there in train datset.
from sklearn import cross_validation as cv
train_data,test_data= cv.train_test_split(df,test_size=0.25)

#I am building my model on Genre as you can see language column have only 11 types of movie. People who watch Marathi are likely to watch Hindi movies also.
#people who watch Hindi movies watch punjabis too.. So recommending on Language will not be a good idea.
#if you choose Directors also, our resommendation model will be poor. As there are many Directors who have directed Tamil,Hindi and different movies. 
#So the only parameter we can choose is Genre to make our recommendation system.  
    
no_of_unique_generes = len(df['Genre'].unique())
no_of_unique_movies = len(df['Movie ID'].unique())

# Sparse matrix filled with zeros
movie_genre_relation_matrix =np.zeros((no_of_unique_movies, no_of_unique_generes))

#Mapping our data to dict and then mapping it to the Sparse matrix
i = 0
movies_List = df['Movie ID']
genres_List = df['Genre']

movie_genre_dict = {}
while (i < no_of_unique_movies):
    movie_genre_dict[movies_List[i]] = genres_List[i]    
    i = i + 1
 
for key,value in movie_genre_dict.items():
    movie_genre_relation_matrix[key-1][value] = 1
    
    
    


#The answer can be given with "MOVIE ID-Genre" matrix
#for each particular MOVIE ID with Genre, there are other IDS who have same Genre

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance.pdist import cosine

genre_movie_similarity = 1-pairwise_distances(movie_genre_relation_matrix, metric='cosine')


#genre-genre similairty matrix will give you how similar items are with each other to recommend user
new_data=pd.DataFrame(movie_genre_relation_matrix)
final_dataFrame= pd.DataFrame(data=new_data,index=new_data.columns,columns=new_data.columns)

genre_genre_similarity= 1- pairwise_distances(final_dataFrame,metric='cosine')














