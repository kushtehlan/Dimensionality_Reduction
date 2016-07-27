# -*- coding: utf-8 -*-
"""
Created on Tue May 31 01:22:08 2016

@author: kush
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd 
import numpy as np
import json
ai5 = []
event =[]
ts =[]
time=[]
game =[]
listh =[]
listp =[]
listb =[]

#code is to clean the data and make "final_csv.csv" file
for line in open("C:\\Users\\kush\\Desktop\\ggevent.log","r+"):
    temp =[]
    temp1 =[]
    temp2 = []
    tt=(json.loads(line))
    for each in tt['headers'].values():
        temp.append(each)
    for each in tt['post'].values():
        temp1.append(each)
    for each in tt['bottle'].values():
        temp2.append(each)
    listh.append(temp)
    listp.append(temp1)
    listb.append(temp2)
    
for i in range(len(listh)):
    ai5.append(listh[i][0])
for i in range(len(listp)):
    event.append(listp[i][0])
    ts.append(listp[i][1])
for i in range(len(listb)):
    time.append(listb[i][0])
    game.append(listb[i][1])
    
print ai5,"\n",event,"\n",ts,"\n",time,"\n",game,"\n"
final_df = pd.DataFrame(index=np.arange(len(listh)),columns=["ai5","event","ts","time","game"])
for i in range(len(final_df)):
    final_df["ai5"][i]= ai5[i]
    final_df["event"][i]= event[i]
    final_df["ts"][i]= ts[i]
    final_df["time"][i]= time[i]
    final_df["game"][i]= game[i]
final_df.to_csv("C:\\Users\\kush\\Desktop\\finalcsv.csv",index= False)
print "Done"



main_data = pd.read_csv("C:\\Users\\kush\\Desktop\\finalcsv.csv")

ans1=main_data.groupby(['game'])['ai5'].nunique()



main_data_duplicate = main_data
df1= main_data_duplicate
df1['time']= pd.to_datetime(main_data_duplicate['time'], format='%Y-%m-%d %H:%M:%S.%f')
df2 =pd.pivot_table(df1,index=['ai5','game'],columns='event',values='time',aggfunc='first').reset_index()

df2['sessions']= df2['ggstop'] - df2['ggstart'] 
print df2

df3=df2['sessions'].mean() 
print df3

df4 =df2[pd.notnull(df2['sessions'])] #not null values  
df5 = df4.mean() #average for not null sessions
print df5



games = list(main_data['game'])
users= list(main_data['ai5'])
unique_game = set(games)

  

game_unique = []
user_unique = []
   
for i in unique_game:
    part_data = main_data[main_data['game']== i]
    uniqueusers= set(part_data['ai5'])
    if len(uniqueusers) == 1:
        game_unique.append(i)
        user_unique.append(uniqueusers)
        
print game_unique
print user_unique
