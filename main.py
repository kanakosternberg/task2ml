# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:37:04 2020

@author: ksdiv
"""

import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/train_features.csv")

#dft = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/test_features.csv")


#garde ça pour plus tard 
#treat differently missing features based on how many are missing 
a= df.isna().sum()
b=len(df)
c=a/b
t=b/12

# less than 20% missing-- replace with median of value for each specific patient 
ess=df
un1= pd.unique(df.loc[:,'pid'])

for id in un1:
    print(id)
    df2= np.where(df['pid']==id)
    df3=df.loc[df2[0],'EtCO2':'pH']
    dfm= df3.mean(axis=0)
    df4= df3.fillna(value=dfm)
    ess.update(df4)




    