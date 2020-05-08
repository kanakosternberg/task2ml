# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:56:30 2020

@author: ksdiv
"""
import pandas as pd
import numpy as np

fpt = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/feature_par_test.csv")
tef = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/test_features.csv")

pid=tef.loc[:,'pid']
pid=pid.to_frame()
tef= tef.loc[:,'Age':'pH']

col=tef.columns

us_val_rrate= fpt.loc[:,'LABEL_RRate']
us_val_rrate = us_val_rrate.astype(int)

us_val_abpm= fpt.loc[:,'LABEL_ABPm']
us_val_abpm= us_val_abpm.astype(int)

us_val_heartrate= fpt.loc[:,'LABEL_Heartrate']
us_val_heartrate = us_val_heartrate.astype(int)

val_gard= us_val_abpm+us_val_heartrate+us_val_rrate

val_gard= pd.DataFrame(np.array(val_gard),index=col)
val_gard= val_gard.transpose()

tefr=pd.concat([val_gard,tef])
tefr= tefr.loc[:, tefr.iloc[0,:] >= 1]
tefr=tefr.iloc[1:,:]
tefr=pid.join(tefr)

un1= pd.unique(tefr.loc[:,'pid'])
train_upd=pd.DataFrame()   #fait av valeurs test

 
for id in un1:
    print(id)
    df2= np.where(tefr['pid']==id)
    df3=tefr.loc[df2[0],tefr.columns != 'pid']
    dfm= df3.mean(axis=0)
    dfm=dfm.to_frame()
     
    dfm.loc['pid']=id
    dfm=dfm.transpose()
    train_upd=train_upd.append(dfm)
 
 # pense peut impute pour recuperer les quelques valeures manquantes 
 
#train_upd.to_csv('test_features_mean.csv', index=False)
