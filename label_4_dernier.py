# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:15:02 2020

@author: ksdiv
"""

#label Rrate ABPm Heartrate

#RidgeCV ça fonctionnait bien nan 
import pandas as pd
import numpy as np

fpt = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/feature_par_test.csv")
tl = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/train_labels_aug.csv")
tf = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/train_features_cleaned.csv")
tef = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/test_features.csv")

pid=tf.loc[:,'pid']
pid=pid.to_frame()
tl=tl.loc[:,'LABEL_BaseExcess':'LABEL_Heartrate']
tf= tf.loc[:,'Age':'pH']
tef= tef.loc[:,'Age':'pH']

col=tf.columns
# Rrate 
# "preprocessing" 
us_val_rrate= fpt.loc[:,'LABEL_RRate']
us_val_rrate = us_val_rrate.astype(int)
#test = np.array(us_val_rrate)
us_val_index= pd.DataFrame(np.array(us_val_rrate),index=col)
us_val_index= us_val_index.transpose()
tf=pd.concat([us_val_index,tf])
tefr=pd.concat([us_val_index,tef])

tfc= tf.loc[:, tf.iloc[0,:] == 1]
tfc=tfc.iloc[1:,:]
LRRate_int= tfc.join(tl.loc[:,'LABEL_RRate'])
LRRate_int= pid.join(LRRate_int)
LRRate_intt = LRRate_int.dropna()

tefr= tefr.loc[:, tefr.iloc[0,:] == 1]
tefr=tefr.iloc[1:,:]
tefr=pid.join(tefr)
TRRate = tefr.dropna()
XTRRate= TRRate.iloc[:,1:]
pid_t= TRRate.loc[:,'pid']  #get patient ids which are kept 

YRRate=LRRate_intt.iloc[:,-1]
XRRate=LRRate_intt.iloc[:,1:-1]

# =============================================================================
# from sklearn.linear_model import RidgeCV
# clfRRate = RidgeCV().fit(XRRate, YRRate)
# res_rrate= clfRRate.predict(XTRRate)
# =============================================================================

