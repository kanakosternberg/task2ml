# -*- coding: utf-8 -*-
"""
Created on Fri May  8 23:37:29 2020

@author: ksdiv
"""

"""
trouve resultats des 4 derniers tests en utilisant RidgeCV 

"""

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV

fpt = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/feature_par_test.csv")
ttef = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/test_last4_rempli.csv")
tf = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/train_features.csv")
tl = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/train_labels_aug.csv")


pid=ttef.loc[:,'pid']
pid=pid.to_frame()

tf= tf.loc[:,'Age':'pH']
col=tf.columns

# Rrate 

# train 

us_val_rrate= fpt.loc[:,'LABEL_RRate']
us_val_rrate = us_val_rrate.astype(int)
us_val_index= pd.DataFrame(np.array(us_val_rrate),index=col)
us_val_index= us_val_index.transpose()
tfrr=pd.concat([us_val_index,tf])

tfc= tfrr.loc[:, tfrr.iloc[0,:] == 1]
tfc=tfc.iloc[1:,:]
LRRate_int= tfc.join(tl.loc[:,'LABEL_RRate'])
LRRate_intt = LRRate_int.dropna()

YRRate=LRRate_intt.iloc[:,-1]
XRRate=LRRate_intt.iloc[:,:-1]

# test 
ttefrr=pd.concat([us_val_index,ttef],sort=False)
ttefrr= ttefrr.loc[:, ttefrr.iloc[0,:] == 1]
Xttefrr=ttefrr.iloc[1:,:]


clfRRate = RidgeCV().fit(XRRate, YRRate)
res_rrate= clfRRate.predict(Xttefrr)


# ABPm

# train 
us_val_abpm= fpt.loc[:,'LABEL_ABPm']
us_val_abpm = us_val_abpm.astype(int)
us_val_index_abpm= pd.DataFrame(np.array(us_val_abpm),index=col)
us_val_index_abpm= us_val_index_abpm.transpose()
tfabpm=pd.concat([us_val_index_abpm,tf])

tfc_abpm= tfabpm.loc[:, tfabpm.iloc[0,:] == 1]
tfc_abpm=tfc_abpm.iloc[1:,:]
LABPm_int= tfc_abpm.join(tl.loc[:,'LABEL_ABPm'])
LABPm_intt = LABPm_int.dropna()

YABPm=LABPm_intt.iloc[:,-1]
XABPm=LABPm_intt.iloc[:,:-1]

# test 
ttefabpm=pd.concat([us_val_index_abpm,ttef],sort=False)
ttefabpm= ttefabpm.loc[:, ttefabpm.iloc[0,:] == 1]
Xttefabpm=ttefabpm.iloc[1:,:]


clfABPm = RidgeCV().fit(XABPm, YABPm)
res_abpm= clfABPm.predict(Xttefabpm)


#SpO2

# train 
us_val_spo= fpt.loc[:,'LABEL_SpO2']
us_val_spo = us_val_spo.astype(int)
#change manuellement parce que ça nous arrange 
us_val_spo.iloc[4]=0
us_val_spo.iloc[34]=0
us_val_index_spo= pd.DataFrame(np.array(us_val_spo),index=col)
us_val_index_spo= us_val_index_spo.transpose()
tfspo=pd.concat([us_val_index_spo,tf])

tfc_spo= tfspo.loc[:, tfspo.iloc[0,:] == 1]
tfc_spo=tfc_spo.iloc[1:,:]
LSpO2_int= tfc_spo.join(tl.loc[:,'LABEL_SpO2'])
LSpO2_intt = LSpO2_int.dropna()

YSpO2=LSpO2_intt.iloc[:,-1]
XSpO2=LSpO2_intt.iloc[:,:-1]

# test 
ttefspo=pd.concat([us_val_index_spo,ttef],sort=False)
ttefspo= ttefspo.loc[:, ttefspo.iloc[0,:] == 1]
Xttefspo=ttefspo.iloc[1:,:]


clfSpO2 = RidgeCV().fit(XSpO2, YSpO2)
res_spo= clfSpO2.predict(Xttefspo)

#Heartrate

# train 
us_val_hrt= fpt.loc[:,'LABEL_Heartrate']
us_val_hrt = us_val_hrt.astype(int)
us_val_index_hrt= pd.DataFrame(np.array(us_val_hrt),index=col)
us_val_index_hrt= us_val_index_hrt.transpose()
tfhrt=pd.concat([us_val_index_hrt,tf])

tfc_hrt= tfhrt.loc[:, tfhrt.iloc[0,:] == 1]
tfc_hrt=tfc_hrt.iloc[1:,:]
LHeartrate_int= tfc_hrt.join(tl.loc[:,'LABEL_Heartrate'])
LHeartrate_intt = LHeartrate_int.dropna()

YHeartrate=LHeartrate_intt.iloc[:,-1]
XHeartrate=LHeartrate_intt.iloc[:,:-1]

# test 
ttefhrt=pd.concat([us_val_index_hrt,ttef],sort=False)
ttefhrt= ttefhrt.loc[:, ttefhrt.iloc[0,:] == 1]
Xttefhrt=ttefhrt.iloc[1:,:]


clfHeartrate = RidgeCV().fit(XHeartrate, YHeartrate)
res_hrt= clfHeartrate.predict(Xttefhrt)

d = {'LABEL_RRate': np.array(res_rrate), 'LABEL_ABPm': np.array(res_abpm),'LABEL_SpO2': np.array(res_spo),'LABEL_Heartrate': np.array(res_hrt)}
RES_L4=pd.DataFrame(data=d)
RES_L4.to_csv('RES_L4.csv', index=False)