# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:58:18 2020

@author: ksdiv
"""

import pandas as pd
import numpy as np


df = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/train_features_cleaned.csv")

a= df.isna().sum()
b=len(df)
c=a/b

# decide de remplir avec knn seulement les valeurs qui sont en dessous de 40% inconnues

val_knn=c [(c < 0.4) & (c > 0.1)] #impression que manque que tres peu de valeurs on s'en balec 

val_sshgb=df[np.isnan(df['Hgb'])]
x_hgb=df[~np.isnan(df['Hgb'])]
xy_hgb=x_hgb.loc[:,'Hgb']
xx_hgb=x_hgb.loc[:,x_hgb.columns != 'Hgb']

# =============================================================================
# 
# from sklearn.neighbors import KNeighborsRegressor
# neigh = KNeighborsRegressor(n_neighbors=2)
# neigh.fit(xx_hgb, xy_hgb)
# a= neigh.predict(val_sshgb)
# 
# 
# 
# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=2, weights="uniform")
# imputer.fit_transform(df)
# 
# =============================================================================

dfo = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/train_features.csv")
C= dfo.corr()

dl = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/train_labels_aug.csv")
L= dl.corr()

da = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/train_aug.csv")
A=da.corr()

ao= dfo.isna().sum()
bo=len(dfo)
co=ao/bo

A_cut= A.loc['Age':'pH','LABEL_BaseExcess':'LABEL_Heartrate']
A_cut=A_cut.abs()

max_A=A_cut.max()
val_ut= A_cut.gt(0.3*max_A)  #garde tous les features qui ont au moins 30% de correlation que le feature qui a le plus de correlation pour ce test

val_inutile= val_ut[val_ut==True].sum(axis=1)
val_ut.to_csv('feature_par_test.csv', index=True)















