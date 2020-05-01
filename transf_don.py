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
xx_hgb=x_hgb.loc[:,~'Hgb']