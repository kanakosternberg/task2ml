# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:14:15 2020

@author: ksdiv
"""

import pandas as pd
import numpy as np

dl = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/train_labels.csv")
newdf = pd.DataFrame(np.repeat(dl.values,12,axis=0))
newdf.columns = dl.columns
newdf2=newdf.loc[:,newdf.columns != 'pid']
#newdf.to_csv('train_labels_aug.csv', index=False)

dfo = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/train_features.csv")
dfp=dfo.join(newdf2)
dfp.to_csv('train_aug.csv', index=False)