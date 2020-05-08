# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:33:56 2020

@author: ksdiv
"""

" code qui utilise impute pour fill in les tests features" 

# =============================================================================
# partie 1: impute ABPm, ABPd, ABPs entre eux car grande correlation
# partie 2: les autres valeurs impute tout ensemble 
# =============================================================================

import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

tf = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/train_features.csv")
tt = pd.read_csv("C:/Users/ksdiv/Documents/ETHZ/mast_sem_II/machine learning/task2/test_features_mean.csv")

pid=tt.loc[:,'pid']
pid=pid.to_frame()

#partie 1 

tfp1=tf.loc[:,['ABPm', 'ABPd','ABPs']]

impp1 = IterativeImputer(max_iter=10, random_state=0)
impp1.fit(tfp1)

ttp1=tt.loc[:,['ABPm', 'ABPd','ABPs']]
ttp1i=impp1.transform(ttp1)

col1=tfp1.columns
ttp1df= pd.DataFrame(np.array(ttp1i),columns=col1)
 
# =============================================================================
# donne des resultats bizarre et prend bcp de temps donc autant pas le faire
# from sklearn.impute import KNNImputer
# impk1 = KNNImputer(n_neighbors=2, weights="uniform")
# ttp1k= impk1.fit_transform(tfp1)
# =============================================================================


#partie 2
tfp2=tf.loc[:,['ABPm', 'ABPd','ABPs','Age','RRate','SpO2','Heartrate']]


impp2 = IterativeImputer(max_iter=10, random_state=0)
impp2.fit(tfp2)

ttp2=tt.loc[:,['ABPm', 'ABPd','ABPs','Age','RRate','SpO2','Heartrate']]
ttp2i=impp2.transform(ttp2)


col2=tfp2.columns
ttp2df= pd.DataFrame(np.array(ttp2i),columns=col2)

ttp2df['ABPm']=ttp1df['ABPm']
ttp2df['ABPd']=ttp1df['ABPd']
ttp2df['ABPs']=ttp1df['ABPs']

ttp2df=pid.join(ttp2df)
#ttp2df.to_csv('test_last4_rempli.csv', index=False)