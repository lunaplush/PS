# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:37:45 2018

@author: Luna
"""

#Experiments 
import pandas as ps
import numpy as np


# Generate experiments parameters

#Генерируем по ходу дела, а в  DataFrame записываем с номером эксперимента, чтобы можно было посмотреть все детали
#%%
data = [{'HIDDEN':i, 'HIDDEN2':j} for i in np.arange(3,30,3) for j in np.arange(3,30,3)]
df = pd.DataFrame(data)
data_s1 = [{'HIDDEN':i} for i in np.arange(3,30,3)]
data_s2 = [{'HIDDEN2':j} for i in np.arange(3,30,3)]
df_s = pd.DataFrame(data_s1)

datas_seria =  [ i for i in np.arange(3,30,3)]
df_s2 = pd.Series(datas_seria)
