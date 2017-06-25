# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 12:20:23 2017

@author: Luna
"""

import numpy as np
import sklearn as skl


import ps_data

#%%
#          1
# Получаем данные из PS
#

X1,Y1,CLs1 = ps_data.open_ps_2007()  
X = X1.as_matrix()
y =Y1.as_matrix()

#%%
#          2
#В моем распоряжении зашумленные данные, а задачу хочу попробовать решить SVM методом.
#SVM чувствителен к масштабированию
#Масштабируем данные
#