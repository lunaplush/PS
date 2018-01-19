# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 10:27:14 2017

@author: Inspiron
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%

df = pd.DataFrame({'X':np.random.standard_normal(100), 'Y':np.random.standard_cauchy(100)}, index = np.linspace(1,50,100))
df.plot('X','Y', kind = 'scatter')
df.plot()


