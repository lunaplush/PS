# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 10:27:14 2017

@author: Inspiron
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
plt.close('all')
x = np.linspace(-1,2,100)
y = np.exp(x)
f, ax = plt.subplots(1,2,sharex=True)

ax[0].semilogy(x,y)
ax[1].plot(x,y)
y1 = np.log(x)
ax[0].scatter(x,y1)
ax[1].semilogy(x,y1)

y2 = np.log10(x)
ax[0].scatter(x,y2)
ax[1].semilogy(x,y2)

print(np.log(1))

#146 400 000