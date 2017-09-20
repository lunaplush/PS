# -*- coding: utf-8 -*-
"""
Parzen Window
Created on Sun Jun  4 11:28:17 2017

@author: Inspiron
"""
import numpy as np
import matplotlib.pyplot as plt


import ps_data

#%%

X1,Y1,CLs1 = ps_data.open_ps_2007()  
X2,Y2,CLs2 = ps_data.open_ps_2009()    

#%%


def plot_X(X, cols,xylim,y_name):
    assert(X.shape[1] == 2)
    f, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X.iloc[:,0], X.iloc[:,1], \
        marker='o', color='green', s=4, alpha=0.3)
    plt.title('PS_examples')
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
  #  ftext = 'p(x) ~ N(mu=(0,0)^t, cov=I)'
  #  plt.figtext(.15,.85, ftext, fontsize=11, ha='left')
    plt.ylim(xylim[1])
    plt.xlim(xylim[0])
    plt.title(y_name)
    
    plt.show()

#xylim = [[X1['RON'].min(),X1['RON'].max()],[X1['MON'].min(),X1['MON'].max()]]
#for i in CLs1:
#    y = i
#    
#    X =  X1[Y1["class_num"] == y]    
#    cols = ['RON','MON'] 
#    plot_X(X[cols],cols,xylim,CLs1[y])
#    