# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:17:20 2017

@author: Luna
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
from sklearn import svm
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB

import ps_data

#%%

X1,Y1,CLs1 = ps_data.open_ps_2007()  
X2,Y2,CLs2 = ps_data.open_ps_2009()    
#%%
sns.set(style = "darkgrid", color_codes = True)
xylim = [[X1['RON'].min(),X1['RON'].max()],[X1['MON'].min(),X1['MON'].max()]]
x =  X1[Y1["class_num"] == 1]    
xi = 0
yi = 1
g =  sns.jointplot(x.iloc[:,xi],x.iloc[:,yi],kind = "reg",color = "b",size= 7)#,xlim = tuple(xylim[xi]),ylim = tuple(xylim[yi]),color = "b",size= 7)
g.savefig("test.png")
#add hue
#sns.pairplot(X1).savefig("pairplot_2007.png")

#sns.pairplot(X2).savefig("pairplot_2009.png")
#%%
#def plot_X(X, cols,xylim,y_name):
#    assert(X.shape[1] == 2)
#    f, ax = plt.subplots(figsize=(7, 7))
#    ax.scatter(X.iloc[:,0], X.iloc[:,1], \
#        marker='o', color='green', s=4, alpha=0.3)
#    plt.title('PS_examples')
#    plt.xlabel(cols[0])
#    plt.ylabel(cols[1])
#  #  ftext = 'p(x) ~ N(mu=(0,0)^t, cov=I)'
#  #  plt.figtext(.15,.85, ftext, fontsize=11, ha='left')
#    plt.ylim(xylim[1])
#    plt.xlim(xylim[0])
#    plt.title(y_name)
#    
#    plt.show()

#xylim = [[X1['RON'].min(),X1['RON'].max()],[X1['MON'].min(),X1['MON'].max()]]
#for i in CLs1:
#    y = i
#    
#    X =  X1[Y1["class_num"] == y]    
#    cols = ['RON','MON'] 
#    plot_X(X[cols],cols,xylim,CLs1[y])
    