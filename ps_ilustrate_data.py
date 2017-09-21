# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:17:20 2017

@author: Luna
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import os

from sklearn import svm
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB

import ps_data
#%%

#os.chdir("c:\\Luna\\Work\\python\\PS")
os.chdir("d:\\Luna\\python\\PS")
cwd = os.getcwd()
os.chdir("..//")

os.chdir("PS_data")


#%%

X1,Y1,CLs1 = ps_data.open_ps_2007()  
X2,Y2,CLs2 = ps_data.open_ps_2009()    
#%%
def save_picture(g, name='',dr ='classes'):
    pwd = os.getcwd()
    iPath = '{}'.format(dr)
    if not os.path.exists(iPath):
        os.mkdir(iPath)
    os.chdir(iPath)
    g.savefig(name)
    os.chdir(pwd)
#%%
sns.set(style = "darkgrid", color_codes = True)
picture_foramt ='png'
colors = ['r','orange','y','g','c','b','violet','m','pink','#ee00ff']
l = len(X1.columns)
for xi in np.arange(l-1):
    for yi in np.arange(xi+1,l):
        print(xi,yi)
        xylim = [[X1.iloc[:,xi].min(),X1.iloc[:,xi].max()],[X1.iloc[:,yi].min(),X1.iloc[:,yi].max()]]
        for cl in CLs1:
          x =  X1[Y1["class_num"] == cl]
          g =  sns.jointplot(x.iloc[:,xi],x.iloc[:,yi],kind = "reg",color = colors[cl-1],size= 7)#,xlim = tuple(xylim[xi]),ylim = tuple(xylim[yi]),color = "b",size= 7)
          save_picture(g,'{}_{}_{}.{}'.format(cl,X1.columns[xi],X1.columns[yi],picture_foramt))
          save_picture(g,'{}_{}_{}.{}'.format(X1.columns[xi],X1.columns[yi],cl,picture_foramt),dr = 'features')
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
    