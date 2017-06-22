# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 08:59:12 2017

@author: Inspiron
"""

from sklearn import svm
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
#%%
X, y = make_classification(n_samples=1000,n_features =2,n_informative=2, n_redundant=0, random_state = 0 )
y = 2 * (y - 0.5)
clf = svm.SVC()
clf.fit(X, y) 
#clf.predict(X[10])
#f = clf.support_vectors_
#%%
plt.scatter(X[:,0][y == -1],X[:,1][y==-1], color = 'r')
plt.scatter(X[:,0][y == 1],X[:,1][y==1], color = 'b') 
#%%
clf.