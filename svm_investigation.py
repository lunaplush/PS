# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 08:59:12 2017

@author: Inspiron
"""

from sklearn import svm
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
#%%
X, y = make_classification(n_samples=100,n_features =2,n_informative=2, n_redundant=0, random_state = 11, n_classes = 3 , n_clusters_per_class = 1)
y = 2 * (y - 0.5)
CLs = {1:"fisrt", -1:"second", 3:"third"}
clf = svm.SVC()
clf.fit(X, y) 
#clf.predict(X[10])
#f = clf.support_vectors_
#%%
#X1,Y1,CLs1 = ps_data.open_ps_2007()
#X=X1
#Y=Y1
#CLs= CLs1

#%%
plt.figure(1)
plt.scatter(X[:,0][y == -1],X[:,1][y==-1], color = 'r')
plt.scatter(X[:,0][y == 1],X[:,1][y==1], color = 'b') 

plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1], color= 'g')#marker ="^",  alpha = 0.7)

#%%
plt.figure() 

# создаём сетку для построения графика
h = 0.1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
 
plt.subplot(1, 1, 1)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#%%
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt 
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('X1') # оси и название укажем на английском
plt.ylabel('X2')
plt.xlim(xx.min(), xx.max())
plt.title('SVM investifation')
indices = [10,20,30]
colors = ['g',"k",'y']#'fuchsia','peachpuff']
i = 0
for x_test_ind in indices:
    plt.scatter(X[x_test_ind,0],X[x_test_ind,1],color = colors[i], marker ="^")
    df = clf.decision_function([X[x_test_ind,0],X[x_test_ind,1]])
    plt.annotate("SVM res %f,%f,%f" % (df[0][0],df[0][1],df[0][2]), \
                 xy = (X[x_test_ind,0] + 0.3,X[x_test_ind,1]+0.3))#,xy = (X[x_test_ind,0]+1,X[x_test_ind,1]+1))#, xytext = (X[x_test_ind,0]+0.3,X[x_test_ind,1]+0.3))
    i= i+1
plt.show()
#%%
y_pred = clf.predict(X)
print(classification_report(y, y_pred, target_names = list(CLs.values())))
