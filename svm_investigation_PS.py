# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 08:59:12 2017

@author: Inspiron
"""
#Визуализация по проекциям на данных PS
from sklearn import svm
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import ps_data
#%%
X1,Y1,CLs = ps_data.open_ps_2007()
X = X1.as_matrix()
y =Y1.as_matrix()
#%%

clf = svm.SVC()
clf.fit(X, y) 
#clf.predict(X[10])
#f = clf.support_vectors_

#%%
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)


plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1], color= 'g')#marker ="^",  alpha = 0.7)

#%%
# В данном блоке можно увидеть границы полученного классификатора.
# Для многомерной задачи (признаков больше 2 х) возможно построить только проекцию.
# Для этого я добавляю xy_pr значения всех признаков с любым одним значением из обучающей выборки. 
# К сожалению, данная процедура не дает наглядной картины
# Лучше использовать это только для класификаторов по два
plt.figure() 

# создаём сетку для построения графика
h = 0.1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
 
plt.subplot(1, 1, 1)
xy = np.c_[xx.ravel(), yy.ravel()]
pr_el = 1000           
xy_pr = np.ones((xy.shape[0], X.shape[1]-2))*X[pr_el,2:17]
xy_add = np.concatenate((xy,xy_pr),axis = 1)
           
Z = clf.predict(xy_add)
Z = Z.reshape(xx.shape)

#%%
eps = 2
ind_x = np.array([False]*len(X))
for  i in np.arange(len(X)):
    flag = True
    for j in np.arange(X.shape[1]-2):
        if X[i,j+2] > X[pr_el,j+2] + eps or X[i,j+2] < X[pr_el,j+2] - eps:
           
            flag = False
    ind_x[i] = flag
#%%
#Нужно понять соответсвие цветов и классов
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
 
plt.scatter(X[ind_x, 0], X[ind_x, 1], c=y[ind_x], cmap=plt.cm.Paired)
plt.xlabel(X1.columns[0]) # оси и название укажем на английском
plt.ylabel(X1.columns[1])
plt.xlim(xx.min(), xx.max())
plt.title('SVM projection on X[%d]' % pr_el)
plt.show()
#%%
y_pred = clf.predict(X)
print(classification_report(y, y_pred, target_names = list(CLs.values())))
    
     
    
    