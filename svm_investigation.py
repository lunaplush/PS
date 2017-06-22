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
plt.figure(1)
plt.scatter(X[:,0][y == -1],X[:,1][y==-1], color = 'r')
plt.scatter(X[:,0][y == 1],X[:,1][y==1], color = 'b') 

plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1], color= 'g',  alpha = 0.3)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
 
# импортируем набор данных (например, возьмём тот же iris)
iris = datasets.load_iris()
X = iris.data[:, :2] # возьмём только первые 2 признака, чтобы проще воспринять вывод
y = iris.target
 
C = 1.0 # параметр регуляризации SVM
svc = svm.SVC(kernel='linear', C=1,gamma=0).fit(X, y) # здесь мы взяли линейный kernel
 
# создаём сетку для построения графика
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))
 
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
 
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length') # оси и название укажем на английском
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()