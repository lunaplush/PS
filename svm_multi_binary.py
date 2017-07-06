# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 12:20:23 2017

@author: Luna
"""

import numpy as np
import sklearn as skl
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import ps_data
#%%
#%%
#          1
# Получаем данные из PS
#

X1,Y1,CLs1 = ps_data.open_ps_2007()  


#%%

#второй важны вопрос - нужно ли балансировать выборку. 
Flag_balance = True
if Flag_balance :
        
    a =  Y1.groupby("class_num")
    mn = max(a.apply(len))
    
    for i in a:
        n = len(i[1] )
        
        dl = int(mn/n)
        for j in np.arange(dl - 1):
            X1 =  X1.append(X1[Y1["class_num"] == i[0]].iloc[0:n])
            Y1 =  Y1.append(Y1[Y1["class_num"] == i[0]].iloc[0:n])
            
        X1 =  X1.append(X1[Y1["class_num"] == i[0]].iloc[0:mn - n*(dl)])
        Y1 =  Y1.append(Y1[Y1["class_num"] == i[0]].iloc[0:mn - n*(dl)])


#%%
X = X1.as_matrix()
Y = Y1.as_matrix()
        #%%
#          2
#В моем распоряжении зашумленные данные, а задачу хочу попробовать решить SVM методом.
#SVM чувствителен к масштабированию svm_base_report_Scaler_compare.txt эксперимент.
# в свою очередь результат масштабрования сильно зависит от  наличия выбросов в исходных данных
#Масштабируем данные сразу все, без разделения на обучающую и тестовую
# 
Flag_scale = True
if  Flag_scale:
        
    coder = StandardScaler()
    coder.fit(X)
    X = coder.transform(X)

    
#%%
# Для выполнения экспериментов разобъем выборку на обучающую и тестовую

(X_train, X_test, y_train, y_test) = train_test_split(X,Y,test_size = 0.3, random_state = 0, stratify= Y)


#%%
#1
#решим задачу для первых двух классов по нескольким разным парам признаков
#используем линейное ядро, визуализируем

#используем SVM c линейным ядром на двух признаках для двух классов
# немного сравнивала результаты на сбалансировнной выборке и нет, разница незаметна, хотя она есть.


def simple_svc(x,y):
    print(x.shape)
    clf = svm.SVC(kernel='linear')
    clf.fit(x, y)
    w = clf.coef_[0]
    print(w)
    a = -w[0] / w[1]
    xx = np.linspace(-2.5, 2.5)
    yy = a * xx - clf.intercept_[0] / w[1]

    h0 = plt.plot(xx, yy, 'k-', label='test')    
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
    plt.legend()
    
    plt.axis('tight')
    plt.show()
    y_pred = clf.predict(x)
    print(classification_report(y, y_pred))
classes = [1,9]    
y_ind = np.logical_or(Y == classes[0], Y == classes[1])
x = X[y_ind.reshape(len(y_ind))]
x = x[:,[3,5]]
y = Y[y_ind]
#simple_svc(x,Y[y_ind])

#%% 
#Решим задачу мультиклассовой классификации разбив на пыры и построив матрицу решений
 
params = {'kernel':'rbf',"C": 8,"gamma":0.125}   
features_ind = np.arange(17)
classes = list(CLs1.keys())
M = len(classes)
CLF = np.empty((M,M), dtype = object)
# Строим матрицу попарных классификаторо
for cl1 in np.arange(1,M):
    for cl2 in np.arange(cl1+1,M+1):
        y_ind = np.logical_or(y_train == cl1, y_train == cl2)
        x = X_train[y_ind.reshape(len(y_ind))]
        x = x[:,features_ind]
        y =  y_train[y_ind]
        CLF[cl1 - 1,cl2 -1] = SVC(**params)
        CLF[cl1 - 1,cl2 -1].fit(x,y)
        #%%
        #Результат
y_predict1 = CLF[0,1].predict(X_test)

MM = len(y_predict1)
y_predict1 = y_predict1.reshape(MM,1)
for cl1 in np.arange(1,M):
    for cl2 in np.arange(cl1+1,M+1):        
        if (cl1 == 1 and cl2 != 2) or cl1!=1:           
           y_predict1 = np.concatenate((y_predict1,CLF[cl1 - 1,cl2 -1].predict(X_test).reshape(MM,1)),axis = 1)

l = len(y_predict1)
y_predict_max = np.ones(l)         
for i in np.arange(l):
    h = np.histogram(y_predict1[i], bins = list(CLs1.keys())+[1000])
    y_predict_max[i] =int(h[1][np.argmax(h[0])]         )
#%%
print(classification_report(y_test, y_predict_max, target_names = list(CLs1.values())))
#%%

clf = SVC(**params)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names = list(CLs1.values())))
