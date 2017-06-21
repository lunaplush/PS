# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 22:03:52 2017

@author: Inspiron
Первый эксперимент с SVM для данных 2007 года 
 
Результат 97% качество распознавания на обучении и 93 на тесте. 
При этом параметры алгоритма params = {'kernel':'rbf',"C":512.0,"gamma":0.015625}

"""
#import os
#os.chdir("c:\\Luna\\Work\\python\\PS\\")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as skl
import math
#import utilities


from sklearn import svm
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_recall_fscore_support

#%%
file = "PS 2007_Р.xlsx"

xfile = pd.ExcelFile(file)

df_cols = xfile.parse(sheetname = "columns", header = 3)
columns = df_cols.columns


PS = xfile.parse(sheetname = "PetroSpec07", skiprows = [0,1,2], parse_cols = "A:V", names = columns )
#Некоторая предобработка значений
PS["name"] = PS["name"].str.lower()
PS["provider"] = PS["provider"].str.lower()
PS["name"] = PS["name"].str.strip()
PS["provider"] = PS["provider"].str.strip()

gr = PS.groupby(["name","provider"])
PS_new = pd.DataFrame()
Class_min_size = 100
for i in gr:
  
    if len(i[1]) > Class_min_size and i[0][0]!= "бензин" and i[0][1] != "заказчик":
       PS_new =  PS_new.append(i[1], ignore_index = True)




#%%"
features = ["ro","RON", "MON", "OLF","ALK","AROMA","BNZ","KIS", \
            "TLL","MASS","METANOL","ETANOL","MTBE","ETBE","TAME","DIPE","TBS"]
X = PS_new[features]
N = len(PS_new)
CLs = {}

cl_num = 0
for i in range(N):
    ind = PS_new.iloc[i][["name", "provider"]]
    list_ind =  list(ind.values)
    if list_ind not in CLs.values():
        cl_num = cl_num+1
        CLs[cl_num] = list_ind
       
def get_key(d, value):
    for k, v in d.items():
        if v == value:
            return k

CLs_ch =  lambda a: get_key(CLs,list(a.values)) 
Y = PS_new[["name","provider"]].apply(CLs_ch, axis = 1)
Y.name = "class_num"

Y = pd.DataFrame(Y)

#%%
#X преобразовать в цифру и заполнить пропуски
def to_float(x):
    try:
        return float(str(x).replace(",","."))
    except:
        return np.NAN

for col in features:    
    X[col] = X[col].apply(to_float)
#%% Анализ неверных значений

Crit1 =  np.logical_and(X['ro']>650, X['ro']< 900)
Crit2 = np.logical_or(Crit1, X['ro'].isnull())
X = X[Crit2]
Y = Y[Crit2]
#%%
for cl in CLs:
    x = X[Y["class_num"] == cl]
    means = (np.max(x) - np.min(x))/2 +np.min(x)
    X[Y["class_num"] == cl] = X[Y["class_num"] == cl].fillna(means)
    # means = calculate_means(X[Y["class_num"] == cl][features]) 
   
    #X[Y["class_num"] == cl][features] = X[Y["class_num"] == cl][features].fillna(means)  
#means = calculate_means(X[features])
#X[features] = X[features].fillna(means) #модифицировать средним по классам

#X[features] = X[features].fillna(0.0) #модифицировать средним по классам
#%%
CCC=X.corr()
# Сильная корреляция обнаружена между : MTBE MASS 0.78 и KIS TLL 0.87
del X["MASS"]
del X["KIS"] # По коэффициенту корреляции ксиолы и толулы сильно коррелируют,
             # но в реальности это разные составляющие
             # Может быть и не стоит убирать. 
#1 METANOL
metanol = X["METANOL"].unique() #Единственное значение  0
del X["METANOL"]


#%% Предварительное разбиен ие на тестовую и обучающую выборки
(X_train, X_test, y_train, y_test) = train_test_split(X,Y,test_size = 0.3, random_state = 0, stratify=Y)
  
#%% Масштабирование

coder = StandardScaler()
coder.fit(X_train)
X_train_scaled = coder.transform(X_train)
X_test_scaled =  coder.transform(X_test)
#%%
params = {'kernel':'rbf',"C":512.0,"gamma":0.015625} 
#params = {'C': 8192.0, 'gamma': 0.001953125}
classifier = SVC()
def best_params():
    
    cv = skl.cross_validation.StratifiedShuffleSplit(y_train, n_iter = 10, test_size = 0.2, random_state = 0)
    parameters_grid = {
        'gamma' :pow(2.0,np.arange(-15,3)),   
        'C' : pow(2.0,np.arange(-5,15,2))
    }
    grid_cv = skl.grid_search.GridSearchCV(classifier, parameters_grid, scoring = 'accuracy', cv = cv)  
    grid_cv.fit(X_train_scaled, y_train)
   
    #print(grid_cv.best_estimator_)
    #print(grid_cv.best_score_)
    #print(grid_cv.best_params_)
    #print(grid_cv.grid_scores_[:10])
    return grid_cv.best_params_
#params = best_params()

#Результат закомментированного вконце алгоритма поиска оптимальные параметров по статье 
#C.-W. Hsu, C.-C. Chang, C.-J. Lin. A practical guide to support vector classification . Technical report, Department of Computer Science, National Taiwan University. July, 2003.
# A Practical Guide to Support Vector Classification. Chih-Wei Hsu, Chih-Chung Chang, and Chih-Jen Lin. Department of Computer Science.
#http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
#Initial version: 2003 Last updated: May 19, 2016

classifier = SVC(**params)
classifier.fit(X_train_scaled, y_train)
#utilities.plot_classifier(classifier,X_train_scaled,y_train, "TR")
#plt.show()
target_names = ["CLass - " + str(int(i)) for i in CLs.keys()]
print("\n" +"#"*30)
print("Perfomance on train set")
s = []
for v in CLs.values():
    s.append(str(v))
y_pred = classifier.predict(X_train_scaled)    
print(classification_report(y_train, y_pred, target_names = s))
print(classification_report(y_test, classifier.predict(X_test_scaled), target_names = s))
#print(precision_recall_fscore_support(y_train, y_pred))
Y_res = y_train.copy()
Y_res["pred"] = y_pred
#%%

#clf = svm.SVC(probability=True, random_state=0)
#DD = cross_val_score(clf, X, Y, scoring='neg_log_loss') 

#%%

##classifier.get_params().keys()) 
##dict_keys(['random_state', 'shrinking', 'class_weight', 'decision_function_shape', 'tol', 'verbose',
##             'coef0', 'degree', 'cache_size', 'probability', 'max_iter', 'kernel', 'gamma', 'C'])
#cv = skl.cross_validation.StratifiedShuffleSplit(X_train_scaled, n_iter = 10, test_size = 0.2, random_state = 0)
#parameters_grid = {
#    'gamma' :pow(2.0,np.arange(-15,3)),   
#    'C' : pow(2.0,np.arange(-5,15,2))
#}
#grid_cv = skl.grid_search.GridSearchCV(classifier, parameters_grid, scoring = 'accuracy', cv = cv)  
#grid_cv.fit(X_train_scaЬЕН88ЖЖ8ШЗАЩНШЕШШШШНШЕled, y_train)
#
#print(grid_cv.best_estimator_)
#print(grid_cv.best_score_)
#print(grid_cv.best_params_)
#print(grid_cv.grid_scores_[:10])
##0.935585585586
##{'gamma': 0.015625, 'C': 512.0}
#print(classification_report(y_train,grid_cv.best_estimator_(X_train_scaled) , target_names = s))