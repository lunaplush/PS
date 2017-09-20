# -*- coding: utf-8 -*-
"""
Created on Mon May 15 20:22:16 2017

@author: Inspiron
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 13 13:35:14 2017
Второй эксперимент. Добавим размножение выборки до одинакового количества примеров в каждом классе.

@author: Inspiron
"""
# -*- coding: utf-8 -*-

#import os
#os.chdir("c:\\Luna\\Work\\python\\PS\\")


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as skl
import itertools
#import utilities
import parzen

from sklearn import svm
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
#%%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#%%

file = "PS 09.xls"

xfile = pd.ExcelFile(file)

df_cols = xfile.parse(sheetname = "columns", header = 3)
columns = df_cols.columns
features = ["ro","RON", "MON", "OLF","ALK","AROMA","BNZ","KIS", \
            "TLL","MASS","METANOL","ETANOL","MTBE","ETBE","TAME","DIPE","TBS"]

PS = xfile.parse(sheetname = "PetroCpec 09", skiprows = [0,1,2], parse_cols = "A:W", names = columns )
#Некоторая предобработка значений
PS["name"] = PS["name"].str.lower()
PS["provider"] = PS["provider"].str.lower()
PS["name"] = PS["name"].str.strip()
PS["provider"] = PS["provider"].str.strip()

gr = PS.groupby(["name","provider"])
PS_new = pd.DataFrame()

Class_min_size = 20
for i in gr:
  
    if len(i[1]) > Class_min_size and i[0][0]!= "бензин" and i[0][1] != "заказчик":
       PS_new =  PS_new.append(i[1], ignore_index = True)
   
    

#%%"

            
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
# For test all possible data

N1 = 100
X1 =  PS[features].iloc[0:N1]
Y1 = pd.DataFrame()
for i in range(N1):
    ind = PS.iloc[i][["name", "provider"]]
    list_ind =  list(ind.values)
    if list_ind  in CLs.values():
        k = get_key(CLs,list_ind)
        k = int(k)
        
        Y1 = Y1.append(pd.Series({"class_num": k }),ignore_index=True)
    else : 
        Y1 = Y1.append(pd.Series({"class_num": int(12) }),ignore_index=True)

#%%

#X преобразовать в цифру и заполнить пропуски
def to_float(x):
  
    try:
        return float(str(x).replace(",","."))
    except:
        return np.NAN
    

for col in features: 
   
    X[col] = X[col].apply(to_float)
    X1[col] = X1[col].apply(to_float)
ll = []
for i in X1.index:
    
    if sum(X1.iloc[i].isnull()) == 1:
        ll.append(i)
X1 = X1.drop(ll,axis = 0)
Y1 = Y1.drop(ll,axis = 0)
#%% Анализ неверных значений

Crit1 =  np.logical_and(X['ro']>650, X['ro']< 900)
Crit2 = np.logical_or(Crit1, X['ro'].isnull())
Crit3 = np.logical_and(X["TLL"]  < 40, Crit2)
Crit4 = np.logical_and(X["ETBE"] < 2, Crit3)
Crit5 = np.logical_and(X["DIPE"] < 2, Crit4)
Crit6 = np.logical_and(X["TAME"] < 2, Crit5)
Crit7 = np.logical_and(X["AROMA"] < 90, Crit6)
Crit8 = np.logical_and(X["RON"] <120, Crit7)
Crit9 = np.logical_and(X["MON"] < 120, Crit8)
Crit10 = np.logical_and(X["OLF"] < 50, Crit9)
Crit11 = np.logical_and(X["BNZ"] < 5, Crit10)
X = X[Crit11]
Y = Y[Crit11]
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
del X1['MASS']
del X1['KIS']
del X1["METANOL"]

#%% Добавим размножением данные по классам, в которых меньшее число примеров.
a =  Y.groupby("class_num")
mn = max(a.apply(len))

for i in a:
    n = len(i[1] )
    print(mn,n)
    dl = int(mn/n)
    for j in np.arange(dl - 1):
        X =  X.append(X[Y["class_num"] == i[0]].iloc[0:n])
        Y =  Y.append(Y[Y["class_num"] == i[0]].iloc[0:n])
        
    X =  X.append(X[Y["class_num"] == i[0]].iloc[0:mn - n*(dl)])
    Y =  Y.append(Y[Y["class_num"] == i[0]].iloc[0:mn - n*(dl)])
    
    

#%% Предварительное разбиен ие на тестовую и обучающую выборки
(X_train, X_test, y_train, y_test) = train_test_split(X,Y,test_size = 0.3, random_state = 0, stratify=Y)
  
#%% Масштабирование

coder = StandardScaler()
coder.fit(X_train)
X_train_scaled = coder.transform(X_train)
X_test_scaled =  coder.transform(X_test)
#X1_scaled = coder.transform(X1)
#%%
params = {'kernel':'rbf',"C":512.0,"gamma":0.015625}
#params = {'C': 8192.0, 'gamma': 0.0078125}
def best_params():
    
    cv = skl.cross_validation.StratifiedShuffleSplit(y_train, n_iter = 10, test_size = 0.2, random_state = 0)
    parameters_grid = {
        'gamma' :pow(2.0,np.arange(-15,3)),   
        'C' : pow(2.0,np.arange(-5,15,2))
    }
    grid_cv = skl.grid_search.GridSearchCV(SVC(), parameters_grid, scoring = 'accuracy', cv = cv)  
    grid_cv.fit(X_train_scaled, y_train)
   
    #print(grid_cv.best_estimator_)
    #print(grid_cv.best_score_)
    #print(grid_cv.best_params_)
    #print(grid_cv.grid_scores_[:10])
    return grid_cv.best_params_
#params = best_params()

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
y_test_pred = classifier.predict(X_test_scaled)
print(classification_report(y_test, y_test_pred, target_names = s))
s1 = s.append(12)

#print(classification_report(Y1, classifier.predict(X1_scaled), target_names = s1))

Y_res = y_train.copy()
Y_res["pred"] = y_pred
#%%
cnf_mtr = confusion_matrix(y_test, y_test_pred)
plot_confusion_matrix(cnf_mtr, classes = CLs, title='Confusion matrix, without normalization')
#clf = svm.SVC(probability=True, random_state=0)
#DD = cross_val_score(clf, X, Y, scoring='neg_log_loss') 

#%%
# тест на плохих данных. Как отделить ???
#params = {'kernel':'rbf',"C":512.0,"gamma":0.015625, "probability" : True}
#classifier2 = SVC(**params)
#classifier2.fit(X_train_scaled, y_train)
#y_pred = classifier2.predict(X_train_scaled)   
#y_pred_pbob = classifier2.predict_proba(X_train_scaled) 
##print(classification_report(y_train, y_pred, target_names = s))

#print(classification_report(y_test, classifier2.predict(X_test_scaled), target_names = s))

#print(classification_report(Y1, classifier2.predict(X1_scaled), target_names = s1))
