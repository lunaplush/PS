# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:55:20 2017

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

import ps_data

#%%

X1,Y1,CLs1 = ps_data.open_ps_2007()  
X2,Y2,CLs2 = ps_data.open_ps_2009()    
#%%
sns.set()
#add hue
#sns.pairplot(X1).savefig("pairplot_2007.png")
#sns.pairplot(X2).savefig("pairplot_2009.png")
#%%

X = X1
Y = Y1
CLs = CLs1

#%% Предварительное разбиен ие на тестовую и обучающую выборки
(X_train, X_test, y_train, y_test) = train_test_split(X,Y,test_size = 0.3, random_state = 0, stratify=Y)
  
#%% Масштабирование

coder = StandardScaler()
coder.fit(X_train)
X_train_scaled = coder.transform(X_train)
X_test_scaled =  coder.transform(X_test)
#%% 
# Дополним классы, в которых меньшее число примеров.

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
    
    


#%%
params = {'kernel':'rbf',"C":512.0,"gamma":0.015625}
#params = {'C': 8192.0, 'gamma': 0.0078125}
#params = {'gamma': 0.000244140625, 'C': 8192.0}
def best_params():
    
    cv = skl.cross_validation.StratifiedShuffleSplit(y_train, n_iter = 10, test_size = 0.2, random_state = 0)
    parameters_grid = {
        'gamma' :pow(2.0,np.arange(-15,3)),   
        'C' : pow(2.0,np.arange(-5,15,2))
    }
    grid_cv = skl.model_selection.GridSearchCV(SVC(), parameters_grid, scoring = 'accuracy', cv = cv)  
    grid_cv.fit(X_train_scaled, y_train)
   
    #print(grid_cv.best_estimator_)
    #print(grid_cv.best_score_)
    #print(grid_cv.best_params_)
    #print(grid_cv.grid_scores_[:10])
    return grid_cv.best_params_
params_func = best_params()
print(params_func)
#%%

#classifier = SVC(**params_func)#params)
classifier = SVC(**params)

classifier.fit(X_train_scaled, y_train)
#utilities.plot_classifier(classifier,X_train_scaled,y_train, "TR")
#plt.show()
target_names = ["CLass - " + str(int(i)) for i in CLs.keys()]
#print("\n" +"#"*30)
#print("Perfomance on train set")
s = []
for v in CLs.values():
    s.append(str(v))
y_pred = classifier.predict(X_train_scaled)    
train_data_report = classification_report(y_train, y_pred, target_names = s)
test_data_report = classification_report(y_test, classifier.predict(X_test_scaled), target_names = s)


file = open('svm_base_report.txt','w')
file.write('#'*30)
file.write('\t\t\t2007\n\n')
file.write('\t\t\t TRAIN \n')
file.write(train_data_report)

file.write('\n\t\t\t TEST \n')
file.write(test_data_report) 
Y_res = y_train.copy()
Y_res["pred"] = y_pred
#%%


