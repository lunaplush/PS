import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
from sklearn import svm
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import os
import ps_data
os.chdir("../PS_data")
#%%

X1,Y1,CLs1 = ps_data.open_ps_2007()  
X2,Y2,CLs2 = ps_data.open_ps_2009()  
#%%


features = ["ro","RON", "MON", "OLF","ALK","AROMA","BNZ","KIS", \
            "TLL","MASS","METANOL","ETANOL","MTBE","ETBE","TAME","DIPE","TBS"]


#%%
X7 = X1.copy()
X7["class_num"] = Y1["class_num"]

X9 = X2.copy()
X9["class_num"] = Y2["class_num"]
#%%
#sns.set()
#sns.pairplot(X7, hue="class_num",size=4, x_vars=["RON","MON"],y_vars=features,markers="." ).savefig("pairplot_2007_classes_RON_other_s4.png")

#sns.pairplot(X9,hue="class_num",size=4,x_vars=["RON","MON"],y_vars=features,markers=".").savefig("pairplot_2009_classes_RON_other_s4.png")
#%%
#https://habrahabr.ru/post/304214/
X7_92 = X7[(X7["class_num"] == 3) | (X7["class_num"] == 4) | (X7["class_num"] == 5)]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


#X = X7_92[X7.columns - ["class_num"]] - X7_92[X7.columns - ["class_num"]].mean()
X = scaler.fit_transform(X7_92[X7.columns - ["class_num"]])
Y = X7_92["class_num"]-3

matrix_covariance = np.cov(X.T)
vectors = np.linalg.eig(matrix_covariance)