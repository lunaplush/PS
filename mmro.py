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


features = ["ro","RON", "MON", "OLF","ALK","AROMA","BNZ","KIS", \
            "TLL","MASS","METANOL","ETANOL","MTBE","ETBE","TAME","DIPE","TBS"]


#%%
X7 = X1.copy()
X7["class_num"] = Y1["class_num"]

X9 = X2.copy()
X9["class_num"] = Y2["class_num"]
#%%
sns.set()
sns.pairplot(X7, hue="class_num",size=4, x_vars=["RON","MON"],y_vars=features,markers="." ).savefig("pairplot_2007_classes_RON_other_s4.png")

sns.pairplot(X9,hue="class_num",size=4,x_vars=["RON","MON"],y_vars=features,markers=".").savefig("pairplot_2009_classes_RON_other_s4.png")