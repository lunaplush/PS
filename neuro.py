 

#pip install git+https://github.com/pybrain/pybrain.git@0.3.3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
from sklearn import svm
from sklearn.cross_validation import train_test_split,cross_val_score,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import os
import random
 from scipy import diag,arange


from sklearn.datasets import make_classification

#%%
from pybrain.datasets import ClassificationDataSet # Структура данных pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.utilities import percentError
#%matplotlib inline
#%%

#X, y = make_classification(n_features=100, n_samples=1000)
 
N = 2
k = 100
K = 1 #clusters

np.random.seed(0)
#corresponds to N, K
means =[(0.1,0.1), (0.4,0.5)]
cov =[diag([0.1,0.2]), diag([0.1,0.05])]
ds = ClassificationDataSet(N,nb_classes = 2)
for i in range(k):
    for cluster in range(K):
        input = np.random.multivariate_normal(means[cluster],cov[cluster])
        for j in range(N):
           if input[j] > 1:
               input[j] = means[cluster][j]
           if input[j] < 0:
               input[j] = means[cluster][j]
        ds.addSample(input,[1])
        ds.addSample(np.random.uniform(low = tuple(np.zeros(N,int)),high = tuple(np.ones(N,int))),[0])
        
#for i in range(k*K):
#    ds.addSample(np.random.uniform(low = tuple(np.zeros(N,int)),high = tuple(np.ones(N,int))),[0])
        
        
#%%
mean = [0,0]
#mean = [0]
cov = [[0.01,0],[0,10]]
#cov = [0.1]
x,y =  np.random.multivariate_normal(mean,cov,5000).T
plt.plot(x,y,'b')
plt.axis('equal')
plt.xticks(np.linspace(-20,20,13))

plt.show()
    


#%%
X = ds['input']
plt.scatter(X[:k,0],X[:k,1])
plt.scatter(X[k:,0],X[k:,1], color = "red")

#%%


TRAIN_SIZE = 0.7 # Разделение данных на обучающую и контрольную части в пропорции 70/30%
#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=0)


#%%
#%%
# Определение основных констант
HIDDEN_NEURONS_NUM = 20 # Количество нейронов, содержащееся в скрытом слое сети
MAX_EPOCHS = 100 # Максимальное число итераций алгоритма оптимизации параметров сети

#%%

# Конвертация данных в структуру ClassificationDataSet
# Обучающая часть
ds_train = ClassificationDataSet(np.shape(X)[1], nb_classes=len(np.unique(y_train)))
# Первый аргумент -- количество признаков np.shape(X)[1], второй аргумент -- количество меток классов len(np.unique(y_train)))
ds_train.setField('input', X_train) # Инициализация объектов
#ds_train.setField('target', y_train[:, np.newaxis]) # Инициализация ответов; np.newaxis создает вектор-столбец
ds_train.setField('target', y_train)

# Контрольная часть
ds_test = ClassificationDataSet(np.shape(X)[1], nb_classes=len(np.unique(y_train)))
ds_test.setField('input', X_test)

#ds_test.setField('target', y_test[:, np.newaxis])
ds_test.setField('target', y_test)

#%%
np.random.seed(0)
net = buildNetwork(ds_train.indim, HIDDEN_NEURONS_NUM, ds_train.outdim)
# ds.indim -- количество нейронов входного слоя, равне количеству признаков
# ds.outdim -- количество нейронов выходного слоя, равное количеству меток классов
# SoftmaxLayer -- функция активации, пригодная для решения задачи многоклассовой классификации

init_params = np.random.random(( len(net.params))) # Инициализируем веса сети для получения воспроизводимого результата
net._setParameters(init_params)
#%%
np.random.seed(0)
# Модуль настройки параметров pybrain использует модуль random; зафиксируем seed для получения воспроизводимого результата
trainer = BackpropTrainer(net, dataset=ds_train) # Инициализируем модуль оптимизации
err_train, err_val = trainer.trainUntilConvergence(maxEpochs=MAX_EPOCHS)
line_train = plt.plot(err_train, 'b', err_val, 'r') # Построение графика
xlab = plt.xlabel('Iterations')
ylab = plt.ylabel('Error')

#%%
res_train = net.activateOnDataset(ds_train) # Подсчет результата на обучающей выборке
print('Error on train: ', percentError(res_train, ds_train['target'])) # Подсчет ошибки
res_test = net.activateOnDataset(ds_test) # Подсчет результата на тестовой выборке
print('Error on test: ', percentError(res_test, ds_test['target'])) # Подсчет ошибки


#%%
#class PsData:
#    name = "PsData class"
#    def __init__(self, read_func = ps_data.open_ps_2007):
#       
#        self.PS_X,self.PS_Y, self.PS_Classes = read_func()   
#                
#        self.features = ["ro","RON", "MON", "OLF","ALK","AROMA","BNZ","KIS", \
#            "TLL","MASS","ETANOL","MTBE","ETBE","TAME","DIPE","TBS"]
#       # without "METANOL",
#
#    def prepare_data(self, y_list = [] ):
#      
#obj =  PsData(ps_data.open_ps_2007)
#obj.prepare_data([3,4,5])
#obj.read(ps_data.open_ps_2007)        