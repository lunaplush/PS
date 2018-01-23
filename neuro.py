 

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
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.utilities import percentError
from pybrain.structure import FullConnection
from pybrain.structure import FeedForwardNetwork
#%matplotlib inline
#%%

#X, y = make_classification(n_features=100, n_samples=1000)
 
#N = 16
N=2
k = 100
K = 2 #clusters

dl = 35
np.random.seed(0)
#corresponds to N, K
#means =[(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.3,0.1,0.7,0.1,0.1,0.4,0.2,0.1,0.7), (0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5)]
#cov =[diag([0.1/35,0.05/35,0.1/35,0.05/35,0.1/35,0.05/35,0.1/35,0.05/35,0.1/35,0.05/35,0.1/35,0.05/35,0.1/35,0.05/35,0.1/35,0.05/35]), diag([0.2/35,0.05/35,0.2/35,0.05/35,0.2/35,0.05/35,0.2/35,0.05/35,0.2/35,0.05/35,0.2/35,0.05/35,0.2/35,0.05/35,0.2/35,0.05/35])]

ch1 = 0
means =[(0.1,0.1), (0.4,0.5)]
cov =[diag([0.1/dl,0.05/dl]), diag([0.2/dl,0.05/dl])]

ds = ClassificationDataSet(N,nb_classes = 2)
for i in range(k):
    for cluster in range(K):
       
        input = np.random.multivariate_normal(means[cluster],cov[cluster])
        
        for j in range(N):
           if input[j] > 1:
               input[j] = means[cluster][j]
               ch1+= 1
           if input[j] < 0:
               input[j] = means[cluster][j]
               ch1+=1
        ds.addSample(input,[1])
        ds.addSample(np.random.uniform(low = tuple(np.zeros(N,int)),high = tuple(np.ones(N,int))),[0])
        
#for i in range(k*K):
#    ds.addSample(np.random.uniform(low = tuple(np.zeros(N,int)),high = tuple(np.ones(N,int))),[0])
        
        

#%%
#mean = [0,0]
##mean = [0]
#cov = [[0.01,0],[0,10]]
##cov = [0.1]
#x,y =  np.random.multivariate_normal(mean,cov,5000).T
#plt.plot(x,y,'b')
#plt.axis('equal')
#plt.xticks(np.linspace(-20,20,13))
#
#plt.show()
    

a = np.array([ds['input'][i][0] for i in arange(K*k*2) if ds['target'][i] == 1]).reshape(K*k,1)
b = np.array([ds['input'][i][1] for i in arange(K*k*2) if ds['target'][i] == 1]).reshape(K*k,1)
X1 = np.hstack((a,b))
a = np.array([ds['input'][i][0] for i in arange(K*k*2) if ds['target'][i] == 0]).reshape(K*k,1)
b = np.array([ds['input'][i][1] for i in arange(K*k*2) if ds['target'][i] == 0]).reshape(K*k,1)
X0 = np.hstack((a,b))
plt.scatter(X1[:,0],X1[:,1])
plt.scatter(X0[:,0],X0[:,1], color = "red")

#%%


TRAIN_SIZE = 0.7 # Разделение данных на обучающую и контрольную части в пропорции 70/30%
#from sklearn.model_selection import train_test_split

ds_train, ds_test = ds.splitWithProportion(TRAIN_SIZE)


#%%
# Определение основных констант
HIDDEN_NEURONS_NUM = 20 # Количество нейронов, содержащееся в скрытом слое сети
HIDDEN_NEURONS_NUM2 = 10
MAX_EPOCHS = 250




 # Максимальное число итераций алгоритма оптимизации параметров сети




#%%
#np.random.seed(0)
#net = FeedForwardNetwork()
#inLayer = LinearLayer(N)
#hiddenLayer = SigmoidLayer(HIDDEN_NEURONS_NUM)
#hiddenLayer2 = SigmoidLayer(HIDDEN_NEURONS_NUM2)
#outLayer = LinearLayer(1)
#
#net.addInputModule(inLayer)
#net.addModule(hiddenLayer)
#net.addModule(hiddenLayer2)
#net.addOutputModule(outLayer)
#
#net.addConnection(FullConnection(inLayer, hiddenLayer))
#net.addConnection(FullConnection(hiddenLayer,hiddenLayer2))
#net.addConnection(FullConnection(hiddenLayer2, outLayer))
#net.sortModules()
#%%
#net = buildNetwork(ds_train.indim, HIDDEN_NEURONS_NUM, ds_train.outdim, bias=True,outclass=SoftmaxLayer)
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
#ROC - кривые - порог
grance = 0.5
res_train = net.activateOnDataset(ds_train) # Подсчет результата на обучающей выборке
res_train_bin = []
for i in res_train:
    if  i > grance:
        res_train_bin.append(1)
    else:
        res_train_bin.append(0)
        
print('Error on train: ', percentError(res_train_bin, ds_train['target'])) # Подсчет ошибки
res_test = net.activateOnDataset(ds_test) # Подсчет результата на тестовой выборке
res_test_bin = []
for i in res_test:
    if  i > grance:
        res_test_bin.append(1)
    else:
        res_test_bin.append(0)
print('Error on test: ', percentError(res_test_bin, ds_test['target'])) # Подсчет ошибки

#%%




X1 = ds_train['input'][(ds_train['target'] == 1).flatten()]
X2 = ds_train['input'][(ds_train['target'] == 0).flatten()]
plt.scatter(X1[:,0],X1[:,1])
plt.scatter(X0[:,0],X0[:,1], color = "red")

xx,yy = np.meshgrid(np.arange(0,1,0.01), np.arange(0,1,0.01))
data = np.c_[xx.ravel(),yy.ravel()] 
Z = [net.activate(i) for i in data]
res_Z_bin = []
for i in Z:
    if  i > grance:
        res_Z_bin.append(1)
    else:
        res_Z_bin.append(0)
res_Z_bin = np.array(res_Z_bin)
XZ0 = data[(res_Z_bin == 0)]
plt.scatter(XZ0[:,0],XZ0[:,1], color = "green")
#X3 =ds_train['input'][(ds_train['target'].T!=np.array(res_train_bin)).flatten()]    

#plt.scatter(X3[:,0],X3[:,1], color = "green", s = 7,alpha = 0.7)
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