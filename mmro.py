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
import ps_data
os.chdir("../PS_data")
#%%

X1,Y1,CLs1 = ps_data.open_ps_2007()  
#X2,Y2,CLs2 = ps_data.open_ps_2009()  
#%%


features = ["ro","RON", "MON", "OLF","ALK","AROMA","BNZ","KIS", \
            "TLL","MASS","METANOL","ETANOL","MTBE","ETBE","TAME","DIPE","TBS"]
features2 = ["ro","RON", "MON", "OLF","ALK","AROMA","BNZ","KIS", \
            "TLL","MASS","ETANOL","MTBE","ETBE","TAME","DIPE","TBS"]


#%%
X7 = X1.copy()
X7["class_num"] = Y1["class_num"]

#X9 = X2.copy()
#X9["class_num"] = Y2["class_num"]
#%%
#sns.set()
#sns.pairplot(X7, hue="class_num",size=4, x_vars=["RON","MON"],y_vars=features,markers="." ).savefig("pairplot_2007_classes_RON_other_s4.png")

#sns.pairplot(X9,hue="class_num",size=4,x_vars=["RON","MON"],y_vars=features,markers=".").savefig("pairplot_2009_classes_RON_other_s4.png")
#%%
#https://habrahabr.ru/post/304214/
X7_92 = X7[(X7["class_num"] == 3) | (X7["class_num"] == 4) | (X7["class_num"] == 5)]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


#X = X7_92[X7.columns - ["class_num"]] - X7_92[X7.columns - ["class_num"]].mean()
X7_92_x = X7_92[list(set(X7.columns) - {"class_num","METANOL"})]
#X7_92_x = X7_92[features2]
minzn = X7_92_x.min(axis =0)
maxzn = X7_92_x.max(axis =0)
dif_min_max = maxzn-minzn


X = scaler.fit_transform( X7_92[list(set(X7.columns) - {"class_num","METANOL"})])
y = np.array(X7_92["class_num"]-3)

#matrix_covariance = np.cov(X.T)
#vectors = np.linalg.eig(matrix_covariance)
#%%
TRAIN_SIZE = 0.7 # Разделение данных на обучающую и контрольную части в пропорции 70/30%
#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=0)
#%%
from pybrain.datasets import ClassificationDataSet # Структура данных pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.utilities import percentError
#%%
# Определение основных констант
HIDDEN_NEURONS_NUM = 10 # Количество нейронов, содержащееся в скрытом слое сети
MAX_EPOCHS = 100 # Максимальное число итераций алгоритма оптимизации параметров сети

#%%
# Конвертация данных в структуру ClassificationDataSet
# Обучающая часть
ds_train = ClassificationDataSet(np.shape(X)[1], nb_classes=len(np.unique(y_train)))
# Первый аргумент -- количество признаков np.shape(X)[1], второй аргумент -- количество меток классов len(np.unique(y_train)))
ds_train.setField('input', X_train) # Инициализация объектов
ds_train.setField('target', y_train[:, np.newaxis]) # Инициализация ответов; np.newaxis создает вектор-столбец
ds_train._convertToOneOfMany( ) # Бинаризация вектора ответов
# Контрольная часть
ds_test = ClassificationDataSet(np.shape(X)[1], nb_classes=len(np.unique(y_train)))
ds_test.setField('input', X_test)
ds_test.setField('target', y_test[:, np.newaxis])
ds_test._convertToOneOfMany( )
#%%
np.random.seed(0) # Зафиксируем seed для получения воспроизводимого результата

# Построение сети прямого распространения (Feedforward network)
net = buildNetwork(ds_train.indim, HIDDEN_NEURONS_NUM, ds_train.outdim, outclass=SoftmaxLayer)
# ds.indim -- количество нейронов входного слоя, равне количеству признаков
# ds.outdim -- количество нейронов выходного слоя, равное количеству меток классов
# SoftmaxLayer -- функция активации, пригодная для решения задачи многоклассовой классификации

init_params = np.random.random(( len(net.params))) # Инициализируем веса сети для получения воспроизводимого результата
net._setParameters(init_params)
#%%
random.seed(0)
# Модуль настройки параметров pybrain использует модуль random; зафиксируем seed для получения воспроизводимого результата
trainer = BackpropTrainer(net, dataset=ds_train) # Инициализируем модуль оптимизации
err_train, err_val = trainer.trainUntilConvergence(maxEpochs=MAX_EPOCHS)
line_train = plt.plot(err_train, 'b', err_val, 'r') # Построение графика
xlab = plt.xlabel('Iterations')
ylab = plt.ylabel('Error')
#%%
res_train = net.activateOnDataset(ds_train).argmax(axis=1) # Подсчет результата на обучающей выборке
print('Error on train: ', percentError(res_train, ds_train['target'].argmax(axis=1)), '%') # Подсчет ошибки
res_test = net.activateOnDataset(ds_test).argmax(axis=1) # Подсчет результата на тестовой выборке
print('Error on test: ', percentError(res_test, ds_test['target'].argmax(axis=1)), '%') # Подсчет ошибки

                                      
#%%
#Эксперимент для одного класса
#[ 0 = 3, 1= 5, 2=5]
class_num = []
for i in X7_92.groupby("class_num"):
    class_num.append(len(i[1]))

class_num_current = 0    
Xp = np.random.rand(class_num[0],X.shape[1])

X_class = np.vstack((X[y==class_num_current],Xp))
y_class = np.hstack((y[y==class_num_current], -1*np.ones(class_num[0], dtype = "int")))
#y_class = [int(i) for i in y_class]
sss = StratifiedShuffleSplit(y = y_class, n_iter = 1, test_size = 0.33, random_state = 10)
for tr,ts in sss: 
    train_index = tr
    test_index =  ts  
    X_class_train = X_class[train_index]
    X_class_test = X_class[test_index]
    y_class_train = y_class[train_index]
    y_class_test = y_class[test_index]
 
ds_train = ClassificationDataSet(np.shape(X_class)[1], nb_classes= 2)
ds_train.setField("input", X_class_train)
ds_train.setField("target", y_class_train[:, np.newaxis])


ds_test = ClassificationDataSet(np.shape(X_class)[1], nb_classes= 2)
ds_test.setField("input", X_class_test)
ds_test.setField("target", y_class_test[:, np.newaxis])

#%%
HIDDEN_NEURONS_NUM = 100
np.random.seed(10)
net = buildNetwork(ds_train.indim,HIDDEN_NEURONS_NUM, ds_train.outdim )
init_params = np.random.random(( len(net.params))) # Инициализируем веса сети для получения воспроизводимого результата
net._setParameters(init_params)

np.random.seed(10)

trainer = BackpropTrainer(net, dataset=ds_train) # Инициализируем модуль оптимизации
err_train, err_val = trainer.trainUntilConvergence(maxEpochs=MAX_EPOCHS)
line_train = plt.plot(err_train, 'b', err_val, 'r') # Построение графика
xlab = plt.xlabel('Iterations')
ylab = plt.ylabel('Error')
#%%
#%%
res_train = net.activateOnDataset(ds_train).argmax(axis=1) # Подсчет результата на обучающей выборке
print('Error on train: ', percentError(res_train, ds_train['target'].argmax(axis=1)), '%') # Подсчет ошибки
res_test = net.activateOnDataset(ds_test).argmax(axis=1) # Подсчет результата на тестовой выборке
print('Error on test: ', percentError(res_test, ds_test['target'].argmax(axis=1)), '%') # Подсчет ошибки

     

#%%
#random.seed(0) # Зафиксируем seed для получния воспроизводимого результата
#
#
#def plot_classification_error(hidden_neurons_num, res_train_vec, res_test_vec):
## hidden_neurons_num -- массив размера h, содержащий количество нейронов, по которому предполагается провести перебор,
##   hidden_neurons_num = [50, 100, 200, 500, 700, 1000];
## res_train_vec -- массив размера h, содержащий значения доли неправильных ответов классификации на обучении;
## res_train_vec -- массив размера h, содержащий значения доли неправильных ответов классификации на контроле
#    plt.figure()
#    plt.plot(hidden_neurons_num, res_train_vec)
#    plt.plot(hidden_neurons_num, res_test_vec, '-r')
#
#def write_answer_nn(optimal_neurons_num):
#    with open("nnets_answer1.txt", "w") as fout:
#        fout.write(str(optimal_neurons_num))
#
#hidden_neurons_num = [50, 100, 200, 500, 700, 1000]
#res_train_vec = list()
#res_test_vec = list()
##np.random.seed(0)
##random.seed(0)
#for nnum in hidden_neurons_num:
#    
#    # Put your code here
#    # Не забудьте про инициализацию весов командой np.random.random((len(net.params)))
#   
#    net = buildNetwork(ds_train.indim, nnum, ds_train.outdim, outclass=SoftmaxLayer)
#    init_params = np.random.random((len(net.params))) # Инициализируем веса сети для получения воспроизводимого результата
#    net._setParameters(init_params)
#    
#  
#    trainer = BackpropTrainer(net, dataset=ds_train) # Инициализируем модуль оптимизации
#    err_train, err_val = trainer.trainUntilConvergence(maxEpochs=MAX_EPOCHS)
#    
#    res_train = net.activateOnDataset(ds_train).argmax(axis=1) 
#    res_train_vec.append(percentError(res_train, ds_train['target'].argmax(axis=1)))
#    res_test = net.activateOnDataset(ds_test).argmax(axis=1)
#    res_test_vec.append(percentError(res_test, ds_test['target'].argmax(axis=1)))
#    
## Постройте график зависимости ошибок на обучении и контроле в зависимости от количества нейронов
#plot_classification_error(hidden_neurons_num, res_train_vec, res_test_vec)          
##  Запишите в файл количество нейронов, при котором достигается минимум ошибки на контроле
#write_answer_nn(hidden_neurons_num[res_test_vec.index(min(res_test_vec))]) 