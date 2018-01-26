 
#pip install keras
#pip install tensorflow
#pip install git+https://github.com/pybrain/pybrain.git@0.3.3


#deep.uran.ru/wiki/index.php?title=Эксперименты_с_многослойным_перцептроном_в_Keras
import numpy as np




import matplotlib.pyplot as plt
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

from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from pybrain.utilities           import percentError

#model = Sequental()
#model.add(Dense(units =10,activation ="relu"))
#model.add(Dence(units=1, activation ="sigmoid"))
#model.compile() #prepaire model
#
#model.fit(x_train, y_train,epoch = 5, batch_size = 32)
#
#
#model.train_on_batch(x_batch, y_batch)
#loss_and_metricks = model.evaluate(x_test, y_test, batch_size = 128)
#
#classes = model.predict(x_test, batch_size = 128)

#%%

#X, y = make_classification(n_features=100, n_samples=1000)
EXP_NUM = 17
#N = 16
N=2
k = 50
K = 2 #clusters

dl = 35
np.random.seed(0)
#corresponds to N, K
#means =[(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.3,0.1,0.7,0.1,0.1,0.4,0.2,0.1,0.7), (0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5)]
#cov =[diag([0.1/35,0.05/35,0.1/35,0.05/35,0.1/35,0.05/35,0.1/35,0.05/35,0.1/35,0.05/35,0.1/35,0.05/35,0.1/35,0.05/35,0.1/35,0.05/35]), diag([0.2/35,0.05/35,0.2/35,0.05/35,0.2/35,0.05/35,0.2/35,0.05/35,0.2/35,0.05/35,0.2/35,0.05/35,0.2/35,0.05/35,0.2/35,0.05/35])]

ch1 = 0
means =[(0.25,0.25), (0.75,0.75), (0.1,0.1)]
cov =[diag([0.1/dl,0.05/dl]), diag([0.2/dl,0.05/dl]),diag([0.2/dl,0.05/dl])]

X = []
y = []
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
        X.append(input)
        y.append(1)
        X.append(np.random.uniform(low = tuple(np.zeros(N,int)),high = tuple(np.ones(N,int))))
        y.append(0)
        
        
X= np.array(X)
y= np.array(y)



#%%

nb_classes = 2

#%%

TRAIN_SIZE = 0.7 # Разделение данных на обучающую и контрольную части в пропорции 70/30%
#from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, train_size =TRAIN_SIZE, random_state = 10)

Y_train = np_utils.to_categorical(y_train,nb_classes)
Y_test = np_utils.to_categorical(y_test,nb_classes)
#Где гарантия, что бинарные представления классов будут одинаковы для разных запуском функции 
#to_categorical для одной задачи.
#%%
# Определение основных констант
HIDDEN_NEURONS_NUM = 15 # Количество нейронов, содержащееся в скрытом слое сети
HIDDEN_NEURONS_NUM_2 = 15
MAX_EPOCHS =1500




 # Максимальное число итераций алгоритма оптимизации параметров сети




#%%
np.random.seed(0)
model = Sequential()
model.add(Dense(units = HIDDEN_NEURONS_NUM,input_dim = N,activation = "relu"))
model.add(Dense(units = HIDDEN_NEURONS_NUM_2,activation ="relu"))
model.add(Dense(units = nb_classes,activation = "softmax"))
model.compile(optimizer = 'adam',loss='categorical_crossentropy', metrics=['accuracy'])

#%%
model.fit(X_train,Y_train,batch_size = 1,nb_epoch = MAX_EPOCHS,verbose = 2, validation_data =(X_test,Y_test))

#%%
score  = model.evaluate(X_test,Y_test,verbose =0)
print('Test score:',score[0])
print('Test accuracy:',score[1])


 

#%%
#ROC - кривые - порог
res_train= model.predict_on_batch(X_train)
#grance = 0.5
res_train_bin = res_train.argmax(axis = 1)
        
print('Error on train: ', percentError(res_train_bin, y_train)) # Подсчет ошибки

res_test = model.predict_on_batch(X_test) # Подсчет результата на тестовой выборке
res_test_bin = res_test.argmax(axis = 1)
print('Error on test: ', percentError(res_test_bin,y_test)) # Подсчет ошибки

#%%




X1 = X_train[(y_train == 1).flatten()]
X0 = X_train[(y_train == 0).flatten()]
plt.scatter(X1[:,0],X1[:,1])
plt.scatter(X0[:,0],X0[:,1], color = "red")

xx,yy = np.meshgrid(np.arange(0,1,0.01), np.arange(0,1,0.01))
data = np.c_[xx.ravel(),yy.ravel()] 

Z = model.predict_on_batch(data)

res_Z_bin = Z.argmax(axis = 1)
XZ0 = data[(res_Z_bin == 0)]
plt.scatter(XZ0[:,0],XZ0[:,1], color = "red", alpha = 0.3)
plt.text(0.9,0.9,s = "{:3f}".format(score[1]))
plt.savefig("exp{}.jpg".format(EXP_NUM))
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