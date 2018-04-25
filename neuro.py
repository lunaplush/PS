# commit exp 24 1 cluster 1 parent dccbb87 commit 096ceca94d1fdede72d0e5a7ddfb0d4395bf0518
#                                                 096ceca94d1fdede72d0e5a7ddfb0d4395bf0518 


#pip install keras
#pip install tensorflow
#pip install git+https://github.com/pybrain/pybrain.git@0.3.3


#deep.uran.ru/wiki/index.php?title=Эксперименты_с_многослойным_перцептроном_в_Keras
import numpy as np
import time



import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cross_validation import train_test_split,cross_val_score,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import os
import random
import pandas as pd
from scipy import diag,arange


from sklearn.datasets import make_classification
#%%
import keras as ks
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.core import Dense
from keras.utils import np_utils
from pybrain.utilities           import percentError
from keras import backend
from keras  import metrics
import pandas as pd 
#os.chdir("c:\\Luna\\Work\\python\\PS")
#%%
#def mean_pred(y_true,y_pred):
#    with backend.tf.Session() as sess:
#        a = sess.run(y_true)
#        b = sess.run(y_pred)
#    return backend.mean(y_pred)

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
#classes = model.predict(x_test, batch_size = 1
#%%

#X, y = make_classification(n_features=100, n_samples=1000)
EXP_NUM = 37  
N = 2
#N=2

k = 500
K = 2 #clusters


#dl = 0.3
dl = 35 # 

np.random.seed(10)
#corresponds to N, K
#means =[(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.3,0.1,0.7,0.1,0.1,0.4,0.2,0.1,0.7), (0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5,0.4,0.5)]
#cov =[diag([0.1/dl,0.05/dl,0.1/dl,0.05/dl,0.1/dl,0.05/dl,0.1/dl,0.05/dl,0.1/dl,0.05/dl,0.1/dl,0.05/dl,0.1/dl,0.05/dl,0.1/dl,0.05/dl]), diag([0.2/dl,0.05/dl,0.2/dl,0.05/dl,0.2/dl,0.05/dl,0.2/dl,0.05/dl,0.2/dl,0.05/dl,0.2/dl,0.05/dl,0.2/dl,0.05/dl,0.2/dl,0.05/dl])]

means =[(0.25,0.25),            (0.75,0.75),           (0.1,0.1),            (0.45,0.7),              (0.8,0.24)]
cov =  [diag([0.1/dl,0.05/dl]), diag([0.2/dl,0.05/dl]),diag([0.2/dl,0.05/dl]), diag([0.02/dl,0.08/dl]),diag([0.02/dl,0.08/dl]) ]
  
MAX_EPOCHS = 50





#%%
class DataParams:
    #N = 2
    #k = 100
    #K = 1
    #means =[(0.25,0.25)]
    #cov = [diag([0.1/35, 0.05/35])]
    exp_num = EXP_NUM
    
    def __init__ (self, N = 2,k = 100,K = 1, means =[(0.25,0.25)], cov = [diag([0.1/35, 0.05/35])]):
        self.N = N 
        self.k = k 
        self.K = K
        self.means = means
        self.cov = cov
    def set_params(self,N,k,K,means,cov):
        self.N = N 
        self.k = k 
        self.K = K
        self.means = means
        self.cov = cov
        
    def create_model_data(self):
        np.random.seed(367)
        X = []
        y = []
        for i in range(self.k):
            for cluster in range(self.K):
               
                input = np.random.multivariate_normal(self.means[cluster],self.cov[cluster])
                
                for j in range(self.N):
                   if input[j] > 1:
                       input[j] = self.means[cluster][j]
                       
                   if input[j] < 0:
                       input[j] = self.means[cluster][j]
                       
                X.append(input)
                y.append(1)
                X.append(np.random.uniform(low = tuple(np.zeros(self.N,int)),high = tuple(np.ones(self.N,int))))
                y.append(0)       
        
        return (np.array(X),np.array(y))
        
    def create_model_data_two(self):
        np.random.seed(367)
        X = []
        y = []
        for i in range(self.k):
            for cluster in range(self.K):
               
                input = np.random.multivariate_normal(self.means[cluster],self.cov[cluster])
                
                for j in range(self.N):
                   if input[j] > 1:
                       input[j] = self.means[cluster][j]
                       
                   if input[j] < 0:
                       input[j] = self.means[cluster][j]
                       
                X.append(input)
                y.append(cluster)
                #X.append(np.random.uniform(low = tuple(np.zeros(self.N,int)),high = tuple(np.ones(self.N,int))))
                #y.append(0)       
        
        return (np.array(X),np.array(y))
        

#%%
dataParams = DataParams(N=N,k=k,K=K,means = means, cov =cov)
(X,y) = dataParams.create_model_data()
(X2,y2) = dataParams.create_model_data_two()



#%%
def visualisation(path,X,y, score_train, score_test, model):
 
    #fig = plt.figure(1, figsize = (2,2))
    X1 = X[(y == 1).flatten()]
    X0 = X[(y == 0).flatten()]
    
    
    xx,yy = np.meshgrid(np.arange(0,1,0.01), np.arange(0,1,0.01))
    data = np.c_[xx.ravel(),yy.ravel()] 
    
    Z = model.predict_on_batch(data)
    
    res_Z_bin = Z.argmax(axis = 1)
    
    level = 0.7
    for i in np.arange(len(Z)): 
        #print(Z[i])
        if max(Z[i]) < level:
            res_Z_bin[i] =  -1
    
    XZ0 = data[(res_Z_bin == 0)]
    XZ1 = data[(res_Z_bin == 1 )]
    plt.subplot(111)
    plt.scatter(XZ0[:,0],XZ0[:,1], color = '#881000', alpha = 0.3)
    plt.scatter(XZ1[:,0],XZ1[:,1], color = "#000083", alpha = 0.3)
    
    
    plt.scatter(X1[:,0],X1[:,1], color = "blue")
    plt.scatter(X0[:,0],X0[:,1], color = "red")
    
    plt.text(1,1.1,s = "Train {:3f}".format(score_train))
    plt.text(1,1,s = "Test {:3f}".format(score_test))
    plt.savefig("models/{}fig.jpg".format(path))


#%%
nb_classes = 2
TRAIN_SIZE = 0.7 # Разделение данных на обучающую и контрольную части в пропорции 70/30%
#from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, train_size =TRAIN_SIZE, random_state = 10)

Y_train = np_utils.to_categorical(y_train,nb_classes)
Y_test = np_utils.to_categorical(y_test,nb_classes)


X_train2,X_test2,y_train2,y_test2 = train_test_split(X2,y2, train_size =TRAIN_SIZE, random_state = 10)

Y_train2 = np_utils.to_categorical(y_train2,nb_classes)
Y_test2 = np_utils.to_categorical(y_test2,nb_classes)


#Где гарантия, что бинарные представления классов будут одинаковы для разных запуском функции 
#to_categorical для одной задачи.
#%%
class NeuroModel:
    nb_classes = 2
    def __init__(self, activation_layer1= 'relu', activation_layer2='sigmoid',  \
                       optimizer =  "adam", loss_func = 'categorical_crossentropy', code = "ASR_"):
        self.activation_layer1 = activation_layer1 
        self.activation_layer2 = activation_layer2
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.code = code
    def compile_model(self, N , hidneuro1 = 10, hidneuro2 = 7, max_epochs = 50, batch_size = 1):
        np.random.seed(473)
        self.batch_size = batch_size
        self.max_epochs = max_epochs  
        self.model = Sequential()
        self.model.add(Dense(units = hidneuro1,input_dim = N,activation = self.activation_layer1))
        self.model.add(Dense(units = hidneuro2,activation = self.activation_layer2))
        self.model.add(Dense(units = self.nb_classes,activation = "softmax"))
        self.model.compile(optimizer = self.optimizer,loss= self.loss_func, metrics=['accuracy'])                  
    def fit_model(self, f_X_train,f_Y_train,f_y_train,f_X_test,f_Y_test):
        h1 = self.model.get_config()[0]['config']['units']
        h2 = self.model.get_config()[1]['config']['units']
        model_name = "{}{}_{}_N{}_cl{}".format(self.code,h1,h2,N,K )
        #не знаю, совпадает ли значение 
        checkpointer = ModelCheckpoint(filepath = "models/{}model.h5".format(model_name),monitor = "val_acc",verbose = 0,save_best_only = 1, save_weights_only = 1)
        #earlystopper = EarlyStopping(monitor ="loss", verbose = 0 , mode = "auto")
        a = time.time()    
        follow_flag = True
        counter = 0
        while  follow_flag:
            counter += 1
            fit_info = self.model.fit(f_X_train,f_Y_train,batch_size = self.batch_size,  \
                                      epochs = self.max_epochs,verbose = 2,  \
                                      validation_data =(f_X_test,f_Y_test), callbacks = [checkpointer])
            
            acc_fit = fit_info.history['acc']
            crit = np.sum(np.array(acc_fit[1:len(acc_fit)]) - np.array(acc_fit[0:len(acc_fit)-1]))
            print("Crit : {} ".format(crit))
#            if crit < 0.2 :
#                follow_flag = False
#                print("Srabotal crit counter ={}, crit = {}".format(counter, crit))
            if  counter *self.max_epochs >= MAX_EPOCHS:
                follow_flag = False
        resTime = time.time() - a #in seconds
        test_accuracy = max(fit_info.history['val_acc'])
        num = np.argmax(fit_info.history['val_acc'])
        train_accuracy = fit_info.history['acc'][num]
        if N == 2 :
            visualisation(model_name,f_X_train, f_y_train,train_accuracy, test_accuracy, self.model)
        return [model_name, resTime, train_accuracy, test_accuracy]
    

 # Максимальное число итераций алгоритма оптимизации параметров сети


#%%
df = pd.DataFrame(columns = ["model", "time","acc_train", "acc_test"])
df2 = pd.DataFrame(columns = ["model", "time","acc_train", "acc_test"])

neuros_num = [ [i,j] for i in np.arange(4,21,2) for j in np.arange(4,20,2)]

#for i in np.arange(len(neuros_num)): 
#for i in np.arange(29,30):
for pp in [1]:
    MAX_EPOCHS = 1150
    i =  29
    try:
        del model
    except:
        pass
    model = NeuroModel()
    model.compile_model(dataParams.N, hidneuro1 = neuros_num[i][0], hidneuro2 = neuros_num[i][1])
    [model_name, t,acc_train, acc_test] = model.fit_model(X_train,Y_train,y_train,X_test,Y_test)
    s = pd.Series({"model":model_name, "time": t,"acc_train":acc_train, "acc_test":acc_test})
    pd.DataFrame(s).to_csv("models/{}.csv".format(model_name))
    df = df.append(s, ignore_index= True)
    print("{} done".format(model_name))
    
    MAX_EPOCHS = 50

    try:
        del model2
    except:
        pass
   
    model2 = NeuroModel(code = "ASR_two_")
    model2.compile_model(dataParams.N, hidneuro1 = neuros_num[i][0], hidneuro2 = neuros_num[i][1])
    [model_name2, t2,acc_train2, acc_test2] = model2.fit_model(X_train2,Y_train2,y_train2,X_test2,Y_test2)
    s2 = pd.Series({"model":model_name2, "time": t2,"acc_train":acc_train2, "acc_test":acc_test2})
    pd.DataFrame(s).to_csv("models/{}.csv".format(model_name2))
    df2 = df2.append(s, ignore_index= True)
    print("{} done".format(model_name))
    
    
#%%
df.to_csv("exp_{}.csv".format(EXP_NUM),sep = ";")
#df["neuro1"] = df.model[:].apply(lambda x : int(x[3:].split("_")[0]))
#df["neuro2"] = df.model[:].apply(lambda x : int(x[3:].split("_")[1]))

#%%


##VISUALIZATION

##Иллюстрация для тезисов
#n1 = 10
#n2 = 14
#model_name1= "ASR_10_14_N2_cl2"
#model.model.load_weights("models\\{}model.h5".format(model_name1))

#ml2 = NeuroModel()
#ml2.compile_model(dataParams.N, hidneuro1 = n1, hidneuro2 = n2)
#ml2.model.load_weights("models\\{}model.h5".format(model_name))

n1 = 7
n2 = 11
#model_name2= "ASR_two_10_14_N2_cl2"
#ml1 = NeuroModel()
#ml1.compile_model(dataParams.N, hidneuro1 = n1, hidneuro2 = n2)
#ml1.model.load_weights("models\\{}model.h5".format(model_name2))
#model2.model.load_weights("models\\{}model.h5".format(model_name2))
#%%
X_train_arch = X_train
X_train2_arch = X_train2
y_train_arch = y_train
y_train2_arch = y_train2
#%%
ind = np.arange(0,X_train.shape[0],2)
X_train = X_train[ind]
y_train = y_train[ind]

ind2 = np.arange(0,X_train2.shape[0],2)
X_train2 = X_train2[ind2]
y_train2 = y_train2[ind2]

#%%
ml1 = model
ml2 = model2


fig = plt.figure(1, figsize = (10,5), dpi = 300)
X1 = X_train[(y_train == 1).flatten()]
X0 = X_train[(y_train == 0).flatten()]



xx,yy = np.meshgrid(np.arange(0,1,0.005), np.arange(0,1,0.005))
data = np.c_[xx.ravel(),yy.ravel()] 

Z = ml1.model.predict_on_batch(data)
res_Z_bin = Z.argmax(axis = 1)

level =0.0 
for i in np.arange(len(Z)): 
    #print(Z[i])
    if max(Z[i]) < level:
        res_Z_bin[i] =  -1

XZ0 = data[(res_Z_bin == 0)]
XZ1 = data[(res_Z_bin == 1 )]
ax1 = plt.subplot(122)

plt.xlim(0.0,1.0)
plt.ylim(0.0,1.0)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.scatter(XZ0[:,0],XZ0[:,1], color = '#888888', alpha = 0.3)
plt.scatter(XZ1[:,0],XZ1[:,1], color = "#BBBBBB", alpha = 0.3)


plt.scatter(X1[:,0],X1[:,1],36,color = "k", marker = "+")
plt.scatter(X0[:,0],X0[:,1],36, color = "k", marker ="_")

ax2 = plt.subplot(121, sharey = ax1)
plt.xlim(0.0,1.0)
plt.ylim(0.0,1.0)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

Z = ml2.model.predict_on_batch(data)
res_Z_bin = Z.argmax(axis = 1)

for i in np.arange(len(Z)): 
    #print(Z[i])
    if max(Z[i]) < level:
        res_Z_bin[i] =  -1
X1_2 = X_train2[(y_train2 == 1).flatten()]
X0_2 = X_train2[(y_train2 == 0).flatten()]


XZ0 = data[(res_Z_bin == 0)]
XZ1 = data[(res_Z_bin == 1 )]
plt.scatter(XZ0[:,0],XZ0[:,1], color = '#888888', alpha = 0.3)
plt.scatter(XZ1[:,0],XZ1[:,1], color = "#BBBBBB", alpha = 0.3)


plt.scatter(X1_2[:,0],X1_2[:,1],36, color = "k", marker = "+")
plt.scatter(X0_2[:,0],X0_2[:,1],36, color = "k", marker ="x")



#plt.text(1,1.1,s = "Train {:3f}".format(score_train))
#plt.text(1,1,s = "Test {:3f}".format(score_test))
plt.savefig("models/illustrate1_mark9.jpg")

score_test_1  = ml1.model.evaluate(X_test,Y_test,verbose =0)
score_test_2  = ml2.model.evaluate(X_test,Y_test,verbose =0)

## END VISUALISATION
#%%
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#n1 = np.arange(4,21,3)
#n2 =  np.arange(5,20,3)
#X3D, Y3D = np.meshgrid(n1, n2)
#surf = ax.scatter(X3D,Y3D,df.acc_train.values, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#surf = ax.plot_surface(X3D,Y3D,df.acc_train.values, cmap=cm.coolwarm, linewidth=0, antialiased=False)

#%%
#fig.colorbar(surf, shrink=0.5, aspect=5)
score_test  = model.model.evaluate(X_test,Y_test,verbose =0)
##print('Test score:',score_test[0])
#print('Test accuracy:',score_test[1])
#
#score_train  = model.evaluate(X_train,Y_train,verbose =0)
##print('Train score:',score_train[0])
#print('Train accuracy:',score_train[1])
# 
# plt.scatter(XZ0[:,0],XZ0[:,1], color = '# ', alpha = 0.3)
#%%
#ROC - кривые - порог
#res_train= model.predict_on_batch(X_train)
#res_train= model.predict_on_batch(X_train)
##grance = 0.5
#res_train_bin = res_train.argmax(axis = 1)
#        
#print('Error on train: ', percentError(res_train_bin, y_train)) # Подсчет ошибки
#
#res_test = model.predict_on_batch(X_test) # Подсчет результата на тестовой выборке
#res_test_bin = res_test.argmax(axis = 1)
#print('Error on test: ', percentError(res_test_bin,y_test)) # Подсчет ошибки

#%%


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