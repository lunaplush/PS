# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 12:44:51 2017
Get data from PS filies
@author: Inspiron
"""
import numpy as np
import pandas as pd

features = ["ro","RON", "MON", "OLF","ALK","AROMA","BNZ","KIS", \
            "TLL","MASS","METANOL","ETANOL","MTBE","ETBE","TAME","DIPE","TBS"]
X = []
Y = []
CLs = {}
            
def hello():
    print("PS data module")
    
def get_key(d, value):
    for k, v in d.items():
        if v == value:
            return k
#X преобразовать в цифру и заполнить пропуски
def to_float(x):
  
    try:
        return float(str(x).replace(",","."))
    except:
        return np.NAN

        
    
def data_processing(PS, fals = 1):
    global X,Y,CLs
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
                

    if fals == 1:
        
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
           
    
    
        CLs_ch =  lambda a: get_key(CLs,list(a.values)) 
        Y = PS_new[["name","provider"]].apply(CLs_ch, axis = 1)
        Y.name = "class_num"
    
        Y = pd.DataFrame(Y)
         
        for col in features:    
            X[col] = X[col].apply(to_float)
        borders = {'ro':[650,780], 'RON':[70,130],'MON':[76,130],'OLF':[-1,30],'ALK':[30,90], \
                  'AROMA':[-1,56], 'BNZ': [0,6], 'KIS':[0,20],'TLL':[-1,20],'MASS':[-1,50], \
                  'METANOL':[-1,1.1],'ETANOL':[-1,4],'MTBE':[-1,20],'ETBE':[-1,2],'TAME':[-1,2],'DIPE':[-1,2],\
                  'TBS':[-1,2]}
        i = 0    
        for (k,v)  in borders.items():
            Crit = np.logical_and(X[k]>v[0],X[k]<v[1]) 
            if i == 0:
                Crit_o = Crit.copy()
                i = 1
            else:
                Crit_o = np.logical_and(Crit_o,Crit)
    #   
        X = X[Crit_o]
        Y = Y[Crit_o]
    #Заполнение пустых значений
        for cl in CLs:
            x = X[Y["class_num"] == cl]
            means = (np.max(x) - np.min(x))/2 +np.min(x)
            X[Y["class_num"] == cl] = X[Y["class_num"] == cl].fillna(means)
    if fals == 100:
        PS_fals = PS[PS == PS_new][features]
        X = PS_fals[features]
        Y = [1]
   
        
#%%        
def open_ps_2007():
    global X,Y,CLs
    file = "PS 2007_Р.xlsx"
    xfile = pd.ExcelFile(file)
    df_cols = xfile.parse(sheetname = "columns", header = 3)
    columns = df_cols.columns   
    PS = xfile.parse(sheetname = "PetroSpec07", skiprows = [0,1,2], parse_cols = "A:V", names = columns )
    data_processing(PS)   
    
    
    return (X,Y,CLs)
#%%
def open_ps_2007_fals():
    global X,Y,CLs
    file = "PS 2007_Р.xlsx"
    xfile = pd.ExcelFile(file)
    df_cols = xfile.parse(sheetname = "columns", header = 3)
    columns = df_cols.columns   
    PS = xfile.parse(sheetname = "PetroSpec07", skiprows = [0,1,2], parse_cols = "A:V", names = columns )
    data_processing(PS,100)   
    
    
    return (X,Y)
#%%        
def open_ps_2009():
    global X,Y,CLs
  
    file = "PS 09.xls"
    xfile = pd.ExcelFile(file) 
    df_cols = xfile.parse(sheetname = "columns", header = 3)
    columns = df_cols.columns       
    PS = xfile.parse(sheetname = "PetroCpec 09", skiprows = [0,1,2], parse_cols = "A:W", names = columns )
    data_processing(PS)
     
    
    return (X,Y,CLs)
    
    
#%%    
if __name__ == "__main__":
    hello()
    X,Y,CLs = open_ps_2007()