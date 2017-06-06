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

        
    
def data_processing(PS):
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
    
    Crit1 =  np.logical_and(X['ro']>650, X['ro']< 900)
    Crit2 = np.logical_or(Crit1, X['ro'].isnull())
    Crit3 = np.logical_and(X["TLL"]  < 40, Crit2)
    Crit4 = np.logical_and(X["ETBE"] < 2, Crit3)
    Crit5 = np.logical_and(X["DIPE"] < 2, Crit4)
    Crit6 = np.logical_and(X["TAME"] < 2, Crit5)
    Crit7 = np.logical_and(X["AROMA"] < 90, Crit6)
    Crit8 = np.logical_and(X["RON"] <130, Crit7)
    Crit9 = np.logical_and(X["MON"] < 130, Crit8)
    Crit81 = np.logical_and(X["RON"] >60, Crit9)
    Crit91 = np.logical_and(X["MON"] > 60, Crit81)
    Crit10 = np.logical_and(X["OLF"] < 50, Crit91)
    Crit11 = np.logical_and(X["BNZ"] < 5, Crit10)
    X = X[Crit11]
    Y = Y[Crit11]
    for cl in CLs:
        x = X[Y["class_num"] == cl]
        means = (np.max(x) - np.min(x))/2 +np.min(x)
        X[Y["class_num"] == cl] = X[Y["class_num"] == cl].fillna(means)
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