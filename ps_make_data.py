## Этот скрипт считывает сырые данные и готовит 
## данные к задаче ИЭ Бензина в унифицированном формате
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ps_data_lib

import os

currentdir = os.getcwd()
os.chdir("c:\\Users\\Public\\ps\\")

df = open_ps_2007()
df = df.append(open_ps_2009(),ignore_index = True)
df = df.append(open_ps_2009_12(),ignore_index = True)
df = df.append(open_ps_2006(),ignore_index = True)
df = df.append(open_ps_2005(),ignore_index = True)
df = df.append(open_ps_2008(),ignore_index = True)
df = df.append(open_ps_2010(),ignore_index = True)

df.to_csv("ps.csv")
writer = pd.ExcelWriter("ps.xlsx")
df.to_excel(writer,"PetroSpec")
df.to_excel("ps.xls", sheet_name="PetroSpec")



  

os.chdir(currentdir)