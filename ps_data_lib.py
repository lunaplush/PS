## это библиотека работы с сырыми данными для задачи ИЭ бензниа
import numpy as np
import pandas as pd


#набор функций, которые считывают сырые данные из EXCEL файлов используя знания о структуре
#представления в них информации
def open_ps_2007():  
    file = "PS 2007_Р.xlsx"
    xfile = pd.ExcelFile(file)
    df_cols = xfile.parse(sheetname = "columns", header = 3)
    columns = df_cols.columns   
    PS = xfile.parse(sheetname = "PetroSpec07", skiprows = [0,1,2], parse_cols = "A:V", names = columns )
   
    return PS
    
def open_ps_2009():
    
    file = "PS 09.xls"
    xfile = pd.ExcelFile(file) 
    df_cols = xfile.parse(sheetname = "columns", header = 3)
    columns = df_cols.columns       
    PS = xfile.parse(sheetname = "PetroCpec 09", skiprows = [0,1,2], parse_cols = "A:W", names = columns )
      
    return PS

def open_ps_2009_12():
    
    file = "PS 09РДек.xls"
    xfile = pd.ExcelFile(file) 
    df_cols = xfile.parse(sheetname = "columns", header = 3)
    columns = df_cols.columns       
    PS = xfile.parse(sheetname = "PetroCpec 09", skiprows = [0,1,2,3], parse_cols = "A:W", names = columns )
      
    return PS


def open_ps_2006():
    file = "PS 2007.xls"
    xfile = pd.ExcelFile(file) 
    df_cols = xfile.parse(sheetname = "columns", header = 3)
    columns = df_cols.columns       
    PS = xfile.parse(sheetname = "PetroSpec06", skiprows = [0,1,2], parse_cols = "A:W", names = columns,skip_footer=106 )
    
    return PS
    
def open_ps_2005():
    file = "PS 2005_6_7.xlsx"
    xfile = pd.ExcelFile(file) 
    df_cols = xfile.parse(sheetname = "columns", header = 3)
    columns = df_cols.columns       
    PS = xfile.parse(sheetname = "PetroSpec 05", skiprows = [0,1,2], parse_cols = "A:V", names = columns,skip_footer=106 )

    return PS    

def open_ps_2008():
    file = "PS 08Мажейка.xls"
    xfile = pd.ExcelFile(file) 
    df_cols = xfile.parse(sheetname = "columns", header = 3)
    columns = df_cols.columns       
    PS = xfile.parse(sheetname = "PetroSpec08", skiprows = [0,1,2,3], parse_cols = "A:W", names = columns,skip_footer=106 )

    return PS    

def open_ps_2010():
    file = "PS 2010.xls"
    xfile = pd.ExcelFile(file) 
    df_cols = xfile.parse(sheetname = "columns", header = 3)
    columns = df_cols.columns       
    PS = xfile.parse(sheetname = "PetroCpec 2010", skiprows = [0,1,2,3], parse_cols = "A:W", names = columns,skip_footer=106 )

    return PS    


    
if __name__ == "__main__":
    print("This is module for PetroSpec data make")
    