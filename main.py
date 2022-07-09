# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 16:27:38 2022

@author: Linus
"""

##imports
import pandas as pd
from factor_analyzer import FactorAnalyzer
#import matplotlib.pyplot as plt


#read in data
# df= pd.read_csv("data_gaeng.csv")


##modifications on data files, needed initially, kept until end though
df1= pd.read_csv("data_gaeng.csv")
# df2= pd.read_csv("data_ga.csv")
# df3= pd.read_csv("data_noga.csv") 

cols1 = df1.columns.tolist()
cols1
lista = (9,8,7)
#remove zeitstempel = 0
df1.columns
###da wo 50% steht = 15
myorder = [1, 3, 4, 6, 2, 7, 8, 19, 9, 15, 16, 17, 18, 45, 38, 
           44, 40, 41, 42, 43, 13, 14, 0, 11, 10,
            12, 20, 21, 22, 23, 31, 26, 24, 25, 27, 29, 28,
            30, 32, 33, 34, 35, 36, 37, 39, 51, 52, 53, 54,
            50, 47, 48, 49, 46] 
#checkd bis 32 in new list

 # HIER 0]#, 1, 4]
list1 = [cols1[i] for i in myorder]

#read in, snip, add together, remove that one header line (or do manually before), then save new name. ez
#then add with noga
#remove header line erst jetzt
#isa funzt
df1.columns
# df2.columns
# df3.columns
#reduce to (smaller, easier to check) testdataframe
# df = df[['How important are the following characteristics of your prospective future job? (1 = not at all important and 7=very important) SWIPE RIGHT TO SEE FULL SCALE [Being able to reach my supervisor easily in person]','How important are the following characteristics of your prospective future job? (1 = not at all important and 7=very important) SWIPE RIGHT TO SEE FULL SCALE [Meetings should be held in person]']]

#to drop empty rows = participants who entered no value
#https://www.datacamp.com/tutorial/introduction-factor-analysis
# df.dropna(inplace=True)


# df.head()

##
#Factorability
#KMO
# from factor_analyzer.factor_analyzer import calculate_kmo
# kmo_all,kmo_model=calculate_kmo(df)

# kmo_model

# #
# #Bartlett
# from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
# chi_square_value,p_value=calculate_bartlett_sphericity(df)
# chi_square_value, p_value


#split into two random samples? try EFA first. 
#because i definitely need to do it and also see first if data works
#if mandatory part is all done (EFA) then maybe add CFA

##Factor extraction (number of factors)
#kaiser kriterium
#scree plot
#parallel analysis
#cumulative variance
#factor loadings
#name factors
#maybe CFA later on 1/2 of sample (other 1/2 will do EFA)

#then cronbach alpha
#w omega composite reliability
#variance extracted scores
#convergent / discriminant validity


print("lol")