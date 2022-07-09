# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 16:27:38 2022

@author: Linus
"""

##imports
import pandas as pd
from factor_analyzer import FactorAnalyzer
import numpy as np
#import matplotlib.pyplot as plt


#read in data
# df= pd.read_csv("data_gaeng.csv")


##modifications on data files, needed initially, kept until end though
#read in files
df1= pd.read_csv("data_gaeng.csv")
df2= pd.read_csv("data_ga.csv")
df3= pd.read_csv("data_noga.csv") 
#list of column titles
cols1 = df1.columns.tolist()
cols2 = df2.columns.tolist()
cols3 = df2.columns.tolist()

#order as established in original questionnaire
myorder1 = [1, 3, 4, 6, 2, 7, 8, 19, 9, 15, 16, 17, 18, 45, 38, 
           44, 40, 41, 42, 43, 13, 14,  #  would go here,
           11, 10,
            12, 20, 21, 22, 23, 31, 26, 24, 25, 27, 29, 28,
            30, 32, 33, 34, 35, 36, 37, 39, 51, 52, 53, 54,
            50, 47, 48, 49, 46] #  5 55

myorder2 = [1,5,6,4,2,7,8,19,9,15,16, 17, 18, 45, 38, 
           44, 40, 41, 42, 43, 13, 14,  #  would go here
           11, 10,
            12, 20, 21, 22, 23, 31, 26, 24, 25, 27, 29, 28,
            30, 32, 33, 34, 35, 36, 37, 39, 51, 52, 53, 54,
            50, 47, 48, 49, 46] #3 55

myorder3 = [1,5,6,4,2,7,8,19,9,15,16, 17, 18, 45, 38, 
           44, 40, 41, 42, 43, 13, 14, #  would go here
           11, 10,
            12, 20, 21, 22, 23, 31, 26, 24, 25, 27, 29, 28,
            30, 32, 33, 34, 35, 36, 37, 39, 51, 52, 53, 54,
            50, 47, 48, 49, 46] # == list2

################
#alles checked, passt. aber ich hab die face-to-face (aktuell = 0) nicht drin.# muss da werte erfinden.
################

#sort them by that order (and rename one pesky column for later merge)
list1 = [cols1[i] for i in myorder1]
df1 = df1[list1]
df1.rename(columns = {'I am studying...':'What is your field of study?'}, inplace = True)

list2 = [cols2[i] for i in myorder2]
df2 = df2[list2]
list3 = [cols3[i] for i in myorder3]
df3 = df3[list3]

#to drop empty rows = participants who entered no value
#https://www.datacamp.com/tutorial/introduction-factor-analysis
df1.dropna(inplace=True)
df2.dropna(inplace=True)
df3.dropna(inplace=True)

#concatenate
df = pd.concat([df1, df2, df3])
df = df.iloc[:, 10:]
#### this is the list that can be worked with

#
# Factorability
# KMO
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df)

kmo_model

# #
#Bartlett
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(df)
chi_square_value, p_value


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