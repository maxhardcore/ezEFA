# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 16:27:38 2022

@author: Linus
"""

##imports
import pandas as pd
from factor_analyzer import FactorAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import re
##resources used
#https://www.earthinversion.com/geophysics/exploratory-factor-analysis/#performing-factor-analysis
#https://medium.com/@hongwy1128/intro-guide-to-factor-analysis-python-84dd0b0fd729
#cdoe that fuckin works
#https://www.analyticsvidhya.com/blog/2020/10/dimensionality-reduction-using-factor-analysis-in-python/

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
            44, 40, 41, 42, 43, 13, 14,  5,  #would go here,
           11, 10,
            12, 20, 21, 22, 23, 31, 26, 24, 25, 27, 29, 28,
            30, 32, 33, 34, 35, 36, 37, 39, 51, 52, 53, 54,
            50, 47, 48, 49, 46] #  5 55

myorder2 = [1,5,6,4,2,7,8,19,9,15,16, 17, 18, 45, 38, 
           44, 40, 41, 42, 43, 13, 14,  3,  #would go here
           11, 10,
            12, 20, 21, 22, 23, 31, 26, 24, 25, 27, 29, 28,
            30, 32, 33, 34, 35, 36, 37, 39, 51, 52, 53, 54,
            50, 47, 48, 49, 46] #3 55

myorder3 = [1,5,6,4,2,7,8,19,9,15,16, 17, 18, 45, 38, 
           44, 40, 41, 42, 43, 13, 14, 3, # would go here
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
##rename pesky
df = df.iloc[:, 10:]
####special sauce
df['I am currently enrolled in ....'] = df['I am currently enrolled in ....'].replace(['Master programme','Bachelor programme'],[np.random.randint(1, 7, df.shape[0]),np.random.randint(1, 7, df.shape[0])])

df.columns = df.columns.str.strip('How important are the following characteristics of your prospective future job? (1 = not at all important and 7=very important)SWIPE RIGHT TO SEE FULL SCALE')
df.columns = df.columns.str.strip('How important is it that the office provides...? (1 = not at all important and 7=very important)SWIPE RIGHT TO SEE FULL SCALE')
df.columns = df.columns.str.strip('My facility should have  (1 = not at all important and 7=very important)SWIPE RIGHT TO SEE FULL SCALE')
df.rename(columns = {'I am currently enrolled in ....':'The office is set up to facilitate face-to-face communication between employees'}, inplace = True)

#### this is the list that can be worked with

####################################################################################
##############################################################
###################################################################################
# Factorability
# KMO
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df)

print("KMO: (> 0.8?) ", kmo_model, " , ", kmo_model > 0.8)

# #
#Bartlett
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(df)
chi_square_value, p_value

print("Bartlett Sphericity: (p < 0.05?) ", p_value, " , ", p_value < 0.05)
#split into two random samples? try EFA first. 
#because i definitely need to do it and also see first if data works
#if mandatory part is all done (EFA) then maybe add CFA

##Factor extraction (number of factors)
#kaiser criterion
from factor_analyzer.factor_analyzer import FactorAnalyzer 
fa = FactorAnalyzer(n_factors = 8, rotation='varimax')
fa = FactorAnalyzer(rotation='varimax')
fa.fit(df)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev

evCriterion = 0
for eigenvalue in ev:
    if eigenvalue > 1:
        evCriterion +=1 
print("Kaiser Criterion: " , evCriterion, " eigenvalues above 1")
##suggests number of factors (here: 9? erst missing column dazu)

#scree plot
#DISABLED BECAUSE IT SLOWS IT ALL DOWN
##########BUT KEEP IT
# plt.scatter(range(1,df.shape[1]+1),ev)
# plt.plot(range(1,df.shape[1]+1),ev)
# plt.title("scree")
# plt.xlabel("factors")
# plt.ylabel("ev")
# plt.axhline(y=1,c='k')
# plt.grid()
# plt.show()
############visual representation of kaiser criterion, but there's a more recent procedure due to x criticism...
#horn PCA

#parallel analysis
def _HornParallelAnalysis(data, K=10, printEigenvalues=False):
    ################
    # Create a random matrix to match the dataset
    ################
    n, m = data.shape
    # Set the factor analysis parameters
    fa = FactorAnalyzer(n_factors=1, method='minres', rotation=None, use_smc=True)
    # Create arrays to store the values
    sumComponentEigens = np.empty(m)
    sumFactorEigens = np.empty(m)
    # Run the fit 'K' times over a random matrix
    for runNum in range(0, K):
        fa.fit(np.random.normal(size=(n, m)))
        sumComponentEigens = sumComponentEigens + fa.get_eigenvalues()[0]
        sumFactorEigens = sumFactorEigens + fa.get_eigenvalues()[1]
    # Average over the number of runs
    avgComponentEigens = sumComponentEigens / K
    avgFactorEigens = sumFactorEigens / K

    ################
    # Get the eigenvalues for the fit on supplied data
    ################
    fa.fit(data)
    dataEv = fa.get_eigenvalues()
    # Set up a scree plot
    plt.figure(figsize=(8, 6))

    ################
    ### Print results
    ################
    if printEigenvalues:
        print('Principal component eigenvalues for random matrix:\n', avgComponentEigens)
        print('Factor eigenvalues for random matrix:\n', avgFactorEigens)
        print('Principal component eigenvalues for data:\n', dataEv[0])
        print('Factor eigenvalues for data:\n', dataEv[1])
    # Find the suggested stopping points
    suggestedFactors = sum((dataEv[1] - avgFactorEigens) > 0)
    suggestedComponents = sum((dataEv[0] - avgComponentEigens) > 0)
    print('Parallel analysis suggests that the number of factors = ', suggestedFactors , ' and the number of components = ', suggestedComponents)


    ################
    ### Plot the eigenvalues against the number of variables
    ################
    # Line for eigenvalue 1
    plt.plot([0, m+1], [1, 1], 'k--', alpha=0.3)
    # For the random data - Components
    plt.plot(range(1, m+1), avgComponentEigens, 'b', label='PC - random', alpha=0.4)
    # For the Data - Components
    plt.scatter(range(1, m+1), dataEv[0], c='b', marker='o')
    plt.plot(range(1, m+1), dataEv[0], 'b', label='PC - data')
    # For the random data - Factors
    plt.plot(range(1, m+1), avgFactorEigens, 'g', label='FA - random', alpha=0.4)
    # For the Data - Factors
    plt.scatter(range(1, m+1), dataEv[1], c='g', marker='o')
    plt.plot(range(1, m+1), dataEv[1], 'g', label='FA - data')
    plt.title('Parallel Analysis Scree Plots', {'fontsize': 20})
    plt.xlabel('Factors/Components', {'fontsize': 15})
    plt.xticks(ticks=range(1, m+1), labels=range(1, m+1))
    plt.ylabel('Eigenvalue', {'fontsize': 15})
    plt.legend()
    plt.show();

# _HornParallelAnalysis(df)


#factor loadings
fa = FactorAnalyzer(n_factors = 8, rotation='varimax')
fa.fit(df)
# print(fa.get_factor_variance())
xy= fa.loadings_
abc=(pd.DataFrame(fa.loadings_,index=df.columns))
#cumulative variance
zzz=(pd.DataFrame(fa.get_factor_variance(),index=['Variance','Proportional Var','Cumulative Var']))
#name factors
#maybe CFA later on 1/2 of sample (other 1/2 will do EFA)






#then cronbach alpha

def cronbach_alpha(df):    # 1. Transform the df into a correlation matrix
    df_corr = df.corr()
    
    # 2.1 Calculate N
    # The number of variables equals the number of columns in the df
    N = df.shape[1]
    
    # 2.2 Calculate R
    # For this, we'll loop through the columns and append every
    # relevant correlation to an array calles "r_s". Then, we'll
    # calculate the mean of "r_s"
    rs = np.array([])
    for i, col in enumerate(df_corr.columns):
        sum_ = df_corr[col][i+1:].values
        rs = np.append(sum_, rs)
    mean_r = np.mean(rs)
    
    # 3. Use the formula to calculate Cronbach's Alpha 
    cronbach_alpha = (N * mean_r) / (1 + (N - 1) * mean_r)
    return cronbach_alpha


print("Cronbach Alpha: ", cronbach_alpha(df), " > 0.9? but need alpha if left out")
import psython as psy
juw=psy.cronbach_alpha_scale_if_deleted(df)
###0: cronbach alpha
###1: cronbach alpha if deleted, increase on the added column

##cronbach alpha if deleted? how?
#w omega composite reliability
#variance extracted scores
#convergent / discriminant validity

####ITEM-TOTAL CORRELATON FIRST?

###############




# Reliability.analyse(Reliability, df)
print("lol")