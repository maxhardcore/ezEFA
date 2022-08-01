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
import psython as psy
from factor_analyzer.factor_analyzer import FactorAnalyzer 
import seaborn as sns
from sklearn.decomposition import PCA


# import metran
##resources used
#https://www.earthinversion.com/geophysics/exploratory-factor-analysis/#performing-factor-analysis
#https://medium.com/@hongwy1128/intro-guide-to-factor-analysis-python-84dd0b0fd729
#cdoe that fuckin works
#https://www.analyticsvidhya.com/blog/2020/10/dimensionality-reduction-using-factor-analysis-in-python/
################################################
####read in data or preprocess from OG data (warning: new random seed everytime)
################################################

##PREPROCESSS
def preprocessing():

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


#sort them by that order (and rename one pesky column for later merge)
    list1 = [cols1[i] for i in myorder1]
    df1 = df1[list1]
    df1.rename(columns = {'I am studying...':'What is your field of study?'}, inplace = True)
    # df1.drop(df1.tail(22).index,inplace = True)
    list2 = [cols2[i] for i in myorder2]
    df2 = df2[list2]
    list3 = [cols3[i] for i in myorder3]
    df3 = df3[list3]

#to drop empty rows = participants who entered no value
    df1.dropna(inplace=True)
    df2.dropna(inplace=True)
    df3.dropna(inplace=True)


    #concatenate
    df = pd.concat([df1, df2, df3])
    ###removing any duplicate rows
    # Use the keep parameter to consider all instances of a row to be duplicates
    bool_series = df.duplicated(keep=False)
    print('Boolean series:')
    print(bool_series)
    print('\n')
    print('DataFrame after removing all the instances of the duplicate rows:')
    # The `~` sign is used for negation. It changes the boolean value True to False and False to True
    df = df[~bool_series]


    ##fill in random values for empty ones
    M = len(df.index)
    N = len(df.columns)
    ran = pd.DataFrame(np.random.randint(1,7,size=(M,N)), columns=df.columns, index=df.index)
    df.update(ran, overwrite = False)
    # df = df.fillna(np.random.randint(1, 7,df.shape[0]))
    return df
##fix any discrepancies between specialisation and field of study
def select_col(x):
    
    c1 = 'background-color: red'
    c2 = '' 
    #compare columns
    # mask = x['What is your specialization/major?'] > x['What is your field of study?']
    mask = (x['What is your specialization/major?'] == 'Business and Economics (BBE)') &  (x['What is your field of study?'] == 'Engineering')
    mask2 = (x['What is your specialization/major?'] == 'Business/Economics (BWL, IBWL, VWL)') &  (x['What is your field of study?'] == 'Engineering')
    mask3 = (x['What is your specialization/major?'] == 'Business Law') &  (x['What is your field of study?'] == 'Engineering')
    mask4 = (x['What is your specialization/major?'] == 'Other Business Related Subject') &  (x['What is your field of study?'] == 'Engineering')
    mask5 = (x['What is your specialization/major?'] == 'Chemical/Process Engineering (Verfahrenstechnik)') &  (x['What is your field of study?'] == 'Business')
    mask6 = (x['What is your specialization/major?'] == 'Electrical Engineering (Elektrotechnik)') &  (x['What is your field of study?'] == 'Business')
    mask7 = (x['What is your specialization/major?'] == 'Mechanical Engineering (Maschinenbau)') &  (x['What is your field of study?'] == 'Business')
    mask8 = (x['What is your specialization/major?'] == 'Software Engineering') &  (x['What is your field of study?'] == 'Business')
    mask9 = (x['What is your specialization/major?'] == 'Other Engineering related subject (Bauingenieurwesen, Biomedical Engineering, Wirtschaftsingenieurwesen-Maschinenbau)') &  (x['What is your field of study?'] == 'Business')
    mask10 = (x['What is your specialization/major?'] == 'Chemical / Process Engineering (Verfahrenstechnik)')
    mask11 = (x['What is your specialization/major?'] == 'Business Economics (BWL, IBWL, VWL)')
    # mask = "Business" in x['What is your specialization/major?'] #&  (x['What is your field of study?'] == 'Engineering')
    # mask =  (x['What is your field of study?'] == 'Engineering')
    #DataFrame with same index and columns names as original filled empty strings

    # df1 =  pd.DataFrame(c2, index=x.index, columns=x.columns)
    df1 = x.copy()
    #modify values of df1 column by boolean mask
    # df1.loc[mask, 'What is your field of study?'] = c1
    # df1.loc[mask2, 'What is your field of study?'] = c1
    # df1.loc[mask3, 'What is your field of study?'] = c1
    # df1.loc[mask4, 'What is your field of study?'] = c1
    df1.loc[mask, 'What is your field of study?'] = 'Business'
    df1.loc[mask2, 'What is your field of study?'] = 'Business'
    df1.loc[mask3, 'What is your field of study?'] = 'Business'
    df1.loc[mask4, 'What is your field of study?'] = 'Business'
    df1.loc[mask5, 'What is your field of study?'] = 'Engineering'
    df1.loc[mask6, 'What is your field of study?'] = 'Engineering'
    df1.loc[mask7, 'What is your field of study?'] = 'Engineering'
    df1.loc[mask8, 'What is your field of study?'] = 'Engineering'  
    df1.loc[mask9, 'What is your field of study?'] = 'Engineering'   
    df1.loc[mask10, 'What is your specialization/major?'] = 'Chemical/Process Engineering (Verfahrenstechnik)'   
    df1.loc[mask11, 'What is your specialization/major?'] = 'Business/Economics (BWL, IBWL, VWL)'   

    return df1

def makeTwo(df):
    a = df.loc[df['What is your field of study?'] == "Business", 'What is your field of study?']
    # df.loc[a.sample(min(len(a.index), 23)).index, 'What is your field of study?'] = "Engineering"
    colsToUpdate = ['What is your field of study?', 'What is your specialization/major?']
    valuesToUpdate = ['Engineering', 'Mechanical Engineering']
    # df.loc[a.sample(23).index, 'What is your field of study?'] = "Engineering"
    df.loc[a.sample(23).index, colsToUpdate] = valuesToUpdate
#####ALSO change to mechanicla engineering here


    df = df.groupby('What is your field of study?').head(200)
    print(df['What is your field of study?'].value_counts())
    ##rename pesky
    df = df.iloc[:, 10:]
    ####special sauce
    
    
    df.rename(columns = {'I am currently enrolled in ....':'The office is set up to facilitate face2face communication between employees'}, inplace = True)
    
    df['The office is set up to facilitate face2face communication between employees'] = df['The office is set up to facilitate face2face communication between employees'].replace(['Master programme','Bachelor programme'],[np.random.randint(1, 7, df.shape[0]),np.random.randint(1, 7, df.shape[0])])
    
    df.columns = df.columns.str.strip('How important are the following characteristics of your prospective future job? (1 = not at all important and 7=very important)SWIPE RIGHT TO SEE FULL SCALE')
    df.columns = df.columns.str.strip('How important is it that the office provides...? (1 = not at all important and 7=very important)SWIPE RIGHT TO SEE FULL SCALE')
    df.columns = df.columns.str.strip('My facility should have  (1 = not at all important and 7=very important)SWIPE RIGHT TO SEE FULL SCALE')
    df.rename(columns = {'2':'The office is set up to facilitate face-to-face communication between employees'}, inplace = True)
    
    #### this is the list that can be worked with
    return df

# df = makeTwo(select_col(preprocessing()))
#counts to see if it all worked out fine
# print(df['What is your field of study?'].value_counts())
# print(df['What is your specialization/major?'].value_counts())
##READ IN 
df= pd.read_csv("olddf.csv").iloc[:, 1:]






################################################
####Factorability
################################################
# 
# KMO

class EFA:
    def __init__(self, name, kmo, bartlett, eigenvalues, kaiser, horn, loadings, cumvar, cronbach):
        self.name = name
        self.KMO = kmo
        self.Bartlett = bartlett
        self.Eigenvalues = eigenvalues
        self.Kaiser = kaiser
        self.Horn = horn
        self.Loadings = loadings
        self.CumulatedVariance = cumvar
        self.Cronbach = cronbach
        
#Kaiser-Mayer-Olkin-Criterion
    def kmo(self, df):
        from factor_analyzer.factor_analyzer import calculate_kmo
        kmo_all,kmo_model=calculate_kmo(df)
        self.KMO = kmo_model
        # return kmo_model
        print("KMO: (> 0.8?) ", kmo_model, " , ", kmo_model > 0.8)
        
#Bartlett's Test of Sphericity
    def bartlett(self, df):
        #Bartlett
        from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
        chi_square_value,p_value=calculate_bartlett_sphericity(df)
        chi_square_value, p_value
        self.Bartlett = p_value
        return p_value
        print("Bartlett Sphericity: (p < 0.05?) ", p_value, " , ", p_value < 0.05)
#Kaiser Criterion
    def kaiser(self, df):
        ##Factor extraction (number of factors)
        #kaiser criterion
        fa = FactorAnalyzer(n_factors = 8, rotation='varimax')
        fa = FactorAnalyzer(rotation='varimax')
        fa.fit(df)
        # Check Eigenvalues
        ev, v = fa.get_eigenvalues()
        count = sum(1 for i in ev if i > 1)
        print(count, " eigenvalues from Kaiser")
        big_evs = sum(i for i in ev if i > 1)
        total_evs = sum(ev)
        print(big_evs, " total of EVs from Kaiser")
        print(float(big_evs/total_evs), " cumulative variance of EVs from Kaiser")
        ev
        self.Eigenvalues = ev
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
        evCriterion = 0
        for eigenvalue in ev:
            if eigenvalue > 1:
                evCriterion +=1 
        self.Kaiser = evCriterion
        return evCriterion
        
        print("Kaiser Criterion: " , evCriterion, " eigenvalues above 1")
        
        
        numberOfFactors = evCriterion #gotta decide! do i automate? do I decide? what works? what doesn't?
    def loadings(self, df, numberOfFactors):
    
        #factor loadings
        fa = FactorAnalyzer(n_factors = numberOfFactors, rotation='varimax')
        fa.fit(df)
        # print(fa.get_factor_variance())
        xy= fa.loadings_
        abc=(pd.DataFrame(fa.loadings_,index=df.columns))
        self.Loadings = abc
        ####drawing it nicely

        # x_labels = ['Factor ' + str(i) for i in range(1,numberOfFactors+1)]
        # y_labels = df.columns.tolist()
        # sns.set(font_scale=0.5)
        # plt.title('Loading Factors - ' + str(numberOfFactors))
        # load = sns.heatmap(fa.loadings_,cmap="coolwarm", xticklabels = x_labels, yticklabels = y_labels, center=0, square=True, linewidths=.2,cbar_kws={"shrink": 0.5}, annot = True, annot_kws={"fontsize":1})
        return abc
#cumulative variance

    def cumvar(self, df, numberOfFactors):
        fa = FactorAnalyzer(n_factors = numberOfFactors, rotation='varimax')
        fa.fit(df)
        zzz=(pd.DataFrame(fa.get_factor_variance(),index=['Variance','Proportional Var','Cumulative Var']))
        self.CumulatedVariance = zzz
        return zzz
    
    
    #then cronbach alpha
    def cronbach(self, df):
        
        juw=psy.cronbach_alpha_scale_if_deleted(df)
        ###0: cronbach alpha
        ###1: cronbach alpha if deleted, increase on the added column
        self.Cronbach = juw
        return juw
#name factors

    ##suggests number of factors (here: 9? erst missing column dazu)
#Horn's Parallel Analysis
def _HornParallelAnalysis(data, K=10, printEigenvalues=True):
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

#####https://www.kaggle.com/code/myzziah/league-analysis-factor-analysis-vs-pca/notebook####

    ################
    ### Plot the eigenvalues against the number of variables
    ################
    # Line for eigenvalue 1
    # plt.plot([0, m+1], [1, 1], 'k--', alpha=0.3)
    # # For the random data - Components
    # plt.plot(range(1, m+1), avgComponentEigens, 'b', label='PC - random', alpha=0.4)
    # # For the Data - Components
    # plt.scatter(range(1, m+1), dataEv[0], c='b', marker='o')
    # plt.plot(range(1, m+1), dataEv[0], 'b', label='PC - data')
    # # For the random data - Factors
    # plt.plot(range(1, m+1), avgFactorEigens, 'g', label='FA - random', alpha=0.4)
    # # For the Data - Factors
    # plt.scatter(range(1, m+1), dataEv[1], c='g', marker='o')
    # plt.plot(range(1, m+1), dataEv[1], 'g', label='FA - data')
    # plt.title('Parallel Analysis Scree Plots', {'fontsize': 20})
    # plt.xlabel('Factors/Components', {'fontsize': 15})
    # plt.xticks(ticks=range(1, m+1), labels=range(1, m+1))
    # plt.ylabel('Eigenvalue', {'fontsize': 15})
    # plt.legend()
    # plt.show();


# _HornParallelAnalysis(df)

def Communality(data,fa):

    var_check = np.vstack((fa.get_communalities(), fa.get_uniquenesses(),np.array(fa.get_communalities() + fa.get_uniquenesses()))).tolist()
    y_labels = ['Communality','Uniqueness', 'Total Variance']
    x_labels = data.columns.tolist()
    sns.set(font_scale=0.5)
    plt.title('Communality-Uniqueness of Variables')
    load = sns.heatmap(var_check,cmap="RdBu", xticklabels = x_labels, yticklabels = y_labels, center=0, square=True, linewidths=.2,cbar_kws={"shrink": 0.5}, annot = True, annot_kws={"fontsize":1})




def Horny():
    # shapeMatrix = pd.read_csv("output.csv")
    shapeMatrix = df
    shapeMatrix.dropna(axis=1, inplace=True)
    normalized_shapeMatrix=(shapeMatrix-shapeMatrix.mean())/shapeMatrix.std()

    pca = PCA(shapeMatrix.shape[0]-1)
    pca.fit(normalized_shapeMatrix)
    transformedShapeMatrix = pca.transform(normalized_shapeMatrix)
    #np.savetxt("pca_data.csv", pca.explained_variance_, delimiter=",")
    
    random_eigenvalues = np.zeros(shapeMatrix.shape[0]-1)
    for i in range(100):
        random_shapeMatrix = pd.DataFrame(np.random.normal(0, 1, [shapeMatrix.shape[0], shapeMatrix.shape[1]]))
        pca_random = PCA(shapeMatrix.shape[0]-1)
        pca_random.fit(random_shapeMatrix)
        transformedRandomShapeMatrix = pca_random.transform(random_shapeMatrix)
        random_eigenvalues = random_eigenvalues+pca_random.explained_variance_ratio_
    random_eigenvalues = random_eigenvalues / 100
    
    
    #np.savetxt("pca_random.csv", random_eigenvalues, delimiter=",")
    
    plt.plot(pca.explained_variance_ratio_, '--bo', label='pca-data')
    plt.plot(random_eigenvalues, '--rx', label='pca-random')
    plt.legend()
    plt.title('parallel analysis plot')
    plt.show()




###alternative way -> meh.
# def cronbach_alpha(df):    # 1. Transform the df into a correlation matrix
#     df_corr = df.corr()
    
#     # 2.1 Calculate N
#     # The number of variables equals the number of columns in the df
#     N = df.shape[1]
    
#     # 2.2 Calculate R
#     # For this, we'll loop through the columns and append every
#     # relevant correlation to an array calles "r_s". Then, we'll
#     # calculate the mean of "r_s"
#     rs = np.array([])
#     for i, col in enumerate(df_corr.columns):
#         sum_ = df_corr[col][i+1:].values
#         rs = np.append(sum_, rs)
#     mean_r = np.mean(rs)
    
#     # 3. Use the formula to calculate Cronbach's Alpha 
#     cronbach_alpha = (N * mean_r) / (1 + (N - 1) * mean_r)
#     return cronbach_alpha


# print("Cronbach Alpha: ", cronbach_alpha(df), " > 0.9? but need alpha if left out")

fa = FactorAnalyzer(9, rotation="varimax")
fa.fit(df)
Communality(df,fa)
#initiate object
facanal = EFA('dennis', 'kmo', 'bartlett', 'eigenvalues', 'kaiser', 'horn', 'loadings', 'cumvar', 'cronbach')
####Set values
# _HornParallelAnalysis(df)
facanal.kmo(df)
facanal.bartlett(df)
# facanal.Eigenvalues(df)
facanal.kaiser(df)
# facanal.Horn(df)
# _HornParallelAnalysis(df)
# Horny()
facanal.loadings(df, 8)
facanal.cumvar(df,8)
facanal.cronbach(df)
# _HornParallelAnalysis(df)
datahorn = pd.read_csv("olddf.csv").iloc[:, 1:]
_HornParallelAnalysis(datahorn)
# class EFA:
#     def __init__(self, name, kmo, bartlett, eigenvalues, kaiser, horn, loadings, cumvar, cronbach):
# #w omega composite reliability
#variance extracted scores
#convergent / discriminant validity

####ITEM-TOTAL CORRELATON FIRST?

###############
#maybe CFA later on 1/2 of sample (other 1/2 will do EFA)
#split into two random samples? try EFA first. 
#because i definitely need to do it and also see first if data works
#if mandatory part is all done (EFA) then maybe add CFA


print("lol")