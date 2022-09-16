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
    
    #move Bach/master question to appropriate position
    column_to_move = df.pop("I am currently enrolled in ....")
    df.insert(2, "I am currently enrolled in ....", column_to_move)
    
    #insert face-to-face column with random integers 1 to 7 at appropriate position
    df.insert(23, "The office is set up to facilitate face-to-face communication between employees", np.random.randint(1, 7, df.shape[0]))
    df = df.sample(frac=1).reset_index(drop=True)
    # df.to_excel("workingQuestionnaireSmpl.xlsx")
    
    #cuts off non-numeric columns, is necessary for ease of mathematical operations in python
    df = df.iloc[:, 11:]
    ####special sauce
    
    #rename the column with "bach/master" to include question I did forget in online questionnaire
    # df.rename(columns = {'I am currently enrolled in ....':'The office is set up to facilitate face2face communication between employees'}, inplace = True)
    


    
    
    #fill with random values
    #replaces both bach and master with np-array of values from 1 to 7
    # df['The office is set up to facilitate face2face communication between employees'] = df['The office is set up to facilitate face2face communication between employees'].replace(['Master programme','Bachelor programme'],[np.random.randint(1, 7, df.shape[0]),np.random.randint(1, 7, df.shape[0])])
    # #strip the length of column names, too hard to read
    df.columns = df.columns.str.strip('How important are the following characteristics of your prospective future job? (1 = not at all important and 7=very important)SWIPE RIGHT TO SEE FULL SCALE')
    df.columns = df.columns.str.strip('How important is it that the office provides...? (1 = not at all important and 7=very important)SWIPE RIGHT TO SEE FULL SCALE')
    df.columns = df.columns.str.strip('My facility should have  (1 = not at all important and 7=very important)SWIPE RIGHT TO SEE FULL SCALE')
    # #rename "face2face" to "face-to-face"
    df.rename(columns = {"-to-":'The office is set up to facilitate face-to-face communication between employees'}, inplace = True)
    df.to_csv("workingQuestionnaireSmpl.csv")
    #### this is the list that can be worked with
    return df




################################################
####START OF EXPLORATORY FACTOR ANALYSIS
################################################

##PRE-PROCESS DATA
df = makeTwo(select_col(preprocessing()))

################################################
####Read in pre-processed data
################################################



##READ IN 
# df= pd.read_csv("workingQuestionnaireSmpl.csv").iloc[:, 1:]
# df = df.sample(frac=1).reset_index(drop=True)
# df= pd.read_excel("workingQuestionnaire_descTry.xlsx").iloc[:, 11:]
################################################
####Preliminary analysis of correlations via correlation matrix
################################################

#shows correlation matrix of data
corrMatrix = df.corr()

#counts number of occurrences of correlation <= 0.3 per row
#'how often does each item show a correlation <= 0.3 with other items?'
def count_values_in_range(series):
    return series.le(0.3).sum()
corrMatrix["n_values_in_range"] = corrMatrix.apply(
    func=lambda row: count_values_in_range(row), axis=1)

#dropping items with more than a specifc # of occurrences of above criterion
#this cutoff was arbitrarily chosen by author as being 2/3 = 66.6%
#Reason: 30 out of 44 "Likert-type"questions equals 68% of data

#number of occurrences noted as comment
dfPostDrop = df.drop(columns=['[There is a mandatory dresscode for employees]', #32
                 'The office is set up to facilitate face-to-face communication between employees', #43
                 '[Not being able to be reached after working hours]', #35,
                 '[Workspace setup in which I am spatially separated from the workspaces of my colleagues]', #31
                 '[To have access to generous, non-shared personal space at my workplace]', #32
                '[Smoking area]', #31
                ], axis = 1)
corrMatrixPostDrop = dfPostDrop.corr()
corrMatrixPostDrop["n_values_in_range"] = corrMatrixPostDrop.apply(
    func=lambda row: count_values_in_range(row), axis=1)

#highest # of occurences is now 25/38 = 65.8%


####Outputting correlation matrix as either MatPlot or CSV


# outputting as .csv
# df.to_csv('LikertData_4corrMatrix_Original.csv')
# dfPostDrop.to_csv('LikertData_4corrMatrix_PostDrop.csv')
# corrMatrix.to_csv('LikertData_corrMatrix_Original.csv')
# corrMatrixPostDrop.to_csv('LikertData_corrMatrix_PostDrop.csv')
# corrMatrixPostDrop.to_excel('LikertData_corrMatrix_PostDrop.xlsx')

####plotting:
# fig, ax = plt.subplots()
# fig.set_tight_layout(True)
# ax.tick_params(axis='both', which='major', labelsize=5)
# ax.tick_params(axis='both', which='minor', labelsize=5)
# ax = sns.heatmap(corrMatrixPostDrop.corr(), annot=True, fmt='.2f', ax=ax, cmap="YlGnBu", linewidths = .5); 
# ax.set_title("Post-drop correlation matrix", fontsize = 25)
# plt.show()



##Setting up a class which contains results of EFA

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
        
        
################################################
####Factorability of post-drop-data: KMO & Bartlett
################################################     



#Kaiser-Mayer-Olkin-Criterion (KMO)
    def kmo(self, df):
        from factor_analyzer.factor_analyzer import calculate_kmo
        kmo_all,kmo_model=calculate_kmo(dfPostDrop)
        self.KMO = kmo_model
        #Returns value of KMO criterion, should be above 0.8
        # return kmo_model
        print("KMO: (> 0.8?) ", kmo_model, " , ", kmo_model > 0.8)
        
#Bartlett's Test of Sphericity
    def bartlett(self, df):
        from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
        chi_square_value,p_value=calculate_bartlett_sphericity(dfPostDrop)
        chi_square_value, p_value
        self.Bartlett = p_value
        return p_value
    #Returns p-value of Bartlett's test, should be below 0.05
        print("Bartlett Sphericity: (p < 0.05?) ", p_value, " , ", p_value < 0.05)
        
        
################################################
####Factor Retention: number of factors considered 
####(Part 1): via Kaiser Criterion, Scree Plot
################################################    


#Kaiser Criterion
    def kaiser(self, df):

        #apply to non rotated data
        fa = FactorAnalyzer(rotation=None)
        fa.fit(df)
        # Check Eigenvalues for those exceeding 1
        ev, v = fa.get_eigenvalues()
        count = sum(1 for i in ev if i > 1)
        print(count, " eigenvalues above 1 - Kaiser")
        big_evs = sum(i for i in ev if i > 1)
        total_evs = sum(ev)
        print(big_evs, " total of EVs - Kaiser")
        print(float(big_evs/total_evs), " cumulative variance of EVs - Kaiser")
        ev
        self.Eigenvalues = ev
        
#Scree Plot
#############disabled for now to speed up

        
        # plt.scatter(range(1,df.shape[1]+1),ev)
        # plt.plot(range(1,df.shape[1]+1),ev)
        # plt.title("Scree Plot")
        # plt.xlabel("Number of Factors")
        # plt.ylabel("Eigenvalue of Factor")
        # plt.axhline(y=1,c='k')
        # plt.grid()
        # plt.show()
        evCriterion = 0
        for eigenvalue in ev:
            if eigenvalue > 1:
                evCriterion +=1 
        self.Kaiser = evCriterion
        return evCriterion
        
        
#Factor Loadings        
    def loadings(self, df, numberOfFactors):
    
        #Comparing results for various rotations
        # fa = FactorAnalyzer(n_factors = numberOfFactors, rotation='promax', method='ml')
        rotationsOrthogonal = ["varimax", "oblimax" , "quartimax" , "equamax" ]
        rotationsOblique = ["promax", "oblimin", "quartimin"]
        factorsOrthogonal = []
        factorsOblique = []
        #Orthogonal rotations
        for rota in rotationsOrthogonal:
            #fitting factor analyzer with various rotations
            fa = fa = FactorAnalyzer(n_factors=numberOfFactors, method='minres', rotation=rota)
            fa.fit(df)
            #getting loadings for data
            loadingsArray= fa.loadings_
            #converting np.array to DataFrame
            loadingsDataframe=(pd.DataFrame(fa.loadings_,index=df.columns))
            ##dropping all values below 0.32 (Lloret)
            loadingsDataframePostDrop = loadingsDataframe.where(abs(loadingsDataframe) > 0.32, np.nan)
            #add empty column with name of rotation for easier overview
            loadingsDataframePostDrop[rota] = ""
            factorsOrthogonal.append(loadingsDataframePostDrop)
        #Oblique rotations
        for rota in rotationsOblique:
            #fitting factor analyzer with various rotations
            fa = fa = FactorAnalyzer(n_factors=numberOfFactors, method='minres', rotation=rota)
            fa.fit(df)
            #getting loadings for data
            loadingsArray= fa.loadings_
            #converting np.array to DataFrame
            loadingsDataframe=(pd.DataFrame(fa.loadings_,index=df.columns))
            ##dropping all values below 0.32 (Lloret)
            loadingsDataframePostDrop = loadingsDataframe.where(loadingsDataframe > 0.32, np.nan)
            #add empty column with name of rotation for easier overview
            loadingsDataframePostDrop[rota] = ""
            factorsOblique.append(loadingsDataframePostDrop)
        
        
        
        
        
        
        self.Loadings = factorsOblique
        return factorsOrthogonal, factorsOblique
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

    
    
################################################
####Factor Retention: number of factors considered 
####(Part 2): via Horn's Parallel Analysis
################################################  

#Horn's Parallel Analysis


####printEigenvalues set to False, so it doesn't spam my screen for now
####repeat 100 times at minimum https://journals.sagepub.com/doi/pdf/10.1177/0095798418771807
def _HornParallelAnalysis(data, K=100, printEigenvalues=False):
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
    # plt.figure(figsize=(8, 6))

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



def Communality(data,fa):

    # var_check = np.vstack((fa.get_communalities(), fa.get_uniquenesses(),np.array(fa.get_communalities() + fa.get_uniquenesses()))).tolist()
    # y_labels = ['Communality','Uniqueness', 'Total Variance']
    # x_labels = data.columns.tolist()
    # sns.set(font_scale=0.5)
    # plt.title('Communality-Uniqueness of Variables')
    # load = sns.heatmap(var_check,cmap="RdBu", xticklabels = x_labels, yticklabels = y_labels, center=0, square=True, linewidths=.2,cbar_kws={"shrink": 0.5}, annot = True, annot_kws={"fontsize":1})
    print("Communality")





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



################################################
####Interpretation: calculating clear Factor loadings 
####without cross- or zero-loading items
################################################   




#initiate object
print("")
print("Initial run of EFA")
facanal = EFA('Initial EFA', 'kmo', 'bartlett', 'eigenvalues', 'kaiser', 'horn', 'loadings', 'cumvar', 'cronbach')

####Set values to EFA object
facanal.kmo(dfPostDrop)
facanal.bartlett(dfPostDrop)
facanal.kaiser(dfPostDrop)

###Calculate loadings with number of factors from Parallel Analysis
facanal.loadings(dfPostDrop, 8)



#---------------#



# drop non-loading items one-by-one and re-run EFA


print("")
print("Dropping first item, re-running")
# dropping first item
dfItemDrop1 = dfPostDrop.drop(columns=['[The office should be located centrally (close to infrastructure such as shops and restaurants)]', #32
                ], axis = 1)
#correlation matrix
corrMatrixItemDrop1 = dfItemDrop1.corr()
corrMatrixItemDrop1["n_values_in_range"] = corrMatrixItemDrop1.apply(
    func=lambda row: count_values_in_range(row), axis=1)

#number of factors?
# _HornParallelAnalysis(dfItemDrop1)

####Set values to EFA object
facanalItemDrop1 = EFA('First non-loading item dropped', 'kmo', 'bartlett', 'eigenvalues', 'kaiser', 'horn', 'loadings', 'cumvar', 'cronbach')

facanalItemDrop1.kmo(dfItemDrop1)
facanalItemDrop1.bartlett(dfItemDrop1)
facanalItemDrop1.kaiser(dfItemDrop1)

###Calculate loadings with number of factors from Parallel Analysis
facanalItemDrop1.loadings(dfItemDrop1, 8)



#---------------#


print("")
print("Dropping second item, re-running")
# dropping second item
dfItemDrop2 = dfItemDrop1.drop(columns=['[The office is set up to be barrier-free]',
                ], axis = 1)
#correlation matrix
corrMatrixItemDrop2 = dfItemDrop2.corr()
corrMatrixItemDrop2["n_values_in_range"] = corrMatrixItemDrop2.apply(
    func=lambda row: count_values_in_range(row), axis=1)

#number of factors?
# _HornParallelAnalysis(dfItemDrop2)

####Set values to EFA object
facanalItemDrop2 = EFA('Second non-loading item dropped', 'kmo', 'bartlett', 'eigenvalues', 'kaiser', 'horn', 'loadings', 'cumvar', 'cronbach')

facanalItemDrop2.kmo(dfItemDrop2)
facanalItemDrop2.bartlett(dfItemDrop2)
facanalItemDrop2.kaiser(dfItemDrop2)

###Calculate loadings with number of factors from Parallel Analysis
facanalItemDrop2.loadings(dfItemDrop2, 8)



#---------------#


print("")
print("Dropping third item, re-running")
# dropping third item
dfItemDrop3 = dfItemDrop2.drop(columns=['[Educational leave]',
                ], axis = 1)
#correlation matrix
corrMatrixItemDrop3 = dfItemDrop3.corr()
corrMatrixItemDrop3["n_values_in_range"] = corrMatrixItemDrop3.apply(
    func=lambda row: count_values_in_range(row), axis=1)

##[The office is easy to reach with public transportation]
##this item has a correlation under the threshold (0.3) for 24/35 (68.57%) items
##and will be dropped for this reason
dfItemDrop3PostDrop = dfItemDrop3.drop(columns=['[The office is easy to reach with public transportation]'], axis = 1)
corrMatrixItemDrop3PostDrop = dfItemDrop3PostDrop.corr()
corrMatrixItemDrop3PostDrop["n_values_in_range"] = corrMatrixItemDrop3PostDrop.apply(
    func=lambda row: count_values_in_range(row), axis=1)
#max. number of items surpassing threshold is now 23/35 = 65.71 %


# number of factors?
# _HornParallelAnalysis(dfItemDrop3PostDrop)

####Set values to EFA object
facanalItemDrop3 = EFA('Third non-loading item dropped', 'kmo', 'bartlett', 'eigenvalues', 'kaiser', 'horn', 'loadings', 'cumvar', 'cronbach')

facanalItemDrop3.kmo(dfItemDrop3PostDrop)
facanalItemDrop3.bartlett(dfItemDrop3PostDrop)
facanalItemDrop3.kaiser(dfItemDrop3PostDrop)

###Calculate loadings with number of factors from Parallel Analysis
facanalItemDrop3.loadings(dfItemDrop3PostDrop, 8)



#---------------#


###item [Cafeteria/restaurants] does not load to any factors for OBLIMIN and QUARTIMIN
#re-run
print("")
print("Dropping fourth item, re-running")
# dropping third item
dfItemDrop4 = dfItemDrop3PostDrop.drop(columns=['[Cafeteria/restaurants]',
                ], axis = 1)
#correlation matrix
corrMatrixItemDrop4 = dfItemDrop4.corr()
corrMatrixItemDrop4["n_values_in_range"] = corrMatrixItemDrop4.apply(
    func=lambda row: count_values_in_range(row), axis=1)

##[Parking facilities should be available for employees.]
##this item has a correlation under the threshold (0.3) for 23/34 (67.65%) items
##and will be dropped for this reason
dfItemDrop4PostDrop = dfItemDrop4.drop(columns=['[Parking facilities should be available for employees.]'], axis = 1)
corrMatrixItemDrop4PostDrop = dfItemDrop4PostDrop.corr()
corrMatrixItemDrop4PostDrop["n_values_in_range"] = corrMatrixItemDrop4PostDrop.apply(
    func=lambda row: count_values_in_range(row), axis=1)
#max. number of items surpassing threshold is now 22/34 = 64.70 %

#number of factors?
# _HornParallelAnalysis(dfItemDrop4PostDrop)

####Set values to EFA object
facanalItemDrop4 = EFA('Fourth non-loading item dropped', 'kmo', 'bartlett', 'eigenvalues', 'kaiser', 'horn', 'loadings', 'cumvar', 'cronbach')

facanalItemDrop4.kmo(dfItemDrop4PostDrop)
facanalItemDrop4.bartlett(dfItemDrop4PostDrop)
facanalItemDrop4.kaiser(dfItemDrop4PostDrop)

###Calculate loadings with number of factors from Parallel Analysis
facanalItemDrop4.loadings(dfItemDrop4PostDrop, 8)



#---------------#



###item [Fixed seating arrangement in which I have a desk/workspace which only I can use] 
###does not load to any factors
#re-run
print("")
print("Dropping fifth item, re-running")
# dropping fifth item
dfItemDrop5 = dfItemDrop4PostDrop.drop(columns=['[Fixed seating arrangement in which I have a desk/workspace which only I can use]',
                ], axis = 1)
#correlation matrix
corrMatrixItemDrop5 = dfItemDrop5.corr()
corrMatrixItemDrop5["n_values_in_range"] = corrMatrixItemDrop5.apply(
    func=lambda row: count_values_in_range(row), axis=1)




#number of factors?
# _HornParallelAnalysis(dfItemDrop5)

####Set values to EFA object
facanalItemDrop5 = EFA('Fifth non-loading item dropped', 'kmo', 'bartlett', 'eigenvalues', 'kaiser', 'horn', 'loadings', 'cumvar', 'cronbach')

facanalItemDrop5.kmo(dfItemDrop5)
facanalItemDrop5.bartlett(dfItemDrop5)
facanalItemDrop5.kaiser(dfItemDrop5)

###Calculate loadings with number of factors from Parallel Analysis
facanalItemDrop5.loadings(dfItemDrop5, 8)



#---------------#



###item [The office is illuminated by some degree of natural light] 
###shows cross-loading according to 
###https://www.researchgate.net/post/How-to-deal-with-cross-loadings-in-Exploratory-Factor-Analysis
###and will therefore be dropped
#re-run
print("")
print("Dropping cross-loading item, re-running")
# dropping cross-loading item
dfItemDrop6 = dfItemDrop5.drop(columns=['[The office is illuminated by some degree of natural light]',
                ], axis = 1)
#correlation matrix
corrMatrixItemDrop6 = dfItemDrop6.corr()
corrMatrixItemDrop6["n_values_in_range"] = corrMatrixItemDrop6.apply(
    func=lambda row: count_values_in_range(row), axis=1)




#number of factors?
# _HornParallelAnalysis(dfItemDrop6)

####Set values to EFA object
facanalItemDrop6 = EFA('Cross-loading item dropped', 'kmo', 'bartlett', 'eigenvalues', 'kaiser', 'horn', 'loadings', 'cumvar', 'cronbach')

facanalItemDrop6.kmo(dfItemDrop6)
facanalItemDrop6.bartlett(dfItemDrop6)
facanalItemDrop6.kaiser(dfItemDrop6)

###Calculate loadings with number of factors from Parallel Analysis
facanalItemDrop6.loadings(dfItemDrop6, 8)



#---------------#



###item [Meeting rooms] 
###shows cross-loading according to 
###https://www.researchgate.net/post/How-to-deal-with-cross-loadings-in-Exploratory-Factor-Analysis
###and will therefore be dropped
#re-run
print("")
print("Dropping second cross-loading item, re-running")
# dropping cross-loading item
dfItemDrop7 = dfItemDrop6.drop(columns=['[Meeting rooms]',
                ], axis = 1)
#correlation matrix
corrMatrixItemDrop7 = dfItemDrop7.corr()
corrMatrixItemDrop7["n_values_in_range"] = corrMatrixItemDrop7.apply(
    func=lambda row: count_values_in_range(row), axis=1)




#number of factors?
_HornParallelAnalysis(dfItemDrop7)

####Set values to EFA object
facanalItemDrop7 = EFA('Second cross-loading item dropped', 'kmo', 'bartlett', 'eigenvalues', 'kaiser', 'horn', 'loadings', 'cumvar', 'cronbach')

facanalItemDrop7.kmo(dfItemDrop7)
facanalItemDrop7.bartlett(dfItemDrop7)
facanalItemDrop7.kaiser(dfItemDrop7)

###Calculate loadings with number of factors from Parallel Analysis
facanalItemDrop7.loadings(dfItemDrop7, 8)




################################################
####Interpretation: naming the Factors
################################################
#since all rotations give the same results, only varying numerically in factor loadings.
#the author decided on OBLIMIN rotation for the sake of the thesis

#dataframe of factor loadings
factorLoadings = facanalItemDrop7.Loadings[1]
#dropping name of rotation from df
factorLoadings = factorLoadings.drop(columns=['oblimin',
                 ], axis = 1)

#renaming columns
factorLoadings.columns = ['Office climate', 'Provisions', 'Nature', 'Aesthetics', 
                          'Interpersonal communication', 'Solitary work', 'Open plan office', 'Autonomy']

##outputting to excel and csv
# factorLoadings.to_excel("FactorLoadingsNew.xlsx")
# factorLoadings.to_csv("FactorLoadingsNew.csv")
