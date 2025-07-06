""" The goal of this file is to combine the data sheets from Hofor with the geographic data from QGIS and run ML algortihms 
on them.
The data files imported to this file should have undergone prior analysis of which rows are relevant to keep, as no rows are
changed in this file. 
""" 

#%% 

# -------------  Importing packages and functions  -------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import datetime
import xgboost as xgb
import time
from functools import reduce

from sklearn.model_selection import train_test_split, KFold,GridSearchCV
from sklearn import svm
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer,f1_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import clone
# from imblearn.pipeline import Pipeline
# from imblearn.over_sampling import RandomOverSampler, SMOTE

import sys
sys.path.append('C:\\Users\\tvrj\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Arbejde\\Machine Learning')

from MLFunctions.utilities import * 
from MLFunctions.data_cleaning import *
from MLFunctions.oversamplers import * 
from MLFunctions.models import * 
from MLFunctions.visualization import * 
from MLFunctions.featureEngineering import * 
from MLFunctions.GAN import *
from MLFunctions.HyperParameterTuning import modelClassifierDic, modelRegressorDic, modelClassifierDic_fast, modelRegressorDic_fast

plt.rcParams.update({'font.size': 18})
plt.rcParams['font.family'] = 'Times New Roman'
red = '#990000'
blue = '#0070C0'
green = '#00B050'
yellow = '#E7B50C'

#%% 
# ---------  Loads data  --------- 
folderLoc = "C:\\Users\\tvrj\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Arbejde\\Data"
resultsLoc= "C:\\Users\\tvrj\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Arbejde\\Machine Learning\\Results"
dataTotal = pd.read_excel("{}\\HOFOR\\DataTotal_Broenshoej.xlsx".format(folderLoc))
pipeLength_total = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\PipeLengths.xlsx".format(folderLoc))
header = dataTotal.columns


#%% 
# ---------  Defines the model parameters  --------- 
# Defines which parts of the model should be run and with which parameters 
targetName = "FaultOccurance" #"FaultOccurance" or "Age". "nFaults" Is no  longer supported (Multiple functions need to be updated for this to work)
# combinePipes = True # Whether to combine the Havari 
splitDataByTime = True
useNewSplitMethod = True
doFeatureEngineering = True 
doFeatureEngineeringWithAllFeat = True
showBorutaScore = True # Indicates whether the Boruta Score of the model should be saved in a file. This is to only overwrite the file when wanted
downSampler = "KMean" # 'KMean' or 'RP' 
DSLevel = 1000 
overSampler = "ExpVSG" # 'ROS' or 'ExpVSG' or 'GAN' or 'Smote'
runGAN = False
desiredImbRatio = 3/4 # Describes the ratio between unbroken and broken pipes after DS and OS. Used to define OSLevel
optimizeHyperParameters = False
fastHPTuning = True 
updateHpOfBestModel = False # False or 'AUC' or ... # Defines which evaluator to optimize the HPs for. 
featuresToKeep = ['InnerDia', 'Age', 'Rain','FaultClustering','InnerPipeMate_Steel','Pressure_Winter','Pressure_Spring','Pressure_Summer','Pressure_Fall']#,"Length",'Anae_Depth','SoilCG_Zink','GWCG_Cya', 'GWCG_Lead','GWCG_Det','GWCG_Pest']#,'SupplyDist', 'nJoints']#,'SoilType_DS - Smeltevandssand','SoilType_FT - Ferskvandstoerv','SoilType_ML - Moraeneler', 'LandUse_Park etc.', 'LandUse_allotments','LandUse_residential','RoadType']
HPTuningName = 'FinalModel' # 'allFeat' 
nReruns = 1 # Number of times to run the simulation 

if nReruns != 1: # Changes randState to be random if multiple reruns are desired 
    optimizeHyperParameters = False
# Updates the parameters to use the optimal ones depending on the method used
if splitDataByTime: 
    eps_CZ = 140
    min_samples_CZ = 5
    if useNewSplitMethod: 
        if overSampler == 'ROS': 
            DSLevel = 3000
            desiredImbRatio = 1/4
        elif overSampler == "Smote": 
            DSLevel = 1000
            desiredImbRatio = 3/4
        elif overSampler == "ExpVSG": 
            DSLevel = 1000
            desiredImbRatio = 3/4
            
else: 
    downSampler = "KMean" 
    DSLevel = 1000 
    overSampler = "ROS" 
    desiredImbRatio = 1/2 
    eps_CZ = 60 
    min_samples_CZ = 3 

# modelType and X and y depending on the targetName: 
if targetName == "nFaults": 
    """When nFaults is used as the target, the havari data set should be neglected. (This the same data setup as Lasse). 
    The removal of the havari-data and the 'FaultOccurance' is done later, as the columns for now are used int he code. """
    y = pd.DataFrame({targetName:dataTotal[targetName]})
    X = dataTotal.drop([targetName], axis=1) 
    modelType = 'Regression'
    modelTypeName = "Reg" # ModelTypeName is used when naming files
    bestModelName = 'RFR'  
else:
    y = pd.DataFrame({targetName:dataTotal[targetName]})
    X = dataTotal.drop([targetName,'nFaults'], axis=1).copy()
    
    if targetName == "FaultOccurance":
        modelType = "Classification" 
        modelTypeName = "Class"
        bestModelName = 'RFC' 
        if useNewSplitMethod and overSampler == 'Smote':
            bestModelName = 'XGBC'
        # X = X.drop("Age",axis=1)
    elif targetName == 'Age':
        modelType = 'Regression'
        modelTypeName = "Reg"
        bestModelName = 'RFR'
        # X = X.drop("FaultOccurance",axis=1)
    else: 
        raise ValueError("Target name must either be 'FaultOccurance', 'Age' or 'nFaults'.")
featureNames = X.columns
print(featureNames)
print("Possible target values:",y[y.columns[0]].unique())

# Plots the distribution of the AgeGroups of the faulty pipes: 
# PlotAgeGroups(X,y)

# Defines general model hyper parameters
desiredTestSize = 0.3
if splitDataByTime: 
    desiredTestSize = 0.5 
trainSize = 1-desiredTestSize
randState = 1
if nReruns != 1: # Changes randState to be random if multiple reruns are desired 
    randState = None 
featureNotInModel = None # If any features should be removes from the model (If not, 'None' should be used)
minorityValue,majorityValue = DetermineMinorityAndMajority(y)
numberOfEnsembles = 3
numberOfEnsemplesSVM = 10
numberOfEstimatorsXGB = numberOfEnsembles 
forestSize = numberOfEnsembles 
nFolds = 5 
fileInfo = "Model:{} \nTrain size: {} \nOversampling Method: {} \nRandom state: {} \ny: {} \nFeature not in model{} \nk-Folds:{} \nEnsembles:{}, SVM:{}, RF:{}, XGB:{} \nDS: {} to {} points, OS: {}, Imb. Ratio: {} \n\n".format(modelType,trainSize,overSampler,randState,targetName,featureNotInModel,nFolds,numberOfEnsembles,numberOfEnsemplesSVM,forestSize,numberOfEstimatorsXGB,downSampler,DSLevel,overSampler,desiredImbRatio)


#%% 
# ---------  Cleaner data  ---------
# Fjerner vaerdier som ligger for langt vaek fra gennemsnittet
# X, y, strX = RemoveBadValues(X,y,featureNames)
strX = GetstrX(X)
if not strX.columns.empty:  
    # Definerer kolonner der skal Label Encodes paa og ranglisten af dataen fra lavest til hoejest. 
    Rankings = {"RoadType": ['FootPath','Residential','tertiary','Train','secondary']} 
    X = LEX(X,Rankings) 

    # Labelencoder y, hvis relevant
    # y = LEY(y,targetName)

    # Definerer hvilke kolonner der skal laves OneHot paa  (Det er ikke noedvendigvis alle kolonner), og foretager OneHot
    OneHotX = strX
    OneHotX = OneHotX.drop(Rankings.keys(),axis=1, errors='ignore')
    X,OneHotXColumns = OneHot(X,OneHotX.columns)
# Redefinerer featureNames, da denne vil have aendret sig: 
featureNames_Redefined = X.columns

if targetName == "nFaults": 
    # Here havari data are removed if the target is nFaults 
    roerIndexes = X[X["FaultOccurance"] == 0].index 
    y = y.loc[roerIndexes]
    X = X.loc[roerIndexes]    
    X = X.drop(["FaultOccurance"], axis=1) # Now Fault Occurance have been used for the last time and it can be removes if necessary for the model.

X_manipulated = RemoveFeature(X,featurename = featureNotInModel) # Removes the un-desired feature (if any):

X_havari = X_manipulated[pd.concat([X_manipulated,y],axis=1)["FaultOccurance"] == 1]
y_havari = y[pd.concat([X_manipulated,y],axis=1)["FaultOccurance"] == 1]
X_unB = X_manipulated[pd.concat([X_manipulated,y],axis=1)["FaultOccurance"] == 0]
y_unB = y[pd.concat([X_manipulated,y],axis=1)["FaultOccurance"] == 0]


#%% 

# ---------  Data Split and Fault Clustering  --------- 
# Starts a clock: 
startTime = time.perf_counter()
def SplitAndClusterZone():
    if splitDataByTime: 
        """For now this is only compatible with the classification model. """
        dataSplitName = 'Time'
        ### Data splitting 
        # Splits the havari data by year:
        if desiredTestSize == 0.5: 
            splitYear = 2016 # This year is part of the training data 
        elif desiredTestSize == 0.3: 
            splitYear = 2017 
        else: 
            raise ValueError("desiredTestSize can only be 0.3 or 0.5. {} was given.".format(round(desiredTestSize,1)))
        X_train_havari, y_train_havari, X_test_havari, y_test_havari, testSize = SplitHavariDataByYear(X_havari,y_havari,splitYear)
        trainSize = 1-testSize
        # Splits the unBroken data by quantity: 
        X_train_unB, X_test_unB, y_train_unB, y_test_unB = train_test_split(X_unB,y_unB,test_size=testSize,random_state=randState,stratify=X_unB['Age'])
        if useNewSplitMethod: 
            # New split method that takes 2017+ 2018 -> val and the rest to test 
            X_val_havari, y_val_havari, X_test_havari, y_test_havari, valSize = SplitHavariDataByYear(X_test_havari,y_test_havari,2017)
            # Splits the non-training, unBroken data by quantity again: 
            X_val_unB, X_test_unB, y_val_unB, y_test_unB = train_test_split(X_test_unB,y_test_unB,test_size=0.5,random_state=randState,stratify=X_test_unB['Age'])

            # Recombines the broken and non-broken data: 
            X_train,y_train = pd.concat([X_train_havari,X_train_unB],axis=0,ignore_index=True) , pd.concat([y_train_havari,y_train_unB],axis=0,ignore_index=True)
            X_test,y_test = pd.concat([X_test_havari,X_test_unB],axis=0,ignore_index=True) , pd.concat([y_test_havari,y_test_unB],axis=0,ignore_index=True)
            X_val,y_val = pd.concat([X_val_havari,X_val_unB],axis=0,ignore_index=True) , pd.concat([y_val_havari,y_val_unB],axis=0,ignore_index=True)

        else: 
            # Recombines the havari and unB data: 
            X_train,y_train = pd.concat([X_train_havari,X_train_unB],axis=0,ignore_index=True) , pd.concat([y_train_havari,y_train_unB],axis=0,ignore_index=True)
            X_test,y_test = pd.concat([X_test_havari,X_test_unB],axis=0,ignore_index=True) , pd.concat([y_test_havari,y_test_unB],axis=0,ignore_index=True)
            # Splits the test data set into vallidation and test data set: 
            # Old split method that randomly seperates validation and testing: 
            X_val, X_test, y_val, y_test = train_test_split(X_test,y_test,test_size=0.5,random_state=randState,stratify=y_test[targetName])
        

    else: 
        dataSplitName = 'Quantity'
        ### Data splitting: 
        # Resets the testSize as this will not change if the data is not split by time: 
        testSize = desiredTestSize
        trainSize = 1-testSize
        # Splits the data into teseting and training by quantity: 
        if modelType == "Classification" or targetName == 'nFaults': 
            X_train, X_test, y_train, y_test = train_test_split(X_manipulated,y,test_size=testSize,random_state=randState,stratify=y[targetName])
        elif modelType == "Regression": 
            X_train, X_test, y_train, y_test = train_test_split(X_manipulated,y,test_size=testSize,random_state=randState)
        X_train_havari = X_train[pd.concat([X_train,y_train],axis=1)["FaultOccurance"] == 1]
        X_val,y_val = pd.DataFrame({}),pd.DataFrame({})

    X_train,y_train = X_train.reset_index(drop=True),y_train.reset_index(drop=True) # Resets the index after the train-Test-split shuffled them
    X_test,y_test = X_test.reset_index(drop=True),y_test.reset_index(drop=True) # Resets the index after the train-Test-split shuffled them
    if splitDataByTime: 
        X_val,y_val = X_val.reset_index(drop=True),y_val.reset_index(drop=True) # Resets the index after the train-Test-split shuffled them
    ### Fault Clustering based on the Training data set only: 
    X_train, corePoints =  CheckIfPointInClusterZone(X_train.copy(), X_train_havari, eps=eps_CZ,min_samples=min_samples_CZ, plot=False)
    X_test, corePoints  =  CheckIfPointInClusterZone(X_test.copy(), X_train_havari, eps=eps_CZ,min_samples=min_samples_CZ)
    if splitDataByTime: 
        X_val, corePoints  =  CheckIfPointInClusterZone(X_val.copy(), X_train_havari, eps=eps_CZ,min_samples=min_samples_CZ)
    
    # Her er indexerne blevet shufflet en sidste gang, saa nu kan jeg seperere ID'erne og koordinaterne fra dem saa jeg altid har hvilket ID der hoerer til hvilket roer/havari-roer ud fra indekset: 
    ID_train, ID_test = X_train['ID'].copy(),X_test['ID'].copy()
    ID_all = pd.concat([ID_train,ID_test],axis=0)
    X_UTM_train, X_UTM_test = X_train[['X_UTM','Y_UTM']].copy(),X_test[['X_UTM','Y_UTM']].copy()
    X_train, X_test = X_train.drop(["ID","X_UTM","Y_UTM","Havari Year"],axis=1) , X_test.drop(["ID","X_UTM","Y_UTM","Havari Year"],axis=1) # Removes some columns, as their purpose over. 
    ID_val, X_UTM_val = pd.DataFrame({}), pd.DataFrame({}) 
    if splitDataByTime: 
        ID_val  = X_val['ID'].copy()
        X_UTM_val = X_val[['X_UTM','Y_UTM']].copy()
        X_val = X_val.drop(["ID","X_UTM","Y_UTM","Havari Year"],axis=1)
        ID_all = pd.concat([ID_all,ID_val],axis=0)

    # Samler testdataen, da det er relevant til regressionsmodellen. 
    data_test = pd.concat([X_test,y_test],axis=1)
    return  X_train,y_train, X_test,y_test, X_val,y_val, dataSplitName, testSize, trainSize, data_test, ID_train,ID_test,ID_val,ID_all, X_UTM_train,X_UTM_test,X_UTM_val
X_train,y_train, X_test,y_test, X_val,y_val, dataSplitName, testSize, trainSize, data_test, ID_train,ID_test,ID_val,ID_all, X_UTM_train,X_UTM_test,X_UTM_val = SplitAndClusterZone()
# PlotHavariYearDistribution(pd.DataFrame(X_train['Havari Year'].copy()))

#%%
# ---------  Scales the data  --------- 
def Scale(X_train,y_train, X_test, X_val):  
    # Skalerer alle tal i X med en standard scaler eller minMax scaler hvis de er blevet OneHOt Encoded (Except Columns describing IDs or x- or y-coordinates)
    X_train1,X_test1 = ScaleX(X_train.drop(OneHotXColumns,axis=1)), ScaleX(X_test.drop(OneHotXColumns,axis=1))
    X_train2,X_test2 = ScaleX(X_train[OneHotXColumns], scalingtype="Normalized"), ScaleX(X_test[OneHotXColumns], scalingtype="Normalized") # Rescales values that should be between 0-1
    X_train_scaled,X_test_scaled = pd.concat([X_train1,X_train2],axis=1), pd.concat([X_test1,X_test2],axis=1)

    X_train_havari_scaled = X_train_scaled[y_train[targetName] == 1]
    nBrokenPipes = X_train_havari_scaled.shape[0]
    nUnbrokenPipes = X_train_scaled.shape[0] - nBrokenPipes
    X_val1, X_val2, X_val_scaled = pd.DataFrame({}), pd.DataFrame({}), pd.DataFrame({})
    if splitDataByTime: 
        X_val1 = ScaleX(X_val.drop(OneHotXColumns,axis=1))
        X_val2 = ScaleX(X_val[OneHotXColumns], scalingtype="Normalized") # Rescales values that should be between 0-1
        X_val_scaled= pd.concat([X_val1,X_val2],axis=1)

    return X_train_scaled, X_test_scaled, X_val_scaled, nBrokenPipes,nUnbrokenPipes
X_train_scaled, X_test_scaled, X_val_scaled, nBrokenPipes,nUnbrokenPipes = Scale(X_train,y_train, X_test, X_val)


#%% 
# ---------  Model definition  ---------
# Definerer classifiers: 
LR = LogisticRegression(penalty='l2',C=1,random_state=randState)       # Probability function (Regression type model used for Classificaiton)
# SVC = svm.SVC(C=1, gamma = 'scale', kernel='rbf')   # Non-linear Decision Bounddary line which is placed as far away from the closest points in each cluster 
DTC = DecisionTreeClassifier() # Decision Tree, Resulting DB consists of small squares which fullfils the DTC criteria 
LRBagged = BaggingClassifier(estimator=LR, n_estimators=numberOfEnsembles,max_samples=trainSize,oob_score=False,random_state=randState) # n_estimators er hvor mange ensembles der skal laves. 
# SVCBagged = BaggingClassifier(estimator=SVC, n_estimators=numberOfEnsemplesSVM,max_samples=trainSize,oob_score=False,random_state=randState)
DTCBagged = BaggingClassifier(estimator=DTC, n_estimators=numberOfEnsembles,max_samples=trainSize,oob_score=False,random_state=randState)
XGBC = xgb.XGBClassifier(n_estimators=numberOfEstimatorsXGB,max_depth=2, learning_rate=1)
# RFC = RandomForestClassifier(n_estimators=20,max_depth=3,max_samples=0.7,warm_start=True,criterion='gini', oob_score=False) # Random Forest
# RFC = RandomForestClassifier(n_estimators=10,max_depth=3,max_samples=0.7,warm_start=True,criterion='log_loss', oob_score=False) # Random Forest
# RFC = RandomForestClassifier(n_estimators=40,max_depth=4,max_samples=0.5,warm_start=True,criterion='gini', oob_score=False) # Random Forest for ExpVSG
if overSampler == "GAN": 
    if downSampler == "KMean":
        RFC  =  RandomForestClassifier(n_estimators=4,max_depth=4,max_samples=0.8,warm_start=False,criterion='log_loss', oob_score=False)
    elif downSampler == "RP":
        RFC =  RandomForestClassifier(n_estimators=4,max_depth=4,max_samples=0.7,warm_start=False,criterion='entropy', oob_score=False)
elif overSampler == "ROS": 
    if downSampler == "KMean": 
        # RFC  =  RandomForestClassifier(n_estimators=10,max_depth=5,max_samples=0.6,warm_start=True,criterion='log_loss', oob_score=False)
        RFC  =  RandomForestClassifier(n_estimators=10,max_depth=6,max_samples=0.7,warm_start=False,criterion='log_loss', oob_score=False) 
    elif downSampler == "RP": 
        RFC =  RandomForestClassifier(n_estimators=4,max_depth=4,max_samples=0.8,warm_start=False,criterion='entropy', oob_score=False)
elif overSampler == "ExpVSG": 
    if downSampler == "KMean": 
        RFC  =  RandomForestClassifier(n_estimators=20,max_depth=8,max_samples=0.8,warm_start=False,criterion='log_loss', oob_score=False)
    elif downSampler == "RP": 
        RFC =  RandomForestClassifier(n_estimators=4,max_depth=4,max_samples=0.7,warm_start=False,criterion='log_loss', oob_score=False)
elif overSampler == "Smote": 
    if downSampler == "KMean": 
        RFC  =  RandomForestClassifier(n_estimators=4,max_depth=5,max_samples=0.5,warm_start=False,criterion='log_loss', oob_score=False)
        XGBC = xgb.XGBClassifier(n_estimators=5,max_depth=2, learning_rate=0.8)
    elif downSampler == "RP": 
        RFC =  RandomForestClassifier(n_estimators=4,max_depth=4,max_samples=0.7,warm_start=False,criterion='log_loss', oob_score=False)
        

# Definerer Regressors:
LinReg = LinearRegression()
# LinSVR = svm.LinearSVR()
# SVR = svm.SVR(C=1,gamma="scale",kernel="rbf")
DTR = DecisionTreeRegressor()
# SVRBagged = BaggingClassifier(estimator=SVR, n_estimators=numberOfEnsemplesSVM,max_samples=trainSize,oob_score=False,random_state=randState)
# DTRBagged = BaggingClassifier(estimator=DTR, n_estimators=numberOfEnsembles,max_samples=trainSize,oob_score=False,random_state=randState)
RFR = RandomForestRegressor(n_estimators=forestSize, oob_score=False)
XGBR = xgb.XGBRegressor(n_estimators=numberOfEstimatorsXGB,max_depth=2, learning_rate=1)

# Defines the best model: 
bestModel = clone(globals()[bestModelName])
print("Model used: {}".format(bestModelName))


#%% 
# ---------  Feature Engineering  ---------
def FeatureEngineering(): 
    featuresToKeep_kBest, featuresToKeep_RFE, featuresToKeep_boruta,featureRankings, evaluationScore_Boruta, borutaScore = None, None,  None, None,  None, None 
    if doFeatureEngineering: 
        ite = '2.0' 
        featEngineeringFeatures = X_train_scaled.columns
        if not doFeatureEngineeringWithAllFeat: 
            featEngineeringFeatures = featuresToKeep
        if modelType == "Classification":
            featuresToKeep_kBest = FeatureEngineering_kBest(X_train_scaled[featEngineeringFeatures],y_train,RFC,estimatorName="RFC{}".format(ite))
            print("kBest:",featuresToKeep_kBest)
            featuresToKeep_RFE = FeatureEngineering_RFE(X_train_scaled[featEngineeringFeatures],y_train,RFC,estimatorName="RFC{}".format(ite))
            print("RFE:",featuresToKeep_RFE)
        bestModel_coldStart = clone(bestModel)
        setattr(bestModel_coldStart,'warm_start',False) # Brouta does not work when 'warm_start' is True, thus it is changed to False
        featuresToKeep_boruta,featureRankings, evaluationScore_Boruta, borutaScore = FeatureEngineering_Boruta(X_train_scaled[featEngineeringFeatures],y_train,bestModel_coldStart,estimatorName="RFC_Cold{}".format(ite))
        print("Boruta:",featuresToKeep_boruta,borutaScore)
        if showBorutaScore: 
            dfBorutaScores = pd.DataFrame(borutaScore)
            dfBorutaScores.columns = featEngineeringFeatures
            dfBorutaScores = add_mean_row(dfBorutaScores)
            dfBorutaScores.to_excel("{}\\Broenshoej\\FeatureImportance\\Boruta\\BorutaScores.xlsx".format(resultsLoc),index=False)

    # We only want to use the relevant features in the model, which were decided previously:
    if HPTuningName == 'allFeat': 
        # Ignores all the chosen features if and selects them all. This is used for FeatureEngineering with RF. 
        X_train_relFeat,X_test_relFeat = X_train_scaled.copy(),X_test_scaled.copy()
        X_val_relFeat =  pd.DataFrame({})
        if splitDataByTime: 
            X_val_relFeat = X_val_scaled.copy()
    else: 
        X_train_relFeat,X_test_relFeat = X_train_scaled[featuresToKeep],X_test_scaled[featuresToKeep]
        X_val_relFeat = pd.DataFrame({})
        if splitDataByTime: 
            X_val_relFeat = X_val_scaled[featuresToKeep]
    return X_train_relFeat, X_test_relFeat, X_val_relFeat, featuresToKeep_kBest,featuresToKeep_RFE,featuresToKeep_boruta,featureRankings,evaluationScore_Boruta,borutaScore 
X_train_relFeat, X_test_relFeat, X_val_relFeat, featuresToKeep_kBest,featuresToKeep_RFE,featuresToKeep_boruta,featureRankings,evaluationScore_Boruta,borutaScore = FeatureEngineering()

#%% 
# ---------  Downsampling  ---------
def DownSampling(X_train_relFeat0,y_train0,downSampler0,DSLevel0,KMeansPlot=False): 
    if downSampler0 == 'RP':
        X_train_DS, y_train_DS = CalcEucledianDistances(X_train_relFeat0,y_train0,Plot=False,DSLevel=DSLevel0) # Undersamples X_train and y_train depending on the Euclidean distance to some reference points 
    elif downSampler0 == 'KMean': 
        X_train_DS, y_train_DS = kMeansClusteringDS(X_train_relFeat0,y_train0,n_clusters=DSLevel0,max_iter=300,Plot=KMeansPlot,targetname=targetName) 
    else: 
        raise ValueError("The down sampling method is incorrect. '{}' was given.".format(downSampler0))

    # Changes the ID df to only have the downsampled ID's 
    ID_train_DS = ID_train[X_train_DS.index]
    X_UTM_train_DS = X_UTM_train.iloc[X_train_DS.index]
    return X_train_DS, y_train_DS, ID_train_DS, X_UTM_train_DS
X_train_DS, y_train_DS, ID_train_DS, X_UTM_train_DS = DownSampling(X_train_relFeat,y_train,downSampler,DSLevel,KMeansPlot=True)


#%% 
# ---------  Oversampling  ---------
def OverSampling(X_train_DS,y_train_DS,overSampler0,OSLevel0:int=0,desiredImbRatio0:float=0): 
    # Oversampler de underrepraesenterede data: 
    X_train_havari_DS = X_train_DS[y_train_DS[targetName] == minorityValue]
    X_train_roer_DS    = X_train_DS[y_train_DS[targetName] == 0]
    nUnbrokenDS = X_train_roer_DS.shape[0] # number of unbroken pipes present in the data set after DS. 
    
    if desiredImbRatio0 !=0:
        OSTimes0 = int((nUnbrokenDS*desiredImbRatio0-nBrokenPipes)/nBrokenPipes) # Defines the number of nBroken pipes skal virtuelt genereres. 1 means the same number as the real samples, 2 means twice as many etc. Calculated from ImbRatio
        OSLevel0 = nBrokenPipes*OSTimes0  # Defines the number of virtual samles present after OS 
    elif OSLevel0 !=0:
        desiredImbRatio0 = OSLevel0/nUnbrokenDS # (OSTimes0*nBrokenPipes+nBrokenPipes)/nUnbrokenDS
        OSTimes0 = int(OSLevel0/nBrokenPipes)

    if runGAN:
        X_GAN = RunGAN(X_train_havari_DS,overSampleLevel=10,Plot=True,nEpochs=15000,saveName="x10,15000")
    else: 
        X_GAN = pd.read_excel("{}\\Broenshoej\\GAN\\fakeSamples_Classx10,15000.xlsx".format(resultsLoc))
        X_GAN = X_GAN[:OSLevel0]
    print("Number of majoritysamples: {}".format(y_train_DS[y_train_DS[targetName] == majorityValue].shape[0]))


    X_train_OS,y_train_OS = DoOverSampling(X_train_DS,y_train_DS,targetName,overSampler0,desiredImbRatio0,OSTimes0,modelType,minorityValue,majorityValue,X_GAN,randstate=randState)
    print("{} oversampling performed: From {} to {} data points".format(overSampler0,y_train_DS[y_train_DS[targetName] != majorityValue].shape[0],y_train_OS[y_train_OS[targetName] != majorityValue].shape[0]))
    print("Imbalance Ratio:{}".format(round(np.sum(y_train_OS[targetName] == 1)/np.sum(y_train_OS[targetName] == 0),2)))
    return X_train_OS,y_train_OS, X_train_havari_DS,X_train_roer_DS, nUnbrokenDS, OSTimes0, OSLevel0, X_GAN
X_train_OS,y_train_OS, X_train_havari_DS,X_train_roer_DS, nUnbrokenDS, OSTimes, OSLevel, X_GAN = OverSampling(X_train_DS, y_train_DS,overSampler,desiredImbRatio0=desiredImbRatio)


#%% 
# ---------  Hyper Parameter Tuning  ---------
""" Only the DS data is used to HP tune as the OS should be done after the val-train split in the k-CV. 
The values deciding the level of OS are used to make the OS identical to the one used in the remainder of this script. """
if fastHPTuning:
    HPTypeStr = "Fast"
else: 
    HPTypeStr = "Slow"

def HPTuner():
    HPTuning, bestModel0 = 0, clone(bestModel) # Creates starting instances of return values in case, no  HP Tuning should be done. 
    saveName_HPTuning="HPTuning{}_{},{}_{}={},{}={}_{}".format(HPTypeStr,dataSplitName,targetName[0:3],downSampler,DSLevel,overSampler,OSLevel,HPTuningName)
    if optimizeHyperParameters:  
        X_train_forHPTuning = pd.concat([X_train_DS,X_UTM_train_DS],axis=1) # Adds the coordinates to the training and testing data to be used on the fault Clustering in the k-CV
        X_test_forHPTuning  = pd.concat([X_test_relFeat,X_UTM_test],axis=1)
        
        if modelType == "Classification": 
            if fastHPTuning:
                modelDic = modelClassifierDic_fast 
            else:
                modelDic = modelClassifierDic
        elif modelType == "Regression": 
            if fastHPTuning:
                modelDic = modelRegressorDic_fast 
            else:
                modelDic = modelRegressorDic
        if splitDataByTime:
            X_val_forHPTuning = pd.concat([X_val_relFeat,X_UTM_val],axis=1) 
            HPTuning = PerformHPTuning(X_train_forHPTuning, y_train_DS,modelType,targetName,overSampler,desiredImbRatio,OSTimes,X_GAN,pipeLength_total,ID_train_DS,ID_test, X0_test=X_test_forHPTuning,y0_test=y_test, performGridSearch=True, X0_val=X_val_forHPTuning, y0_val=y_val, val_ID=ID_val,fileinfo=fileInfo,modeldic=modelDic,n_folds=nFolds,randstate=randState,savename=saveName_HPTuning) 
        else: 
            HPTuning = PerformHPTuning(X_train_forHPTuning, y_train_DS,modelType,targetName,overSampler,desiredImbRatio,OSTimes,X_GAN,pipeLength_total,ID_train_DS,ID_test, X0_test=X_test_forHPTuning,y0_test=y_test,fileinfo=fileInfo,modeldic=modelDic,n_folds=nFolds,randstate=randState,savename=saveName_HPTuning)
    else: 
        if updateHpOfBestModel: # If this is True, then HPTuning must be defined, which is the reason for the logic setup
            try:
                HPTuning = pd.read_excel("{}\\Broenshoej\\k-CrossValidation\\{}.xlsx".format(resultsLoc,saveName_HPTuning))
            except: 
                raise ValueError("HPTuning with these parameters has not been performed before or the file is unable to be loaded. Please set optimizeHyperParameters to True or check that the file is not open.")

    if updateHpOfBestModel != False:  # Automatically updates the bestModel to contain the best hyper parameters: 
        bestModel_forAUC, bestModel_forRecall, bestModel_forPrecision, bestModel_overall = DetermineBestModel(HPTuning)
        if updateHpOfBestModel == 'AUC':
            bestModel0 = clone(bestModel_forAUC)
        elif updateHpOfBestModel == 'Recall':
            bestModel0 = clone(bestModel_forRecall)
        elif updateHpOfBestModel == 'Precision':
            bestModel0 = clone(bestModel_forPrecision)
        elif updateHpOfBestModel == 'Overall':
            bestModel0 = clone(bestModel_overall)
        else: 
            print("--- No Desired best model parameter inputted. AUC is used. ---")
            bestModel0 = clone(bestModel_forAUC)
    # if isinstance(bestModel, (svm.SVC)): 
    #     setattr(bestModel,'probability',True)

    return HPTuning, bestModel0
HPTuning, bestModel = HPTuner()

#%% 
# ---------  Plots Results of 'bestModel'  ---------
def DetermineFinalDataSets(): 
    X_train_final, y_train_final = X_train_OS,y_train_OS
    X_test_final, y_test_final = X_test_relFeat, y_test
    X_val_final, y_val_final = pd.DataFrame({}), pd.DataFrame({})
    if splitDataByTime: 
        X_val_final, y_val_final = X_val_relFeat, y_val
    y_test_final_withID = pd.concat([ID_test,y_test_final],axis=1)
    return X_train_final,y_train_final, X_test_final,y_test_final, X_val_final, y_val_final, y_test_final_withID 
X_train_final,y_train_final, X_test_final,y_test_final, X_val_final, y_val_final, y_test_final_withID = DetermineFinalDataSets()

if modelType == "Classification":
    # Defines the hyper parameter's of the best model:
    HPStr = FindChangedHPs(bestModel)
    print("Model used: {}".format(bestModel))
    print("Used HPs: {}".format(HPStr))
    saveName = "{},{},{},{},{},{}={},{}={},{}".format(modelType[0:3],trainSize,randState,nFolds,featureNotInModel,downSampler,DSLevel,overSampler,OSLevel,HPStr)

    y_pred, y_pred_perc, fittedClassifier,f1Score, accuracy, recall,precision, modelWith2Feat = TestClassificationModel(bestModel,X_train_final,y_train_final,X_test_final,y_test_final, feat1_name='Age',feat2_name='InnerDia')
    # Second part of the feature Engineering (That has to be done after the simulation is finished. )
    if doFeatureEngineering and isinstance(fittedClassifier, RandomForestClassifier):
        featuresToKeep_RF,scoreList_RF = FeatureEngineering_RF(X_train_final.columns,fittedClassifier)
        print("Features to keep according to RF: {}".format(list(featuresToKeep_RF)))
        print("Feature scores from RF:\t\t  {}".format(np.round(np.array(scoreList_RF),2)))
    
    ### Visualisering af diverse resultater 
    # Confusion Matrix:
    ConfusionMatrix(y_test_final,y_pred, savename = saveName,title="Recall",Normalise=None)
    
    # ROC curves: 
    # PlotROC(y_test_final, y_pred_perc,savename=saveName,bestmodelname=bestModelName)
    # PlotPRCurve(y_test_final, y_pred_perc,savename=saveName,bestmodelname=bestModelName)
    
    y_pred_perc_df = pd.DataFrame({"Pred%":y_pred_perc})
    PlotLvsFCapture(pipeLength_total,y_test_final_withID,y_pred_perc_df, savename=saveName)

    # aendrer cut-off threshold (phi), og plotter CM igen. 
    phi_bestPrecision, phi_50Precision,phi_bestF1, CMvalues_df, bestF1Score = DeterminePhiAt50Precision(y_pred_perc_df, y_test_final, plot = True)
    y_pred_phiBestPrecision = np.ones(len(y_pred))
    y_pred_phi50Precision = np.ones(len(y_pred))
    y_pred_phiBestF1 = np.ones(len(y_pred))
    y_pred_phiBestPrecision[y_pred_perc < phi_bestPrecision] = 0 # Changes the percentage at which a pipe is declared 'faulty' 
    y_pred_phi50Precision[y_pred_perc < phi_50Precision] = 0 # Changes the percentage at which a pipe is declared 'faulty' 
    y_pred_phiBestF1[y_pred_perc < phi_bestF1] = 0 # Changes the percentage at which a pipe is declared 'faulty' 
    ConfusionMatrix(y_test_final,y_pred_phiBestPrecision, savename = saveName + ",phi={}".format(phi_bestPrecision),title="Phi",Normalise=None, phi=phi_bestPrecision)
    # ConfusionMatrix(y_test_final,y_pred_phi50Precision, savename = saveName + ",phi={}".format(phi_50Precision),title="Phi",Normalise=None, phi=phi_50Precision)
    ConfusionMatrix(y_test_final,y_pred_phiBestF1, savename = saveName + ",phi={}".format(phi_bestF1),title="Phi",Normalise=None, phi=phi_bestF1)

    # Plots the y_pred_perc Distribution: 
    y_havari_pred_perc = y_pred_perc[y_test_final[targetName] == 1]
    y_roer_pred_perc   = y_pred_perc[y_test_final[targetName] == 0]
    Ploty_percDistribution(y_roer_pred_perc,y_havari_pred_perc,lowerylim = 0.0,lowerxlim=0,phi=phi_bestF1*100)

    # Plotter laengde-distributionen af havari-roer: 
    # PlotDistributionOfDfColumn(X_havari,'Length',plotType='Bar',title="Length Distribution in Faults", barIntervals=np.linspace(0,0.7,11))
    # PlotDistributionOfDfColumn(X,'Length',plotType='Bar',title="Length Distribution in all pipes", barIntervals=np.linspace(0,0.7,11))
    
    # Plotter resultaterne som funktion af 2 af featuerne
    # ScatterTestWithDB(X_train_ROS,  y_train_ROS, X_test_final, y_test_final, modelWith2Feat,feat1Name,feat2Name,Title="Targets")


elif targetName == 'nFaults': 
    y_pred, Model, R2_test, R2_train = TestRegressionModel(bestModel,X_train_final,y_train_final,X_test_final,y_test_final)
    print("R2 (test): {} " \
    "R2 (train): {}".format(round(R2_test,2),round(R2_train,2)))
    # Visualize results: 
    plt.scatter(range(len(y_pred[targetName])),y_pred[targetName])

    # Length capture vs Fault capture: 
    # y_test_final_withID = pd.concat([ID_test,y_test_final],axis=1)
    # y_pred_perc_df = pd.DataFrame({"Pred%":y_pred_perc})
    # PlotLvsFCapture(pipeLength_total,y_test_final_withID,y_pred_perc_df, savename=saveName)

elif targetName == "Age":
    # Beregner Predictede Ages: 
    # print(X_test_final.head(), y_test_final.head())
    # data_havari = pd.concat([X_test_final,y_test_final],axis=1,ignore_index=True).copy()
    # print(data_havari)
    # print(data_test)
    data_havari = data_test[data_test["FaultOccurance"] == data_test["FaultOccurance"].unique()[1]].copy()
    y_test_final_havari = pd.DataFrame(data_havari["Age"]).copy()
    X_test_final_havari = data_havari.drop("Age",axis=1).copy()
    
    

    # data_havari = data_havari[data_havari["FaultOccurance"] == data_havari["FaultOccurance"].unique()[1]].copy()
    
    # y_test_final_havari = y_test_final[X_test_final["FaultOccurance"] == X_test_final["FaultOccurance"].unique()[1]].copy()
    
    # print(X_test_final_havari.head(),y_test_final_havari.head())
    # X_test_final_havari =X_test_final[havariBool].copy()
    # y_test_final_havari = y_test_final[havariBool].copy()
    # print(X_test_final_havari, y_test_final_havari)
    
    y_pred, Model, R2_test, R2_train = TestRegressionModel(LinReg,X_train_final,y_train_final,X_test_final,y_test_final)
    print("LinReg: Test: {}, Train: {}".format(R2_test,R2_train))
    y_pred, Model, R2_test, R2_train = TestRegressionModel(DTR,X_train_final,y_train_final,X_test_final,y_test_final)
    print("LinReg: Test: {}, Train: {}".format(R2_test,R2_train))
    y_pred, Model, R2_test, R2_train = TestRegressionModel(RFR,X_train_final,y_train_final,X_test_final,y_test_final)
    print("LinReg: Test: {}, Train: {}".format(R2_test,R2_train))
    y_pred, Model, R2_test, R2_train = TestRegressionModel(XGBR,X_train_final,y_train_final,X_test_final,y_test_final)
    print("LinReg: Test: {}, Train: {}".format(R2_test,R2_train))

    y_pred_havari, fittedRegressor, R2_test_havari, R2_train_havari = TestRegressionModel(RFR,X_train_final,y_train_final,X_test_final_havari,y_test_final_havari)
    
    # Feature Importance: 
    FIR = fittedRegressor.feature_importances_
    FIRdf = pd.DataFrame({"Features":X_test_final.columns, "FIR":FIR})
    FIRdf.to_excel("{}\\Broenshoej\\HyperParameterTuning\\FIR_{}.xlsx".format(resultsLoc,"RFC"))
    # for i,fir in enumerate(FIR):

    #     print(X_test_final.columns[i],":",fir) 

    # print(X_test_final["FaultOccurance"].unique())
    # havariBool = X_test_final["FaultOccurance"] == X_test_final["FaultOccurance"].unique()[1]
    # print(havariBool, np.sum(havariBool))
    # print(y_test_final, y_pred)
    # y_test_final_havari = y_test_final[havariBool]
    # y_pred_havari = y_pred[havariBool]
    # print(y_test_final_havari,y_pred_havari)
    # Plotter: 
    BarplotRegressionModel(y_test_final,y_predict=y_pred,savename=saveName,R2=R2_test)
    BarplotRegressionModel(y_test_final_havari,y_predict=y_pred_havari,savename=saveName,R2=R2_test)

    # Plotter alderfordelingen af de roer vi undersoeger: 
    BarplotRegressionModel(pd.concat([y_test_final,y_train_OS],axis=0,ignore_index=True))
    BarplotRegressionModel(y_havari)    

    # BarplotRegressionModel(y_test_final_havari)

#%% 
# ---------  Validation curves  --------- 
def PlotValidationCurve(data,modeltoplot:str,paramofinterest:str,otherparams={},n_folds:int=5,evaluation='all'):
    """Plots the accuracy, recall and precision of the test results from the hyper parameter tuning. 
    Every hyperparameter that is not the one being plotted must be given a specific value in the otherparams-dic with the parameter name as the key. 
    input: 
        data: df of results from hyper parameter tuning 
        otherparams: dic of other hyper parameters used in the estimator. These must have a fixed value presented in the dic. The HP name is the key. 
    output: 
        Desired plot of the data 
    """
    plotData = data[data["Model"] == modeltoplot].copy()
    plotData = plotData[~plotData[paramofinterest].isna()].copy()
    if len(plotData) == 0:    
        raise ValueError("{} is not one of the hyperparameters in {}.".format(paramofinterest,modeltoplot))
    # Chooses the data with the desired hyperparameters 
    if otherparams: 
        for key in otherparams.keys(): 
            value = otherparams[key]
            plotData = plotData[plotData[key] == value]
    parameter = plotData[paramofinterest]

    AUCVal = plotData["AUCValScores"]
    AUCTrain = plotData["AUCTrainScores"]
    recallVal = plotData["recallValScores"]
    recallTrain = plotData["recallTrainScores"] 
    accuracyVal = plotData["accuracyValScores"]
    accuracyTrain = plotData["accuracyTrainScores"]
    precisionVal = plotData["precisionValScores"]
    precisionTrain = plotData["precisionTrainScores"]

    # Standard Deviation of the mean / Error: 
    AUCValSDOM = plotData["AUCValStd"]/np.sqrt(n_folds)
    AUCTrainSDOM = plotData["AUCTrainStd"]/np.sqrt(n_folds)
    recallValSDOM = plotData["recallValStd"]/np.sqrt(n_folds)
    recallTrainSDOM = plotData["recallTrainStd"]/np.sqrt(n_folds) 
    accuracyValSDOM = plotData["accuracyValStd"]/np.sqrt(n_folds)
    accuracyTrainSDOM = plotData["accuracyTrainStd"]/np.sqrt(n_folds)
    precisionValSDOM = plotData["precisionValStd"]/np.sqrt(n_folds)
    precisionTrainSDOM = plotData["precisionTrainStd"]/np.sqrt(n_folds)
    
    # Plots: 
    plt.figure() 
    if evaluation == 'all':
        plt.errorbar(parameter, AUCVal, yerr=AUCValSDOM, fmt='-o', label='AUC Val Scores', color='blue', alpha=0.7, ecolor='blue', capsize=5)
        plt.errorbar(parameter, AUCTrain, yerr=AUCTrainSDOM, fmt='-o', label='AUC Train Scores', color='cyan', alpha=0.7, ecolor='cyan', capsize=5)
        plt.errorbar(parameter, recallVal, yerr=recallValSDOM, fmt='-o', label='Recall Val Scores', color='green', alpha=0.7, ecolor='green', capsize=5)
        plt.errorbar(parameter, recallTrain, yerr=recallTrainSDOM, fmt='-o', label='Recall Train Scores', color='lightgreen', alpha=0.7, ecolor='lightgreen', capsize=5)
        plt.errorbar(parameter, accuracyVal, yerr=accuracyValSDOM, fmt='-o', label='Accuracy Val Scores', color='red', alpha=0.7, ecolor='red', capsize=5)
        plt.errorbar(parameter, accuracyTrain, yerr=accuracyTrainSDOM, fmt='-o', label='Accuracy Train Scores', color='orange', alpha=0.7, ecolor='orange', capsize=5)
        plt.legend(loc ='center right', bbox_to_anchor=(1.36, 0.5))
    elif evaluation == 'AUC': 
        plt.errorbar(parameter, AUCVal, yerr=AUCValSDOM, fmt='-o', label='AUC Val Scores', color='blue', alpha=0.7, ecolor='blue', capsize=5)
        plt.errorbar(parameter, AUCTrain, yerr=AUCTrainSDOM, fmt='-o', label='AUC Train Scores', color='cyan', alpha=0.7, ecolor='cyan', capsize=5)
        plt.ylim(np.min(AUCVal-AUCValSDOM)-0.05,1)
        plt.legend(loc ='lower right')

    plt.xlabel(paramofinterest)
    plt.ylabel('Scores')
    plt.title('Validation Curve for {}'.format(modeltoplot))
    plt.grid(True)
    fileName = modeltoplot+"_"+paramofinterest
    plt.savefig("{}\\Broenshoej\\HyperParameterTuning\\ValidationCurves\\{}.png".format(resultsLoc,fileName))
    return plotData, parameter
# HPTuning = pd.read_excel("{}\\Broenshoej\\k-CrossValidation\\{}.xlsx".format(resultsLoc,'HPTuningSlow_Quantity,Fau_KMean=1000,ROS=415_noAgeGroup'), sheet_name='Sheet1')
# print(HPTuning.head()) 
try:
    # Only allowes validation curves to be made, if a HPTuning file has been loaded
    HPTuning.shape  
except: 
    print("HPTuning not defined, No Validation curves will be made.")
else: 
    _,__ = PlotValidationCurve(HPTuning,"RFC",'max_samples',otherparams={'n_estimators':20,'max_depth':6,'criterion':'entropy'},n_folds=nFolds, evaluation='AUC')
    _,__ = PlotValidationCurve(HPTuning,"RFC",'n_estimators',otherparams={'max_samples':0.9,'max_depth':6,'criterion':'entropy'},n_folds=nFolds, evaluation='AUC')
    _,__ = PlotValidationCurve(HPTuning,"RFC",'max_depth',otherparams={'n_estimators':20,'max_samples':0.9,'criterion':'entropy'},n_folds=nFolds, evaluation='AUC')


#%%
# ---------  Performs Reruns  --------- 


if nReruns != 1: 
    rerunResults = {'AUC':[], 'CutoffThreshold':[]}
    pipeRankings_df = ID_all.copy()
    predPerc_df     = ID_all.copy() 
    for i in range(nReruns):
         
        X_train,y_train, X_test,y_test, X_val,y_val, dataSplitName, testSize, trainSize, data_test, ID_train,ID_test,ID_val,ID_all, X_UTM_train,X_UTM_test,X_UTM_val = SplitAndClusterZone()
        X_train_scaled, X_test_scaled, X_val_scaled, nBrokenPipes,nUnbrokenPipes = Scale(X_train,y_train, X_test, X_val)
        X_train_relFeat, X_test_relFeat, X_val_relFeat, featuresToKeep_kBest,featuresToKeep_RFE,featuresToKeep_boruta,featureRankings,evaluationScore_Boruta,borutaScore = FeatureEngineering()
        X_train_DS, y_train_DS, ID_train_DS, X_UTM_train_DS = DownSampling(X_train_relFeat,y_train,downSampler,DSLevel)
        X_train_OS,y_train_OS, X_train_havari_DS,X_train_roer_DS, nUnbrokenDS, OSTimes, OSLevel, X_GAN = OverSampling(X_train_DS, y_train_DS,overSampler,desiredImbRatio0=desiredImbRatio)
        X_train_final,y_train_final, X_test_final,y_test_final, X_val_final, y_val_final, y_test_final_withID = DetermineFinalDataSets()

        y_pred, y_pred_perc, fittedClassifier,f1Score, accuracy, recall,precision, modelWith2Feat = TestClassificationModel(bestModel,X_train_final,y_train_final,X_test_final,y_test_final)
        y_pred_perc_df = pd.DataFrame({"Pred%":y_pred_perc})
        lengthCapture, faultCapture, AUC, LFC_sorted = CalcLFCaptureArea(pipeLength_total,y_test_final_withID,y_pred_perc_df)
        phi_bestPrecision, phi_50Precision,phi_bestF1, CMvalues_df, bestF1Score = DeterminePhiAt50Precision(y_pred_perc_df, y_test_final, plot = False)
        LFC_sorted = LFC_sorted[["ID","Pred%"]]
        LFC_sorted.columns = ["ID",'Pred%_{}'.format(i)]
        IDsRanked = LFC_sorted[["ID"]].reset_index().drop(['index'],axis=1)
        IDsRanked["Ranking_{}".format(i)] = IDsRanked.index.tolist()
        
        pipeRankings_df = pd.merge(pipeRankings_df,IDsRanked,how='left',on='ID')
        predPerc_df     = pd.merge(predPerc_df,LFC_sorted,how='left',on='ID')

        # Saves the results in a dic: 
        rerunResults["AUC"].append(AUC)
        rerunResults["CutoffThreshold"].append(phi_bestF1)
        if (i+1)%10 == 0:
            print(i+1)
        

    rerunResults_df = pd.DataFrame(rerunResults)

    # pipeRankings_df = pipeRankings_df.transpose(copy=True)
    # predPerc_df = predPerc_df.transpose(copy=True)
    rerunResults_df.to_excel("{}\\Broenshoej\\RerunResults\\AUCandCutoffThreshold_{}_{}.xlsx".format(resultsLoc,nReruns,overSampler))
    pipeRankings_df.to_excel("{}\\Broenshoej\\RerunResults\\PipeRankings_{}_{}.xlsx".format(resultsLoc,nReruns,overSampler))
    predPerc_df.to_excel("{}\\Broenshoej\\RerunResults\\PredPerc_{}_{}.xlsx".format(resultsLoc,nReruns,overSampler))


#  --------------  Using rerunDic  ---------------
rerunDic = {}
# rerunDic = {'overSampler':['ROS','Smote','ExpVSG'],
            # 'DSLevel':[12000,10000,8000,5000,3000,1000],
            # 'OSLevel':[100,250,500,750,1000]
            # }

if not rerunDic == {}: 
    rerunResults = {'OverSampler':[],'DS':[],'OS':[],'AUC':[],'CutoffThreshold':[],'F1':[]}
    # pipeRankings_df = ID_all.copy()
    # predPerc_df     = ID_all.copy() 
    i = 0
    for overSampler in rerunDic["overSampler"]:
        for DSLevel in rerunDic["DSLevel"]:
            for OSLevel in rerunDic["OSLevel"]:
                # X_train,y_train, X_test,y_test, X_val,y_val, dataSplitName, testSize, trainSize, data_test, ID_train,ID_test,ID_val,ID_all, X_UTM_train,X_UTM_test,X_UTM_val = SplitAndClusterZone()
                # X_train_scaled, X_test_scaled, X_val_scaled, nBrokenPipes,nUnbrokenPipes = Scale(X_train,y_train, X_test, X_val)
                # X_train_relFeat, X_test_relFeat, X_val_relFeat, featuresToKeep_kBest,featuresToKeep_RFE,featuresToKeep_boruta,featureRankings,evaluationScore_Boruta,borutaScore = FeatureEngineering()
                
                X_train_DS, y_train_DS, ID_train_DS, X_UTM_train_DS = DownSampling(X_train_relFeat,y_train,downSampler,DSLevel,KMeansPlot=False)
                X_train_OS,y_train_OS, X_train_havari_DS,X_train_roer_DS, nUnbrokenDS, OSTimes, OSLevel, X_GAN = OverSampling(X_train_DS, y_train_DS,overSampler,OSLevel0=OSLevel)
                desiredImbRatio = OSLevel/nUnbrokenDS 
                HPTuning, bestModel = HPTuner() 
                
                X_train_final,y_train_final, X_test_final,y_test_final, X_val_final, y_val_final, y_test_final_withID = DetermineFinalDataSets()

                y_pred, y_pred_perc, fittedClassifier,f1Score, accuracy, recall,precision, modelWith2Feat = TestClassificationModel(bestModel,X_train_final,y_train_final,X_test_final,y_test_final)
                y_pred_perc_df = pd.DataFrame({"Pred%":y_pred_perc})
                lengthCapture, faultCapture, AUC, LFC_sorted = CalcLFCaptureArea(pipeLength_total,y_test_final_withID,y_pred_perc_df)
                phi_bestPrecision, phi_50Precision,phi_bestF1, CMvalues_df, bestF1Score = DeterminePhiAt50Precision(y_pred_perc_df, y_test_final, plot = False)

                # Saves the results in a dic: 
                rerunResults["OverSampler"].append(overSampler)
                rerunResults["DS"].append(DSLevel)
                rerunResults["OS"].append(OSLevel) 
                rerunResults["AUC"].append(AUC)
                rerunResults["CutoffThreshold"].append(phi_bestF1)
                rerunResults["F1"].append(f1Score)
                if (i+1)%10 == 0:   
                    print(i+1)
                i += 1 
        
    rerunResults_df = pd.DataFrame(rerunResults)
    rerunResults_df.to_excel("{}\\Broenshoej\\Undersampling\\DSOS MatrixStudy.xlsx".format(resultsLoc))



#%%


def PlotResultDistribution(rerunResults_df): 

    AUC_df = rerunResults_df[["AUC"]]
    phi_values,phi_counts = 100*np.array(rerunResults_df["CutoffThreshold"].value_counts().keys()), rerunResults_df["CutoffThreshold"].value_counts().values
    # AUCstd = np.std(AUCList)
    
    def GetBins(list):
        listRange = max(list) - min(list)
        binWidth = listRange / 10
        bins = [min(list) + i * binWidth for i in range(11)]
        return bins 
    AUC_bins = GetBins(AUC_df["AUC"])
    print(np.std(AUC_df["AUC"]))
    print(np.mean(AUC_df["AUC"]))


    plt.figure(figsize = (8,6)) 
    plt.hist(AUC_df["AUC"], bins=AUC_bins, edgecolor='black')
    plt.xlabel('AUC')
    plt.ylabel('Observations')
    plt.title('AUC for {} runs'.format(nReruns))
    plt.savefig("{}\\Broenshoej\\RerunResults\\AUC_{}.png".format(resultsLoc,nReruns))

    plt.figure(figsize=(8,6))
    plt.bar(phi_values, phi_counts,width=0.5)
    plt.xlabel('phi')
    plt.ylabel('Observations')
    plt.title('phi for {} runs'.format(nReruns))
    plt.savefig("{}\\Broenshoej\\RerunResults\\phi_{}.png".format(resultsLoc,nReruns))

# PlotResultDistribution(rerunResults_df)

# TODO: Færdiggør Rerun Plots. 
# def PlotPipeRankingDistribution(pipeRankings_df): 
    

def PlotOSDistribution(smote, VSG, ROS): 

    AUC_df = rerunResults_df[["AUC"]]
    # AUCstd = np.std(AUCList)
    
    def GetBins(list):
        listRange = max(list) - min(list)
        binWidth = listRange / 10
        bins = [min(list) + i * binWidth for i in range(11)]
        return bins 
    bins = GetBins(smote["AUC"])
    # VSG_bins = GetBins(VSG["AUC"])
    # ROS_bins = GetBins(ROS["AUC"])

    plt.figure(figsize = (8,6)) 
    plt.hist(smote["AUC"], bins=bins, color=red, alpha=0.5,edgecolor='black')
    plt.hist(VSG["AUC"], bins=bins, color=green, alpha=0.5, edgecolor='black')
    plt.hist(ROS["AUC"], bins=bins, color=blue, alpha=0.5, edgecolor='black')
    plt.legend(['SMOTE','ExpVSG','ROS'])
    plt.xlabel('AUC')
    plt.ylabel('Observations')
    plt.title('Oversample Result Distribution')
    plt.savefig("{}\\Broenshoej\\Undersampling\\Rerun investigation\\AUC_Distribution{}.png".format(resultsLoc,nReruns))

    plt.figure(figsize = (8,6)) 
    plt.hist(smote["AUC"], bins=bins, color=red, alpha=0.5,edgecolor='black')
    plt.legend()
    plt.xlabel('AUC')
    plt.ylabel('Observations')
    plt.title('Oversample Result Distribution')
    plt.savefig("{}\\Broenshoej\\Undersampling\\Rerun investigation\\AUC_Distribution{}.png".format(resultsLoc,nReruns))

    plt.figure(figsize = (8,6)) 
    plt.hist(VSG["AUC"], bins=bins, color=green, alpha=0.5, edgecolor='black')
    plt.xlabel('AUC')
    plt.ylabel('Observations')
    plt.title('Oversample Result Distribution')
    plt.savefig("{}\\Broenshoej\\Undersampling\\Rerun investigation\\AUC_Distribution{}.png".format(resultsLoc,nReruns))

    plt.figure(figsize = (8,6)) 
    plt.hist(ROS["AUC"], bins=bins, color=blue, alpha=0.5, edgecolor='black')
    plt.xlabel('AUC')
    plt.ylabel('Observations')
    plt.title('Oversample Result Distribution')
    plt.savefig("{}\\Broenshoej\\Undersampling\\Rerun investigation\\AUC_Distribution{}.png".format(resultsLoc,nReruns))

smoteResults = pd.read_excel("{}\\Broenshoej\\Undersampling\\Rerun investigation\\AUCandCutoffThreshold_100_Smote.xlsx".format(resultsLoc))
VSGResults = pd.read_excel("{}\\Broenshoej\\Undersampling\\Rerun investigation\\AUCandCutoffThreshold_100_ExpVSG.xlsx".format(resultsLoc))
ROSResults = pd.read_excel("{}\\Broenshoej\\Undersampling\\Rerun investigation\\AUCandCutoffThreshold_100_ROS.xlsx".format(resultsLoc))


def PlotOSDistributionV2(smote, VSG, ROS):
    def GetBins(data):
        listRange = 0.7 - 0.5
        binWidth = listRange / 10
        bins = [0.5 + i * binWidth for i in range(11)]
        return bins

    bins = GetBins(smote["AUC"])

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axs[0].hist(ROS["AUC"], bins=bins, color=blue, alpha=0.6, edgecolor='black')
    # axs[0].set_ylabel('Observations')
    axs[0].set_title('ROS', fontsize=18)
    axs[0].set_ylim(0, 20)
    axs[0].axhline(y=10, color='lightgrey', linestyle='--')

    axs[1].hist(smote["AUC"], bins=bins, color=red, alpha=0.6, edgecolor='black')
    axs[1].set_ylabel('Observations')
    axs[1].set_title('SMOTE', fontsize=18)
    axs[1].set_ylim(0, 20)
    axs[1].axhline(y=10, color='lightgrey', linestyle='--')

    axs[2].hist(VSG["AUC"], bins=bins, color=green, alpha=0.6, edgecolor='black')
    # axs[2].set_ylabel('Observations')
    axs[2].set_title('Exponential VSG', fontsize=18)
    axs[2].set_ylim(0, 20)
    axs[2].set_xlabel('AUC')
    axs[2].axhline(y=10, color='lightgrey', linestyle='--')


    # fig.suptitle('Oversample Result Distribution', fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{resultsLoc}\\Broenshoej\\Undersampling\\Rerun investigation\\AUC_Distribution_OS_{nReruns}.png")
    plt.show()



PlotOSDistributionV2(smoteResults, VSGResults, ROSResults)




#%%
# Stops clock: 
runTime = round((time.perf_counter()-startTime)/60,1)
print("Total Runtime:{} min".format(runTime))


# %%






