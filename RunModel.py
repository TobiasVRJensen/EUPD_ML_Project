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
plt.rcParams.update({'font.size': 18})
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


#%% 
# ---------  Loads data  --------- 
folderLoc = "C:\\Users\\tvrj\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Arbejde\\Data"
resultsLoc= "C:\\Users\\tvrj\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Arbejde\\Machine Learning\\Results"
dataTotal = pd.read_excel("{}\\HOFOR\\DataTotal_Broenshoej.xlsx".format(folderLoc))
# HPTuningData = pd.read_excel("{}\\Broenshoej\\k-CrossValidation\\{}.xlsx".format(resultsLoc,"HyperParameterTuning - Total v2.0 - ROS")) # Loads the Hyper parameter tuning results 
pipeLength_total = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\PipeLengths.xlsx".format(folderLoc))
header = dataTotal.columns


#%% 
# ---------  Defines the model parameters  --------- 
# Defines which parts of the model should be run and with which parameters 
targetName = "FaultOccurance" #"FaultOccurance" or "Age". "nFaults" Is no  longer supported (Multiple functions need to be updated for this to work)
# combinePipes = True # Whether to combine the Havari 
splitDataByTime = True
doFeatureEngineering = False
showBorutaScore = True # Indicates whether the Boruta Score of the model should be saved in a file. This is to only overwrite the file when wanted
downSampler = "KMean" # 'KMean' or 'RP' 
DSLevel = 10000
overSampler = "ROS" # 'ROS' or 'ExpVSG' or 'GAN' 
runGAN = False
desiredImbRatio = 1/10 # Describes the ratio between unbroken and broken pipes after DS and OS. Used to define OSLevel
optimizeHyperParameters = True
fastHPTuning = True
updateHpOfBestModel = 'AUC' # False # Defines which evaluator to optimize the HPs for. 
featuresToKeep = ['InnerDia', 'AgeGroup', 'Rain', "Length",'Anae_Depth','FaultClustering','SoilCG_Zink','GWCG_Cya', 'GWCG_Lead','GWCG_Det','GWCG_Pest', 'InnerPipeMate_Steel','Pressure']#, 'nJoints']#,'SoilType_DS - Smeltevandssand','SoilType_FT - Ferskvandstoerv','SoilType_ML - Moraeneler', 'LandUse_Park etc.', 'LandUse_allotments','LandUse_residential','RoadType']
HPTuningName = 'withPressure' 
eps_CZ = 30
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

    # Definerer hvilket kolonner der skal laves OneHot paa  (Det er ikke noedvendigvis alle kolonner), og foretager OneHot
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

if splitDataByTime: 
    """For now this is only compatible with the classification model. """
    dataSplitName = 'Time'
    ### Data splitting 
    # Splits the havari data by year:
    X_train_havari, y_train_havari, X_test_havari, y_test_havari, testSize = SplitHavariDataByYear(X_havari,y_havari,desiredTestSize)
    trainSize = 1-testSize
    # Splits the unBroken data by quantity: 
    X_train_unB, X_test_unB, y_train_unB, y_test_unB = train_test_split(X_unB,y_unB,test_size=testSize,random_state=randState,stratify=X_unB['Age'])
    # Recombines the havari and unB data: 
    X_train,y_train = pd.concat([X_train_havari,X_train_unB],axis=0,ignore_index=True) , pd.concat([y_train_havari,y_train_unB],axis=0,ignore_index=True)
    X_test,y_test = pd.concat([X_test_havari,X_test_unB],axis=0,ignore_index=True) , pd.concat([y_test_havari,y_test_unB],axis=0,ignore_index=True)
    # Removes the 'Age' columns from the data sets as they are not relevant any more: 
    X_train, X_test = X_train.drop(['Age'],axis=1) , X_test.drop(['Age'],axis=1)
    # Splits the test data set into vallidation and test data set: 
    X_val, X_test, y_val, y_test = train_test_split(X_test,y_test,test_size=0.5,random_state=randState,stratify=y_test[targetName])

else: 
    dataSplitName = 'Quantity'
    ### Fault Clustering based on all data points: 
    # X_withCZ, corePoints = CheckIfPointInClusterZone(X_manipulated,X_havari,eps=30,min_samples=3,plot=True)
    
    ### Data splitting: 
    # Resets the testSize as this will not change if the data is not split by time: 
    testSize = desiredTestSize
    # Splits the data into teseting and training by quantity: 
    if modelType == "Classification" or targetName == 'nFaults': 
        if targetName == "FaultOccurance":
            X_manipulated_noAge = X_manipulated.copy().drop("Age",axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X_manipulated_noAge,y,test_size=testSize,random_state=randState,stratify=y[targetName])
    elif modelType == "Regression": 
        X_train, X_test, y_train, y_test = train_test_split(X_manipulated,y,test_size=testSize,random_state=randState)
    X_train_havari = X_train[pd.concat([X_train,y_train],axis=1)["FaultOccurance"] == 1]

X_train,y_train = X_train.reset_index(drop=True),y_train.reset_index(drop=True) # Resets the index after the train-Test-split shuffled them
X_test,y_test = X_test.reset_index(drop=True),y_test.reset_index(drop=True) # Resets the index after the train-Test-split shuffled them
if splitDataByTime: 
    X_val,y_val = X_val.reset_index(drop=True),y_val.reset_index(drop=True) # Resets the index after the train-Test-split shuffled them

# PlotHavariYearDistribution(pd.DataFrame(X_train['Havari Year'].copy()))

### Fault Clustering based on the Training data set only: 
X_train, corePoints =  CheckIfPointInClusterZone(X_train.copy(), X_train_havari, eps=eps_CZ,min_samples=min_samples_CZ)
X_test, corePoints  =  CheckIfPointInClusterZone(X_test.copy(), X_train_havari, eps=eps_CZ,min_samples=min_samples_CZ)
if splitDataByTime: 
    X_val, corePoints  =  CheckIfPointInClusterZone(X_val.copy(), X_train_havari, eps=eps_CZ,min_samples=min_samples_CZ)

# Her er indexerne blevet shufflet en sidste gang, saa nu kan jeg seperere ID'erne og koordinaterne fra dem saa jeg altid har hvilket ID der hoerer til hvilket roer/havari-roer ud fra indekset: 
ID_train, ID_test = X_train['ID'].copy(),X_test['ID'].copy()
X_UTM_train, X_UTM_test = X_train[['X_UTM','Y_UTM']].copy(),X_test[['X_UTM','Y_UTM']].copy()
X_train, X_test = X_train.drop(["ID","X_UTM","Y_UTM","Havari Year"],axis=1) , X_test.drop(["ID","X_UTM","Y_UTM","Havari Year"],axis=1) # Removes some columns, as their purpose over. 
if splitDataByTime: 
    ID_val  = X_val['ID'].copy()
    X_UTM_val = X_val[['X_UTM','Y_UTM']].copy()
    X_val = X_val.drop(["ID","X_UTM","Y_UTM","Havari Year"],axis=1)

# Samler testdataen, da det er relevant til regressionsmodellen. 
data_test = pd.concat([X_test,y_test],axis=1)

# Skalerer alle tal i X med en standard scaler eller minMax scaler hvis de er blevet OneHOt Encoded (Except Columns describing IDs or x- or y-coordinates)
X_train1,X_test1 = ScaleX(X_train.drop(OneHotXColumns,axis=1)), ScaleX(X_test.drop(OneHotXColumns,axis=1))
X_train2,X_test2 = ScaleX(X_train[OneHotXColumns], scalingtype="Normalized"), ScaleX(X_test[OneHotXColumns], scalingtype="Normalized") # Rescales values that should be between 0-1
X_train_scaled,X_test_scaled = pd.concat([X_train1,X_train2],axis=1), pd.concat([X_test1,X_test2],axis=1)

X_train_havari_scaled = X_train_scaled[y_train[targetName] == 1]
nBrokenPipes = X_train_havari_scaled.shape[0]
nUnbrokenPipes = X_train_scaled.shape[0] - nBrokenPipes
if splitDataByTime: 
    X_val1 = ScaleX(X_val.drop(OneHotXColumns,axis=1))
    X_val2 = ScaleX(X_val[OneHotXColumns], scalingtype="Normalized") # Rescales values that should be between 0-1
    X_val_scaled= pd.concat([X_val1,X_val2],axis=1)


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
        RFC  =  RandomForestClassifier(n_estimators=4,max_depth=4,max_samples=0.8,warm_start=True,criterion='log_loss', oob_score=False)
    elif downSampler == "RP":
        RFC =  RandomForestClassifier(n_estimators=4,max_depth=4,max_samples=0.7,warm_start=True,criterion='entropy', oob_score=False)
elif overSampler == "ROS": 
    if downSampler == "KMean": 
        # RFC  =  RandomForestClassifier(n_estimators=10,max_depth=5,max_samples=0.6,warm_start=True,criterion='log_loss', oob_score=False)
        RFC  =  RandomForestClassifier(n_estimators=8,max_depth=7,max_samples=0.5,warm_start=True,criterion='entropy', oob_score=False)
    elif downSampler == "RP": 
        RFC =  RandomForestClassifier(n_estimators=4,max_depth=4,max_samples=0.8,warm_start=True,criterion='entropy', oob_score=False)
elif overSampler == "ExpVSG": 
    if downSampler == "KMean": 
        RFC  =  RandomForestClassifier(n_estimators=8,max_depth=8,max_samples=0.6,warm_start=True,criterion='entropy', oob_score=False)
    elif downSampler == "RP": 
        RFC =  RandomForestClassifier(n_estimators=4,max_depth=4,max_samples=0.7,warm_start=True,criterion='log_loss', oob_score=False)

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
if doFeatureEngineering: 
    ite = '2.0' 
    if modelType == "Classification":
        featuresToKeep_kBest = FeatureEngineering_kBest(X_train_scaled,y_train,RFC,estimatorName="RFC{}".format(ite))
        print("kBest:",featuresToKeep_kBest)
        featuresToKeep_RFE = FeatureEngineering_RFE(X_train_scaled,y_train,RFC,estimatorName="RFC{}".format(ite))
        print("RFE:",featuresToKeep_RFE)
    bestModel_coldStart = clone(bestModel)
    setattr(bestModel_coldStart,'warm_start',False) # Brouta does not work when 'warm_start' is True, thus it is changed to False
    featuresToKeep_boruta,featureRankings, evaluationScore_Boruta, borutaScore = FeatureEngineering_Boruta(X_train_scaled,y_train,bestModel_coldStart,estimatorName="RFC_Cold{}".format(ite))
    print("Boruta:",featuresToKeep_boruta,evaluationScore_Boruta)
    if showBorutaScore: 
        dfBorutaScores = pd.DataFrame(borutaScore)
        dfBorutaScores.columns = X_train.columns
        dfBorutaScores = add_mean_row(dfBorutaScores)
        dfBorutaScores.to_excel("{}\\Broenshoej\\FeatureImportance\\Boruta\\BorutaScores.xlsx".format(resultsLoc),index=False)

# We only want to use the relevant features in the model:
X_train_relFeat,X_test_relFeat = X_train_scaled[featuresToKeep],X_test_scaled[featuresToKeep]
if splitDataByTime: 
    X_val_relFeat = X_val_scaled[featuresToKeep]

#%% 
# ---------  Downsampling  ---------
if downSampler == 'RP':
    X_train_DS, y_train_DS = CalcEucledianDistances(X_train_relFeat,y_train,Plot=False,DSLevel=DSLevel) # Undersamples X_train and y_train depending on the Euclidean distance to some reference points 
elif downSampler == 'KMean':
    X_train_DS, y_train_DS = kMeansClusteringDS(X_train_relFeat,y_train,n_clusters=DSLevel,max_iter=300,Plot=True,targetname=targetName) 
else: 
    raise ValueError("The down sampling method is incorrect. '{}' was given.".format(downSampler))

# Changes the ID df to only have the downsampled ID's 
ID_train_DS = ID_train[X_train_DS.index]
X_UTM_train_DS = X_UTM_train.iloc[X_train_DS.index]


#%% 
# ---------  Oversampling  ---------
# Oversampler de underrepraesenterede data: 
X_train_havari_DS = X_train_DS[y_train_DS[targetName] == minorityValue]
X_train_roer_DS    = X_train_DS[y_train_DS[targetName] == 0]
nUnbrokenDS = X_train_roer_DS.shape[0] # number of unbroken pipes present in the data set after DS. 
OSTimes = int((nUnbrokenDS*desiredImbRatio-nBrokenPipes)/nBrokenPipes) # Defines the number of nBroken pipes skal virtuelt genereres. 1 means the same number as the real samples, 2 means twice as many etc. Calculated from ImbRatio
OSLevel = nBrokenPipes*OSTimes  # Defines the number of virtual samles present after OS 

if runGAN:
    X_GAN = RunGAN(X_train_havari_DS,overSampleLevel=10,Plot=True,nEpochs=15000,saveName="x10,15000")
else: 
    X_GAN = pd.read_excel("{}\\Broenshoej\\GAN\\fakeSamples_Classx10,15000.xlsx".format(resultsLoc))
    X_GAN = X_GAN[:OSLevel]
print("Number of majoritysamples: {}".format(y_train_DS[y_train_DS[targetName] == majorityValue].shape[0]))


X_train_OS,y_train_OS = DoOverSampling(X_train_DS,y_train_DS,targetName,overSampler,desiredImbRatio,OSTimes,modelType,minorityValue,majorityValue,X_GAN,randstate=randState)
print("{} oversampling performed: From {} to {} data points".format(overSampler,y_train_DS[y_train_DS[targetName] != majorityValue].shape[0],y_train_OS[y_train_OS[targetName] != majorityValue].shape[0]))
print("Imbalance Ratio:{}".format(round(np.sum(y_train_OS[targetName] == 1)/np.sum(y_train_OS[targetName] == 0),2)))

#%% 

# ---------  Hyper Parameter Tuning  ---------
""" Only the DS data is used to HP tune as the OS should be done after the val-train split in the k-CV. 
The values deciding the level of OS are used to make the OS identical to the one used in the remainder of this script. """
if fastHPTuning:
    HPTypeStr = "Fast"
else: 
    HPTypeStr = "Slow"
saveName_HPTuning="HPTuning{}_{},{}_{}={},{}={}_{}".format(HPTypeStr,dataSplitName,targetName[0:3],downSampler,DSLevel,overSampler,OSLevel,HPTuningName)
if optimizeHyperParameters:  
    X_train_forHPTuning = pd.concat([X_train_DS,X_UTM_train_DS],axis=1) # Adds the coordinates to the training and testing data to be used on the fault Clustering in the k-CV
    X_test_forHPTuning  = pd.concat([X_test_relFeat,X_UTM_test],axis=1)
    from MLFunctions.HyperParameterTuning import modelClassifierDic, modelRegressorDic, modelClassifierDic_fast, modelRegressorDic_fast
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
        bestModel = clone(bestModel_forAUC)
    elif updateHpOfBestModel == 'Recall':
        bestModel = clone(bestModel_forRecall)
    elif updateHpOfBestModel == 'Precision':
        bestModel = clone(bestModel_forPrecision)
    elif updateHpOfBestModel == 'Overall':
        bestModel = clone(bestModel_overall)
    else: 
        print("--- No Desired best model parameter inputted. AUC is used. ---")
        bestModel = clone(bestModel_forAUC)
    


#%% 
# ---------  Plots Results of 'bestModel'  ---------
X_train_final, y_train_final = X_train_OS,y_train_OS
X_test_final, y_test_final = X_test_relFeat, y_test
if splitDataByTime: 
    X_val_final, y_val_final = X_val_relFeat, y_val

if modelType == "Classification":
    # Vaelger hvilke features der skal plottes efter senere: 
    feat1Index = 2
    feat2Index = 3
    feat1Name = featureNames_Redefined[feat1Index]
    feat2Name = featureNames_Redefined[feat2Index]
    feat1Name = "InnerDia"
    feat2Name = "AgeGroup"
    if featureNotInModel == feat1Name:   
        feat1Name = "Rain"    
    elif featureNotInModel == feat2Name: 
        feat2Name = "Rain" 
    # Defines the hyper parameter's of the best model:
    HPStr = FindChangedHPs(bestModel)
    print("Model used: {}".format(bestModel))
    print("Used HPs: {}".format(HPStr))
    saveName = "{},{},{},{},{},{}={},{}={},{}".format(modelType[0:3],trainSize,randState,nFolds,featureNotInModel,downSampler,DSLevel,overSampler,OSLevel,HPStr)

    y_pred, y_pred_perc, fittedClassifier,f1Score, accuracy, recall,precision, modelWith2Feat = TestClassificationModel(bestModel,X_train_final,y_train_final,X_test_final,y_test_final, feat1Name,feat2Name)
    # Second part of the feature Engineering (That has to be done after the simulation is finished. )
    if doFeatureEngineering and isinstance(fittedClassifier, RandomForestClassifier):
        featuresToKeep_RF,scoreList_RF = FeatureEngineering_RF(X_train_final.columns,fittedClassifier)
        print("Features to keep according to RF: {}".format(list(featuresToKeep_RF)))
        print("Feature scores from RF:\t\t  {}".format(np.round(np.array(scoreList_RF),2)))
    
    ### Visualisering af diverse resultater 
    # Confusion Matrix:
    ConfusionMatrix(y_test_final,y_pred, savename = saveName,title="Recall",Normalise=None)
    
    # ROC curves: 
    PlotROC(y_test_final, y_pred_perc,savename=saveName,bestmodelname=bestModelName)
    PlotPRCurve(y_test_final, y_pred_perc,savename=saveName,bestmodelname=bestModelName)
    y_test_final_withID = pd.concat([ID_test,y_test_final],axis=1)
    y_pred_perc_df = pd.DataFrame({"Pred%":y_pred_perc})
    PlotLvsFCapture(pipeLength_total,y_test_final_withID,y_pred_perc_df, savename=saveName)

    # aendrer cut-off threshold (phi), og plotter CM igen. 
    phi_bestPrecision, phi_50Precision,phi_bestF1, CMvalues_df = DeterminePhiAt50Precision(y_pred_perc_df, y_test_final, plot = True)
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

# Validation curve: 
# PlotValidationCurve(HPTuningData,"LR","C",otherparams={'solver':'lbfgs'},n_folds=nFolds)
# PlotValidationCurve(HPTuningData,"XGBC","max_depth",otherparams={'n_estimators':5,'learning_rate':0.7,"eval_metric":"<function recall_score at 0x000001F73327C2C0>","booster":"gbtree"},n_folds=nFolds)
# PlotValidationCurve(HPTuningData,"RFC","max_depth",otherparams={'max_samples':0.5,'n_estimators':5,"warm_start":False,"criterion":"log_loss"},n_folds=nFolds)
#%%

# Stops clock: 
runTime = round((time.perf_counter()-startTime)/60,1)
print("Total Runtime:{} min".format(runTime))


# %%






