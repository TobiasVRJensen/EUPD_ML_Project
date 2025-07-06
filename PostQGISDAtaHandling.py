"""The goal of this file is to handle and manipulate the QGIS data and prepare it for using it in wither HOFOR_Data.py or 
RunModel.py"""

#%% 
# ---------  Loading packages  ---------
import numpy as np 
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from MLFunctions.utilities import * 
from MLFunctions.data_cleaning import *

resultsLoc= "C:\\Users\\tvrj\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Arbejde\\Machine Learning\\Results"

#%% 
# ---------  Loading Generally relevant data  ---------
GisLoc = "C:\\Users\\tvrj\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Arbejde\\Data\\HOFOR\\Data Broenshoej"
folderLoc = "C:\\Users\\tvrj\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Arbejde\\Data"
dataTotal_Broens = pd.read_excel("{}\\HOFOR\\DataTotal_Broenshoej.xlsx".format(folderLoc))
havari_Broens = dataTotal_Broens[dataTotal_Broens["FaultOccurance"] == 1]
roer_Broens = dataTotal_Broens[dataTotal_Broens["FaultOccurance"] == 0]
havarier = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\Havarier.xlsx".format(folderLoc))

#%% 
# ----------  Distance Matrix  -------------
# Laver Distance matricer for havari i Broenshoej:
distanceMatrix_Broens = DistanceMatrix(havari_Broens)
distanceMatrix_Broens.to_csv("{}\\DistanceMatrix_Broens.csv".format(GisLoc))


#%% 
# ----------  Adds length to data set  -------------
# (And saves necessary values for Fault Capture vs Length Capture ROC)
# Length data:
FaultToPipeDM = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\Fault2PipeDistanceMatrix_Brøns.xlsx".format(folderLoc)).iloc[:,[1,0]]
pipeLength_roer_data = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\Ledninger_Brønshøj_MedRørlængde.xlsx".format(folderLoc)).iloc[:,[0,-1]]
FaultToPipeDM.columns = ["ID_Roer","ID_Havari"]
pipeLength_roer_data.columns = ["ID_Roer","Length"]
 
pipeLength_roer = pipeLength_roer_data.copy()
LFCapture_df = FaultToPipeDM.merge(pipeLength_roer,how='left',left_on='ID_Roer', right_on='ID_Roer') # Adds length data to the DM
# LFCapture_df.columns = ["ID_Roer","ID_Havari","Length"]
pipeLength_havari = LFCapture_df.loc[:,["ID_Havari","Length"]].copy()
pipeLength_roer.columns = ["ID","Length"] 
pipeLength_havari.columns = ["ID","Length"] 
pipeLength_total = pd.concat([pipeLength_roer,pipeLength_havari],axis=0)

pipeLength_roer.to_excel("{}\\HOFOR\\Data Broenshoej\\PipeLengths_Rør.xlsx".format(folderLoc),index=False)
pipeLength_havari.to_excel("{}\\HOFOR\\Data Broenshoej\\PipeLengths_Havari.xlsx".format(folderLoc),index=False)
pipeLength_total.to_excel("{}\\HOFOR\\Data Broenshoej\\PipeLengths.xlsx".format(folderLoc),index=False)


# %%
# ---------- Adds joints to the data set  -------------
pipeVertexes = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\PipeVertexes.xlsx".format(folderLoc))
vertexPipeOverlap_Data = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\VertexPipeOverlap 2mm.xlsx".format(folderLoc))
vertexVertexOverlap_Data = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\VertexVertexOverlap 2mm.xlsx".format(folderLoc)) 

# nJoints_Turns = AddsJointsIfPipeTurns(pipeVertexes) # Calculates the number of joints added to a pipe by the overall angle change in the pipe
nJoints_Turns = AddsJointsIfSubpipeTurns(pipeVertexes, angleOfTurn=85) # Calculates the number of joints added to a pipe by the angle difference between each subpipe (Lasse's method)
nJoints_Length = AddsJointsBasedOnLength(pipeLength_roer,nJoints_Turns)
nJoints_roer = AddJointsBetweenPipes(vertexPipeOverlap_Data,vertexVertexOverlap_Data, nJoints_Length.copy())

print("Total number of joints after turns: \t\t{}".format(nJoints_Turns.nJoints.sum()))
print("Total number of joints after length: \t\t{}".format(int(nJoints_Length.nJoints.sum())))
print("Total number of joints after intersections: \t{}".format(int(nJoints_roer.nJoints.sum())))

# Adds joints to the havari data: 
nJointsDM = FaultToPipeDM.merge(nJoints_roer,how='left',left_on='ID_Roer', right_on='ID') # Adds Joints to DM
# nJointsDM.columns = ["ID_Roer","ID_Havari","nJoints"]
nJoints_havari = nJointsDM.loc[:,["ID_Havari","nJoints"]].copy()
nJoints_roer.columns = ["ID","nJoints"] 
nJoints_havari.columns = ["ID","nJoints"] 

nJoints_Turns.to_excel("{}\\HOFOR\\Data Broenshoej\\Joints_SubpipeTurns.xlsx".format(folderLoc),index=False)
nJoints_Length.to_excel("{}\\HOFOR\\Data Broenshoej\\Joints_SubpipeTurns+Length.xlsx".format(folderLoc),index=False)
nJoints_roer.to_excel("{}\\HOFOR\\Data Broenshoej\\Joints_Subpipe_Roer.xlsx".format(folderLoc),index=False)
nJoints_havari.to_excel("{}\\HOFOR\\Data Broenshoej\\Joints_Subpipe_Havari.xlsx".format(folderLoc),index=False)

#%% 
# ------  Investisgates the best Fault Clustering HPs  -----------

pipeLengths_rør = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\PipeLengths_Rør.xlsx".format(folderLoc))
pipeLengths_havari = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\PipeLengths_Havari.xlsx".format(folderLoc))

timeWise = True
if timeWise: 
    dataTotal_train_Broens = dataTotal_Broens[dataTotal_Broens['Havari Year'] <=2016.0]
    fileName = 'HPTuningOfCZ_TimeWise'
else: 
    dataTotal_train_Broens = dataTotal_Broens.copy()
    fileName = 'HPTuningOfCZ'
pipeLengths_rør["FaultOccurance"] = 0
pipeLengths_havari["FaultOccurance"] = 1
pipeLengths_withFO = pd.concat([pipeLengths_havari,pipeLengths_rør],axis=0,ignore_index=True)
dataForFC = pd.merge(pipeLengths_withFO,dataTotal_train_Broens[['ID','X_UTM','Y_UTM']],on='ID',how='right')
havariDataForFC = dataForFC[dataForFC['FaultOccurance'] == 1]


nFaults = dataForFC["FaultOccurance"].sum()
networkLength = dataForFC["Length"].sum()

epsList = [20,30,40,50,60,70,80,90,100,120,140,160,180, 200, 300, 400, 500, 600, 800, 1000] # The Radii investigated in meters
minSamplesList = [3,4,5,7,9,11] # THe minimum number of samples investigated 
resultsFC_df = pd.DataFrame(columns=["eps",'Min_Samples','Length%', 'Fault%'])

for eps in epsList: 
    for minSamples in minSamplesList: 
        dataWithFC,corePoints = CheckIfPointInClusterZone(dataForFC, havariDataForFC, eps=eps, min_samples=minSamples)
        FCLength = dataWithFC[dataWithFC["FaultClustering"] == 1]['Length'].sum()
        FCFaults = dataWithFC[dataWithFC["FaultClustering"] == 1]['FaultOccurance'].sum()
        thisResult_df = pd.DataFrame(data = {'eps':[eps],'Min_Samples':[minSamples],'Length%':[FCLength/networkLength * 100],'Fault%':[FCFaults/nFaults * 100]})
        resultsFC_df = pd.concat([resultsFC_df,thisResult_df],axis=0,ignore_index=True)

resultsFC_df.to_excel("{}\\Broenshoej\\Cluster Zones\\{}.xlsx".format(resultsLoc,fileName))

resultsFC_df_forPlotting = resultsFC_df[resultsFC_df['Length%'] > 0.01]
resultsFC_df_forPlotting = resultsFC_df_forPlotting[resultsFC_df_forPlotting['Length%'] < 50]
resultsFC_df_forPlotting = resultsFC_df_forPlotting[resultsFC_df_forPlotting['Fault%']  > 10]

plt.figure(figsize=(8,6)) 
indexes = resultsFC_df_forPlotting.index
plt.plot(indexes,resultsFC_df_forPlotting["Length%"],c='blue')
plt.plot(indexes,resultsFC_df_forPlotting["Fault%"],c='green')
plt.plot(indexes,resultsFC_df_forPlotting["Fault%"]/resultsFC_df_forPlotting["Length%"],c='orange')
plt.savefig("{}\\Broenshoej\\Cluster Zones\\HPTuningOfCZ.PNG".format(resultsLoc))
plt.legend(["Length%","Fault%",'Fault%/Length%'])
plt.grid()
plt.ylim([0,100])

# resultsFC_df = resultsFC_df[resultsFC_df['Length%'] < 0.01 or resultsFC_df['Length%'] > 99 or resultsFC_df['Fault%'] < 0.01 or resultsFC_df['Fault%'] > 99]
print(resultsFC_df_forPlotting)


# %%

red = '#990000'
blue = '#0070C0'

# def PlotAgeDistribution(X0):
X0 = dataTotal_Broens
"""Makes a bar plot of the number of broken pipes as a function of the agegroup""" 
havarier = X0[X0['FaultOccurance'] == 1]
roer = X0[X0['FaultOccurance'] == 0]

havariCounts = havarier["Age"].value_counts()
havariÅr, havariAntal = havariCounts.keys(), havariCounts.values
roerCounts = roer["Age"].value_counts()
roerÅr, roerAntal = roerCounts.keys(), roerCounts.values

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot broken pipes on primary y-axis
bar1 = ax1.bar(np.array(havariÅr) - 0.2, havariAntal, width=0.4, color=red, label='Broken pipes')
ax1.set_ylabel('Broken pipes [count]', color=red)
ax1.tick_params(axis='y', labelcolor=red)

# Create secondary y-axis
ax2 = ax1.twinx()
bar2 = ax2.bar(np.array(roerÅr) + 0.2, roerAntal, width=0.4, color=blue, label='Non-broken pipes')
ax2.set_ylabel('Non-broken pipes [count]', color=blue)
ax2.tick_params(axis='y', labelcolor=blue)

# X-axis label and title
ax1.set_xlabel('Age')

plt.title('Age Distribution of Pipes')

plt.savefig("{}\\Broenshoej\\AgeDistribution.png".format(resultsLoc))

# PlotAgeDistribution(dataTotal_Broens)




# %%
