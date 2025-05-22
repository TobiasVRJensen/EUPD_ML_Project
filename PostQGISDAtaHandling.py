"""The goal of this file is to handle and manipulate the QGIS data and prepare it for using it in wither HOFOR_Data.py or 
RunModel.py"""

#%% 
# ---------  Loading packages  ---------
import numpy as np 
import pandas as pd
from MLFunctions.utilities import * 
from MLFunctions.data_cleaning import *
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






# %%
