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
GisLoc = "C:\\Users\\tvrj\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Arbejde\\Data\\HOFOR\\Data Brønshøj"
folderLoc = "C:\\Users\\tvrj\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Arbejde\\Data"
dataTotal_Brøns = pd.read_excel("{}\\HOFOR\\DataTotal_Brønshøj.xlsx".format(folderLoc))
havari_Brøns = dataTotal_Brøns[dataTotal_Brøns["FaultOccurance"] == 1]
rør_Brøns = dataTotal_Brøns[dataTotal_Brøns["FaultOccurance"] == 0]
havarier = pd.read_excel("{}\\HOFOR\\Data Brønshøj\\Havarier.xlsx".format(folderLoc))

#%% 
# ----------  Distance Matrix  -------------
# Laver Distance matricer for havari i Brønshøj:
distanceMatrix_Brøns = DistanceMatrix(havari_Brøns)
distanceMatrix_Brøns.to_csv("{}\\DistanceMatrix_Brøns.csv".format(GisLoc))


#%% 
# ----------  Adds length to data set  -------------
# (And saves necessary values for Fault Capture vs Length Capture ROC)
# Length data:
FaultToPipeDM = pd.read_excel("{}\\HOFOR\\Data Brønshøj\\Fault2PipeDistanceMatrix_Brøns.xlsx".format(folderLoc)).iloc[:,[1,0]]
pipeLength_rør_data = pd.read_excel("{}\\HOFOR\\Data Brønshøj\\Ledninger_Brønshøj_MedRørlængde.xlsx".format(folderLoc)).iloc[:,[0,-1]]
FaultToPipeDM.columns = ["ID_Rør","ID_Havari"]
pipeLength_rør_data.columns = ["ID_Rør","Length"]
 
pipeLength_rør = pipeLength_rør_data.copy()
LFCapture_df = FaultToPipeDM.merge(pipeLength_rør,how='left',left_on='ID_Rør', right_on='ID_Rør') # Adds length data to the DM
# LFCapture_df.columns = ["ID_Rør","ID_Havari","Length"]
pipeLength_havari = LFCapture_df.loc[:,["ID_Havari","Length"]].copy()
pipeLength_rør.columns = ["ID","Length"] 
pipeLength_havari.columns = ["ID","Length"] 
pipeLength_total = pd.concat([pipeLength_rør,pipeLength_havari],axis=0)

pipeLength_rør.to_excel("{}\\HOFOR\\Data Brønshøj\\PipeLengths_Rør.xlsx".format(folderLoc),index=False)
pipeLength_havari.to_excel("{}\\HOFOR\\Data Brønshøj\\PipeLengths_Havari.xlsx".format(folderLoc),index=False)
pipeLength_total.to_excel("{}\\HOFOR\\Data Brønshøj\\PipeLengths.xlsx".format(folderLoc),index=False)


# %%
# ---------- Adds joints to the data set  -------------
pipeVertexes = pd.read_excel("{}\\HOFOR\\Data Brønshøj\\PipeVertexes.xlsx".format(folderLoc))
vertexPipeOverlap_Data = pd.read_excel("{}\\HOFOR\\Data Brønshøj\\VertexPipeOverlap 2mm.xlsx".format(folderLoc))
vertexVertexOverlap_Data = pd.read_excel("{}\\HOFOR\\Data Brønshøj\\VertexVertexOverlap 2mm.xlsx".format(folderLoc)) 

# nJoints_Turns = AddsJointsIfPipeTurns(pipeVertexes) # Calculates the number of joints added to a pipe by the overall angle change in the pipe
nJoints_Turns = AddsJointsIfSubpipeTurns(pipeVertexes, angleOfTurn=85) # Calculates the number of joints added to a pipe by the angle difference between each subpipe (Lasse's method)
nJoints_Length = AddsJointsBasedOnLength(pipeLength_rør,nJoints_Turns)
nJoints_rør = AddJointsBetweenPipes(vertexPipeOverlap_Data,vertexVertexOverlap_Data, nJoints_Length.copy())

print("Total number of joints after turns: \t\t{}".format(nJoints_Turns.nJoints.sum()))
print("Total number of joints after length: \t\t{}".format(int(nJoints_Length.nJoints.sum())))
print("Total number of joints after intersections: \t{}".format(int(nJoints_rør.nJoints.sum())))

# Adds joints to the havari data: 
nJointsDM = FaultToPipeDM.merge(nJoints_rør,how='left',left_on='ID_Rør', right_on='ID') # Adds Joints to DM
# nJointsDM.columns = ["ID_Rør","ID_Havari","nJoints"]
nJoints_havari = nJointsDM.loc[:,["ID_Havari","nJoints"]].copy()
nJoints_rør.columns = ["ID","nJoints"] 
nJoints_havari.columns = ["ID","nJoints"] 

nJoints_Turns.to_excel("{}\\HOFOR\\Data Brønshøj\\Joints_SubpipeTurns.xlsx".format(folderLoc),index=False)
nJoints_Length.to_excel("{}\\HOFOR\\Data Brønshøj\\Joints_SubpipeTurns+Length.xlsx".format(folderLoc),index=False)
nJoints_rør.to_excel("{}\\HOFOR\\Data Brønshøj\\Joints_Subpipe_Rør.xlsx".format(folderLoc),index=False)
nJoints_havari.to_excel("{}\\HOFOR\\Data Brønshøj\\Joints_Subpipe_Havari.xlsx".format(folderLoc),index=False)






# %%
