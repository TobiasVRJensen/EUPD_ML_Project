""" The goal of this file is to manipulate the Hofor data to only contain the relevant columns. The resulting data files will be saved as excel files."""
 
#%% 
# -------------  Impmorting packages and functions  -------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import sys
sys.path.append('C:\\Users\\tvrj\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Arbejde\\Machine Learning')
from MLFunctions.utilities import * 
from MLFunctions.data_cleaning import *
from MLFunctions.oversamplers import * 
from MLFunctions.models import * 
from MLFunctions.visualization import * 


#%% 
# -----------  Loading data  -------------
folderLoc = "C:\\Users\\tvrj\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Arbejde\\Data"
data = {}
# data["Afgrening"] = pd.read_excel("{}\\HOFOR\\EUDP_Afgrening.xlsx".format(folderLoc), sheet_name="EUDP Afgrening")
# data["ArmLure"] = pd.read_excel("{}\\HOFOR\\EUDP_Arm_lure.xlsx".format(folderLoc), sheet_name="EUDP Arm lure")
# data["Bøjning"] = pd.read_excel("{}\\HOFOR\\EUDP_Bøjning.xlsx".format(folderLoc), sheet_name="EUDP Bøjning")
# data["FastspændingsKlods"] = pd.read_excel("{}\\HOFOR\\EUDP_Fastspændings-klods.xlsx".format(folderLoc), sheet_name="EUDP Fastspændings-klods")
# data["FastspændingsPunkt"] = pd.read_excel("{}\\HOFOR\\EUDP_Fastspændings-punkt.xlsx".format(folderLoc), sheet_name="EUDP Fastspændings-punkt")
# data["Hane"] = pd.read_excel("{}\\HOFOR\\EUDP_Hane.xlsx".format(folderLoc), sheet_name="EUDP Hane")
data["HavariNuværendeNet_KBH"] = pd.read_excel("{}\\HOFOR\\EUDP_Havari-nuværende-net.xlsx".format(folderLoc), sheet_name="EUDP Havari-nuværende-net")
# data["KammerVV"] = pd.read_excel("{}\\HOFOR\\EUDP_Kammer_-_VV_-_udført_aktiv.xlsx".format(folderLoc), sheet_name="EUDP Kammer - VV - Udført Ak")
# data["Kompensator"] = pd.read_excel("{}\\HOFOR\\EUDP_Kompensator.xlsx".format(folderLoc), sheet_name="EUDP Kompensator")
data["LedningVF_KBH"] = pd.read_excel("{}\\HOFOR\\EUDP_Ledning-VF-Aktiv.xlsx".format(folderLoc), sheet_name="EUDP Ledning-VF-Aktiv")
# data["Reduktion"] = pd.read_excel("{}\\HOFOR\\EUDP_Reduktion.xlsx".format(folderLoc), sheet_name="EUDP Reduktion")
# data["Samling"] = pd.read_excel("{}\\HOFOR\\EUDP_Samling.xlsx".format(folderLoc), sheet_name="EUDP Samling")
# data["TunnelAlleHofor"] = pd.read_excel("{}\\HOFOR\\EUDP_Tunnel-alle-HOFOR.xlsx".format(folderLoc), sheet_name="EUDP Tunnel-alle-HOFOR")
data["Ledninger_Brøns"] = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\Ledninger og havarier fra KOX.xlsx".format(folderLoc))
data["Havarier_Brøns"] = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\Havarier.xlsx".format(folderLoc))

dataKeys = data.keys()

# GISdata: 
GisLoc = "{}\\HOFOR\\Data Broenshoej\\Raster Data".format(folderLoc)
# Assuming the relevant column is the last column and the first column is an ID. 
rørRain_Brøns = pd.read_excel("{}\\Rør+Rain_Brøns.xlsx".format(GisLoc)).iloc[:,[0,-3,-2,-1]]
rørGWL_Brøns = pd.read_excel("{}\\Rør+GWLevel_Brøns.xlsx".format(GisLoc)).iloc[:,[0,-1]]
rørSoilCG_Zink_Brøns = pd.read_excel("{}\\Rør+SoilCG_Zink_Brøns.xlsx".format(GisLoc)).iloc[:,[0,-1]]
rørGWCG_Bly_Brøns = pd.read_excel("{}\\Rør+GWCG_Bly_Brøns.xlsx".format(GisLoc)).iloc[:,[0,-1]]
rørGWCG_Cyanide_Brøns = pd.read_excel("{}\\Rør+GWCG_Cyanide_Brøns.xlsx".format(GisLoc)).iloc[:,[0,-1]]
rørGWCG_Detergents_Brøns = pd.read_excel("{}\\Rør+GWCG_Detergents_Brøns.xlsx".format(GisLoc)).iloc[:,[0,-1]]
rørGWCG_Pesticides_Brøns = pd.read_excel("{}\\Rør+GWCG_Pesticides_Brøns.xlsx".format(GisLoc)).iloc[:,[0,-1]]
# rørRoad_Main_Brøns = pd.read_excel("{}\\Kortdata\\RoadType\\Rør_MainRoadDistances_Brøns.xlsx".format(folderLoc)).iloc[:,[2,-1]] # Her er ID'et af en eller anden årsag i kolonne 3
# rørRoad_Train_Brøns = pd.read_excel("{}\\Kortdata\\RoadType\\Rør_TrainDistances_Brøns.xlsx".format(folderLoc)).iloc[:,[2,-1]] # Her er ID'et af en eller anden årsag i kolonne 3
rørSoilType_Brøns = pd.read_excel("{}\\Kortdata\\SoilType\\SoilType_Rør_Brøns.xlsx".format(folderLoc)).iloc[:,[0,-7]]
rørRoadType_Brøns = pd.read_excel("{}\\Kortdata\\RoadType\\RoadType_Rør_Brøns.xlsx".format(folderLoc)).iloc[:,[0,-8]] # For at tilføje MaxSpeed, tilføj '-7'
rørLandUse_Brøns = pd.read_excel("{}\\Kortdata\\LandUse\\LandUse_Rør_Brøns.xlsx".format(folderLoc)).iloc[:,[0,-7]]
rørAnaerobicDepth_Brøns = pd.read_excel("{}\\Kortdata\\DepthToAnaerobicSoilConditions\\AnaerobicDepth_Rør_Brøns.xlsx".format(folderLoc)).iloc[:,[0,-1]]

havariRain_Brøns = pd.read_excel("{}\\Havari+Rain_Brøns.xlsx".format(GisLoc)).iloc[:,[0,-3,-2,-1]]
havariGWL_Brøns = pd.read_excel("{}\\Havari+GWLevel_Brøns.xlsx".format(GisLoc)).iloc[:,[0,-1]]
havariSoilCG_Zink_Brøns = pd.read_excel("{}\\Havari+SoilCG_Zink_Brøns.xlsx".format(GisLoc)).iloc[:,[0,-1]]
havariGWCG_Bly_Brøns = pd.read_excel("{}\\Havari+GWCG_Bly_Brøns.xlsx".format(GisLoc)).iloc[:,[0,-1]]
havariGWCG_Cyanide_Brøns = pd.read_excel("{}\\Havari+GWCG_Cyanide_Brøns.xlsx".format(GisLoc)).iloc[:,[0,-1]]
havariGWCG_Detergents_Brøns = pd.read_excel("{}\\Havari+GWCG_Detergents_Brøns.xlsx".format(GisLoc)).iloc[:,[0,-1]]
havariGWCG_Pesticides_Brøns = pd.read_excel("{}\\Havari+GWCG_Pesticides_Brøns.xlsx".format(GisLoc)).iloc[:,[0,-1]]
# havariRoad_Main_Brøns = pd.read_excel("{}\\Kortdata\\RoadType\\Havari_MainRoadDistances_Brøns.xlsx".format(folderLoc)).iloc[:,[2,-1]] # Her er ID'et af en eller anden årsag i kolonne 3
# havariRoad_Train_Brøns = pd.read_excel("{}\\Kortdata\\RoadType\\Havari_TrainDistances_Brøns.xlsx".format(folderLoc)).iloc[:,[2,-1]] # Her er ID'et af en eller anden årsag i kolonne 3
havariSoilType_Brøns = pd.read_excel("{}\\Kortdata\\SoilType\\SoilType_Havari_Brøns.xlsx".format(folderLoc)).iloc[:,[0,-7]]
havariRoadType_Brøns = pd.read_excel("{}\\Kortdata\\RoadType\\RoadType_Havari_Brøns.xlsx".format(folderLoc)).iloc[:,[0,-8]] # For at tilføje MaxSpeed, tilføj '-7'
havariLandUse_Brøns = pd.read_excel("{}\\Kortdata\\LandUse\\LandUse_Havari_Brøns.xlsx".format(folderLoc)).iloc[:,[0,-7]]
havariAnaerobicDepth_Brøns = pd.read_excel("{}\\Kortdata\\DepthToAnaerobicSoilConditions\\AnaerobicDepth_Havari_Brøns.xlsx".format(folderLoc)).iloc[:,[0,-1]]

rørLandUse_Brøns.columns = ["SYSTEM_ID","LandUse"]
havariLandUse_Brøns.columns = ["SYSTEM_ID","LandUse"]

# Length data: 
pipeLength_rør = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\PipeLengths_Rør.xlsx".format(folderLoc))
pipeLength_havari = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\PipeLengths_Havari.xlsx".format(folderLoc))

# Joint data: 
nJoints_rør = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\Joints_Subpipe_Rør.xlsx".format(folderLoc))
nJoints_havari = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\Joints_Subpipe_Havari.xlsx".format(folderLoc))
nJoints_rør_hofor = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\Samlinger_Rør_Brøns(Samling-rør-forbindelse).xlsx".format(folderLoc))

# Operational Data: 
rørSupplyDist_brøns = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\Rør_SupplyDist_Brøns.xlsx".format(folderLoc)).iloc[:,[0,-1,-7,-6]]
havariSupplyDist_brøns = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\Havari_SupplyDist_Brøns.xlsx".format(folderLoc)).iloc[:,[0,-1,-7,-6]]
rørSupplyDist_brøns.columns = ["ID","SupplyDist","X_UTM","Y_UTM"]
havariSupplyDist_brøns.columns = ["ID","SupplyDist","X_UTM","Y_UTM"]
operationalData = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\CSV_Temperature and Pressure data\\CSV_Temperature and Pressure data.xlsx".format(folderLoc))
havariTopography_Brøns = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\Havari+Topo_Brøns.xlsx".format(folderLoc)).iloc[:,[0,-7]]
rørTopography_Brøns = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\Rør+Topo_Brøns.xlsx".format(folderLoc)).iloc[:,[0,-7]]

#%%
endPipesData = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\Joints_Subpipe_Connections.xlsx".format(folderLoc))

# Concatenating all relevant comtaminants 
rørSoilCG_Brøns = rørSoilCG_Zink_Brøns
havariSoilCG_Brøns = havariSoilCG_Zink_Brøns

BrønsRørList = [rørRain_Brøns,rørGWL_Brøns,rørSoilCG_Brøns,rørGWCG_Bly_Brøns,rørGWCG_Cyanide_Brøns,rørGWCG_Detergents_Brøns,rørGWCG_Pesticides_Brøns,rørSoilType_Brøns,rørRoadType_Brøns,rørLandUse_Brøns,rørAnaerobicDepth_Brøns]
BrønsHavariList = [havariRain_Brøns,havariGWL_Brøns,havariSoilCG_Brøns,havariGWCG_Bly_Brøns,havariGWCG_Cyanide_Brøns,havariGWCG_Detergents_Brøns,havariGWCG_Pesticides_Brøns,havariSoilType_Brøns,havariRoadType_Brøns,havariLandUse_Brøns,havariAnaerobicDepth_Brøns]


#%% 
# ------------  Funktioner  -------------
#  ----- Til behandling af Havari-data -----
def DetermineHavariAge(df0,cborn = "Dato for anlæg udført",cdie="Dato for hændelse"): 
    """Determines the age of the broken pipes, and removes all the rows where a faulty date is used. 
    input:df of broken pipes, columnname of when the pipes were put in the ground and when they broke"""
    bornDate = df0[cborn].apply(ConvertStr2Date)
    deathDate = df0[cdie].apply(ConvertStr2Date)
    agediffs = np.zeros(len(bornDate))
    deathYears = agediffs.copy() 
    for i in range(len(bornDate)):
        deathYears[i] = deathDate[i].year 
        agediffs[i] = deathDate[i].year - bornDate[i].year
    age = pd.DataFrame({"Age":agediffs})
    deathYears_df = pd.DataFrame({'Havari Year':deathYears})
    df0 = df0.drop([cborn,cdie],axis=1)
    df0 = pd.concat([df0,age,deathYears_df],axis=1)
    df0 = GroupAges(df0) 

     # Fjerner alle de rækker, hvor datoen ikke er indskrevet, og hvor vi derfor ikke kan bestemme alderen. (I disse data er anlægsdaten for alle sammen forstået som for flere hundrede år siden)
    df0 = df0[df0["AgeGroup"] < 100] # Ingen rør må være mere end 100 år gamle. 
    return df0

def WorkingWithHavariData(havariname:str = "Havarier_Brøns"):
    havariData = relData[havariname]
    # Bestemmer alderen og grupperer alderen for rørene: 
    havaridata_sorted = DetermineHavariAge(havariData)

    # Fjerner al dårlig data: 
    """Jeg har i model v1.0 fokuseret på at medtage alle rørfejl som er sket. Det kan altså både være fejl som er sket under monteringen eller som er sket pga. korrosion. 
    Dette har jeg valgt, da det alt sammen forårsager at rørene skal graves op, og det er alt sammen med i bathtub kurven. """

    # havaridata_sorted = havariData_KBH[havariData_KBH["Hændelses-type"].isin(["Skade","Defekt","Anden"])] # Dette er i virkeligheden nok ikke så relevant at sortere i
    havaridata_sorted = havaridata_sorted[havaridata_sorted["Beskadiget udstyr"].isin(["Rør","Kappe","Ukendt","Reduktion","Muffe - krympe","Muffe - svejse"])]#,"Alarm-wire","T","T-lige","Samling"])] # For nu fokuserer vi kun på rørene, det kan give mening senere at udvide horisonten en smule!!!
    havaridata_sorted = havaridata_sorted[~havaridata_sorted["Skade - type"].isin(["Graveskade","Monteringsfejl","Ingen"])] # Der kan muligvis fjernes flere, men holder det simpelt for nu
    # Fjerner de kolonner som er blevet brugt til at sortere dataen med, da de ikke skal bruges til mere. 
    havaridata_sorted = havaridata_sorted.drop(["Beskadiget udstyr","Skade - type","Skade - specificeret"],axis=1)
    return havaridata_sorted

# ----- Til behandling af ledningsdataen -----
def DeterminePipeAge(df0,cborn="Udført-dato"): 
    currentyear = 2024
    bornDate = df0[cborn]
    agediffs = np.zeros(len(bornDate)) 

    for i in range(len(bornDate)): 
        agediffs[i] =currentyear - bornDate[i].year
    age = pd.DataFrame({"Age":agediffs})
    df0 = df0.drop(cborn,axis=1)
    df0 = pd.concat([df0,age],axis=1)
    df0 = GroupAges(df0) 

     # Fjerner alle de rækker, hvor datoen ikke er indskrevet, og hvor vi derfor ikke kan bestemme alderen. (I disse data er anlægsdaten for alle sammen forstået som for flere hundrede år siden)
    df0 = df0[df0["AgeGroup"] < 100] # Ingen rør må være mere end 100 år gamle. 
    return df0

def WorkingWithLedningsData(ledningsname:str="Ledninger_Brøns"):
    rørdata = relData[ledningsname] 

    # Bestemmer alderen på rørene:
    rørdata_sorted = DeterminePipeAge(rørdata)

    """Jeg opdeler her dataen i de 2 primære rør-lokationer. Rørene som er placeret udenfor og i jorden og rørene, som er placeret i huse. 
    Mængden af rør, som ikke falder under en af disse kategorier er neglegerbar.
    For nu er dataen fra rørene i huset dog irrelevante, da vi ikke har informationer om hvor de defekte rør har været placeret 
    - og derfor antager vi at de ikke sammen har været placeret udenfor under jorden. """

    rørdata_jord, rørdata_hus =  rørdata_sorted.copy(), rørdata_sorted.copy()
    rørdata_jord = rørdata_jord[rørdata_jord["Placering"].isin(["I jord","I kammer"])] # Fjerner alle rør, som ikke er placeret udenfor i jorden. (Mængden af de andre datapunkter er så lille, at det er irelevant at tage med)
    rørdata_hus = rørdata_hus[rørdata_hus["Placering"].isin(["I hus"])] # Fjerner alle rør som ikke er placeret inde i huset. 
    # Fjerner de kolonner som er blevet brugt til at sortere dataen med, da de ikke skal bruges til mere. 
    rørdata_jord = rørdata_jord.drop(["Placering"],axis=1)
    rørdata_hus = rørdata_hus.drop(["Placering"],axis=1)

    return rørdata_jord,rørdata_hus

#  ----- Til opdeling af dataen -----
def RemoveIrelevantColumns(df0,relevantcolumns):
     # Fjerner alle irelevante kolonner
     dfout = df0["ID"].copy()
     for j in relevantcolumns:
         if not j:
            dfout = pd.DataFrame({})
         else: 
             dfout = pd.concat([dfout,df0[j]],axis=1)
    #  dfout = dfout.drop("ID",axis=1)
     return dfout

# ----- Til mergin af ledning og havari data -----
def ChangeValues(df0, dictionary):
    """Changes the values in a df column according to the values in the dctionary. 
    input: pd.df, dictionary""" 
    for i in dictionary.keys(): 
        for j in dictionary[i].keys():
            df0[i] = df0[i].apply(lambda x: j if x in dictionary[i][j] else x)
    return df0

# ----- Til GWL vs Pipe Depth -----
def GWLvsPipeDepth(GISdf,lowerlimit:bool = True):
    """Sets the lower and upper limit for the GWLevel. """
    avgPipeDepth = -0.8
    pipeRange = 1
    if np.mean(GISdf["GWLevel_1"]) > 0:
        GISdf["GWLevel_1"] = GISdf["GWLevel_1"] * -1
    copy = GISdf["GWLevel_1"].copy()
    if lowerlimit == True:
        GISdf.loc[copy < avgPipeDepth - pipeRange,"GWLevel_1"] = avgPipeDepth - pipeRange
    GISdf.loc[copy > 0,"GWLevel_1"] = 0

    # GISdf.loc[copy > avgPipeDepth + piperange,"GWLevel_1"] = "AbovePipe"
    # GISdf.loc[copy < avgPipeDepth - piperange,"GWLevel_1"] = "BelowPipe"
    # copy[copy >= avgPipeDepth + piperange] = avgPipeDepth - piperange-1
    # GISdf.loc[copy >= avgPipeDepth - piperange,"GWLevel_1"] = "AtPipe" 
    return GISdf


#%% 
# ----------  Opdeler dataen ------------
# Omskriver ID kolonne navne til at hedde det samme: 
for i in dataKeys: 
    data[i].rename(columns={"{}".format(data[i].columns[0]): 'ID'}, inplace=True) 

relData = {}
# Definerer de kolonner i hvert datasæt, som er relevante (ID kolonnen er automatisk taget med da den skal bruges til at sammenligne GIS dataen)
RelevanteKolonner = {
    # "Afgrening"   : [],
    # "ArmLure"     : [],
    # "Bøjning"     : [],
    # "FastspændingsKlods" : [],
    # "FastspændingsPunkt" : [],
    # "Hane"        : [],
    "HavariNuværendeNet_KBH" : ["Lednings-type","Dia. medie-rør mm","Materiale medie-rør","Prærør-fabrikat","Kanaltype ikke prærør","Dato for hændelse","Dato for anlæg udført","Skade - type","Skade - specificeret","Beskadiget udstyr"],
    # "KammerVV"    : [],
    # "Kompensator" : [],
    "LedningVF_KBH"   : ["Type","Rør-dia mm","Rør-matr.","Rør-fabr.","Kanal type","Udført-dato","Placering"],
    # "Reduktion"   : [], 
    # "Samling"     : []
    # "TunnelAlleHofor" : []
    "Ledninger_Brøns"   : ["Type","Rør-dia mm","Rør-matr.","Rør-fabr.","Kanal type","Udført-dato","Placering"],
    "Havarier_Brøns"   : ["Lednings-type","Dia. medie-rør mm","Materiale medie-rør","Prærør-fabrikat","Kanaltype ikke prærør","Dato for hændelse","Dato for anlæg udført","Skade - type","Skade - specificeret","Beskadiget udstyr"]

}
# Fjerner alle ikke-relevante kolonner: 
for i in dataKeys: 
    relData[i] = RemoveIrelevantColumns(data[i],RelevanteKolonner[i])


#%% 
# ---------  Behandler Havari og lednings dataen  ---------  
havariData_KBH_Sorted = WorkingWithHavariData("HavariNuværendeNet_KBH")
havariData_Brøns_Sorted = WorkingWithHavariData("Havarier_Brøns")

rørData_KBH_Jord, rørData_KBH_Hus = WorkingWithLedningsData("LedningVF_KBH")
rørData_Brøns_Jord, rørData_Brøns_Hus = WorkingWithLedningsData("Ledninger_Brøns")




#%% 
# ---------  Adds the GIS data to the data set  --------- 
GIS_Rør_Brøns = CombineDFonID(BrønsRørList)
GIS_Havari_Brøns = CombineDFonID(BrønsHavariList)

# Manipulates the GWL: 
GIS_Rør_Brøns = GWLvsPipeDepth(GIS_Rør_Brøns,lowerlimit=False)
GIS_Havari_Brøns = GWLvsPipeDepth(GIS_Havari_Brøns,lowerlimit=False)
GIS_Rør_Brøns.rename(columns={"SYSTEM_ID":"ID"},inplace=True) 
GIS_Havari_Brøns.rename(columns={"SYSTEM_ID":"ID"},inplace=True) 

# Merger GISdataen på Hofor dataen: 
havariData_Brøns_WGIS = havariData_Brøns_Sorted.merge(GIS_Havari_Brøns,how='left',left_on='ID',right_on="ID")
rørData_Brøns_WGIS = rørData_Brøns_Jord.merge(GIS_Rør_Brøns,how='left',left_on='ID',right_on='ID')

# Ændrer GWContamination til 0, hvis røret ligger over GWLevel: 
# FIX THIS to include all the new dataa files (If relevant)!!! 
# lowerPipeLocLimit = np.min(havariData_Brøns_WGIS["GWLevel_1"])
# havariBrøns_PipeLevelIndex = havariData_Brøns_WGIS[havariData_Brøns_WGIS["GWLevel_1"] == lowerPipeLocLimit].index
# RørBrøns_PipeLevelIndex = rørData_Brøns_WGIS[rørData_Brøns_WGIS["GWLevel_1"] == lowerPipeLocLimit].index
# havariData_Brøns_WGIS.loc[havariBrøns_PipeLevelIndex,"GWContamin"],rørData_Brøns_WGIS.loc[RørBrøns_PipeLevelIndex,"GWContamin"] = 0,0

# Renames columns: 
renameDic = {'Type':"InsulationType", 'Lednings-type':"InsulationType",'Rør-dia mm':'InnerDia','Dia. medie-rør mm':'InnerDia','Rør-matr.':'InnerPipeMate','Materiale medie-rør':'InnerPipeMate',
            'Rør-fabr.':'Fabricator','Prærør-fabrikat':'Fabricator','Kanal type':'KanalType','Kanaltype ikke prærør':'KanalType',
            "Rain_1":"Rain","GWLevel_1":"GWLevel","SoilCG_Zin":"SoilCG_Zink",'GWCG_Bly_1':'GWCG_Lead', 'GWCG_Cyani':'GWCG_Cya', 'GWCG_Deter':'GWCG_Det', 'GWCG_Pesti':'GWCG_Pest',
            'Hovedvej_D':'MainRoad_Dist', 'Train_Dist':'Train_Dist', 'jordart':'SoilType', 'fclass':'RoadType', 'maxspeed':'MaxSpeed','Anaerobic_':'Anae_Depth'
            }
havariData_KBH_Sorted.rename(columns=renameDic,inplace=True)
rørData_KBH_Jord.rename(columns=renameDic,inplace=True)
havariData_Brøns_WGIS.rename(columns=renameDic,inplace=True)
rørData_Brøns_WGIS.rename(columns=renameDic,inplace=True)

cNames = havariData_Brøns_WGIS.columns
print("Kolonnenavne: {}".format(cNames)) 


#%% 
# ----------  Sørger for at data strukturen i de forskellige datafiler er ens  ---------- 

"""Definerer de kolonner, hvor der skal udskiftes værdier (De er øverste niveau i dic'et), og laver et underdic, hvor hver key er en str som 
skal erstatte de nuværende strings, og hver værdi i underdic'et er en list med de strings der skal erstattes. """
# dataTotal = pd.concat([havariData_KBH_Sorted,rørData_KBH_Jord],axis=0,ignore_index=True) # Samler dataen i en df, for nemmere at kunne finde unikke values - Kan udkommenteres npr valueDic er lavet
valueDic_KBH = {
    'InsulationType':{"PreInsulated" : ["Præ-isoleret - enkelt","Præ-isoleret-dobb","Præ-isoleret-enk.","Præ-isoleret - dobbelt rund"],
        "NotPreInsulated" : ['Ikke præ-isoleret i kanal','Ikke præ-isoleret i bøsningsrør', 'Ikke præ-isoleret i kælder','Ikke præ-isoleret i kammer','Ikke præ-isoleret'],
        "Unknown" : "Ukendt"},
    'InnerDia' : {"Unknown" : 'Ukendt'},
    'InnerPipeMate':{'Other' : 'Andet', 
        "Unknown"  : "Ukendt", 
        'Steel' : 'Stål'},
    'Fabricator':{"Unknown" : ['Ukendt','-'], 
        "Other"   : ['Andet','Cusinus','CuSINUS','Dürotan',"Flexwell",'Stjerne'], # Alle firmaer derhar produceret 3 fejlrør eller mindre er sat som "Other"
        "Iso-Plus": 'IsoWarm', # Antager at dette er en tastefejl, da der kun er 1 IsoWarm 
        'ICM/Alstom/ABB' : ['ICM','ABB','ICM / Alstom / ABB'],
        'Loegstoer' : 'Løgstør'},
    'KanalType':{"Unknown" : ['Ukendt'],
        "NoChannel"     : ['-'], 
        "Other"   : ['Andre','Andre','Anden',"C","D","D-speciel"],
        "B"       : ["B-speciel"]}    
}
valueDic_Brøns = valueDic_KBH | {
    'LandUse':{'Park etc.':['forest','grass','park','meadow'],
        'Commercial':['commercial','retail','industrial'],
        'Unknown':['scrub']},
    'RoadType':{'FootPath':['footway','steps','pedestrian','cycleway','path'],  # Pedestrian burde allokeres automatisk til større veje med det nye system, men hvis der ikke er nogen stor vej i nærhedne må det være bedst at allokere det til gå-veje. 
        'Residential':['residential','service']}, 
    'SoilType':{'FT - Ferskvandstørv':['SØ - Ferskvand'],  # Det eneste 'SØ - Ferskvand' datapunkt der er, skyldes at geometrien er placeret en smule forkert. Det ligger ikke rent faktisk under en sø. 
        'ML - Moraeneler':'ML - Moræneler'}
} 
NaNDic = {
    'KanalType':"NoChannel"
}

#TODO: 
# Lav et 'slet' dic, som sletter de værdier som jeg vælger at slette. 
    # Tilføj {LandUse:scrub}
# Ændre den automatiske slet funktion længere nede til at basere sig på denne nye sletDic. 

# Sørger for at forskellige værdier med ens betydning får ens værdi ifølge valueDic:  
havariData_KBH_Sorted, rørData_KBH_Jord = ChangeValues(havariData_KBH_Sorted,valueDic_KBH), ChangeValues(rørData_KBH_Jord,valueDic_KBH)
havariData_Brøns_WGIS, rørData_Brøns_WGIS = ChangeValues(havariData_Brøns_WGIS,valueDic_Brøns), ChangeValues(rørData_Brøns_WGIS,valueDic_Brøns)

# Sætter alle NaN værdier til specifikke værdier ifølge NaNDic: 
havariData_KBH_Sorted, rørData_KBH_Jord = ChangeNaN(havariData_KBH_Sorted,NaNDic), ChangeNaN(rørData_KBH_Jord,NaNDic)
havariData_Brøns_WGIS, rørData_Brøns_WGIS = ChangeNaN(havariData_Brøns_WGIS,NaNDic), ChangeNaN(rørData_Brøns_WGIS,NaNDic)

# Behandler "Unknown" og NaN data.
"""Først omdanner jeg alle de unknown værdier som jeg kan sætte i sammenhæng med en anden kolonne:""" 
havariData_KBH_Sorted = CorrelateUnknowns(havariData_KBH_Sorted,"InsulationType","KanalType",strcorrelate="NoChannel",insert = "PreInsulated",invert=False)
havariData_KBH_Sorted = CorrelateUnknowns(havariData_KBH_Sorted,"InsulationType","KanalType",strcorrelate="NoChannel",insert = "NotPreInsulated",invert=True)
havariData_Brøns_WGIS = CorrelateUnknowns(havariData_Brøns_WGIS,"InsulationType","KanalType",strcorrelate="NoChannel",insert = "PreInsulated",invert=False)
havariData_Brøns_WGIS = CorrelateUnknowns(havariData_Brøns_WGIS,"InsulationType","KanalType",strcorrelate="NoChannel",insert = "NotPreInsulated",invert=True)

# Resetter index: 
havariData_KBH_Sorted = ResetIndex(havariData_KBH_Sorted)
rørData_KBH_Jord = ResetIndex(rørData_KBH_Jord)
havariData_Brøns_WGIS = ResetIndex(havariData_Brøns_WGIS)
rørData_Brøns_WGIS = ResetIndex(rørData_Brøns_WGIS)

# Sørger for at alle floats faktisk er floats og ikke strings: 
"""For at gøre dette skal index være resettet"""
havariData_KBH_Sorted, rørData_KBH_Jord = dfStr2dfFloat(havariData_KBH_Sorted), dfStr2dfFloat(rørData_KBH_Jord)
havariData_Brøns_WGIS, rørData_Brøns_WGIS = dfStr2dfFloat(havariData_Brøns_WGIS),dfStr2dfFloat(rørData_Brøns_WGIS)

"""Tjekker hvor mange "Unknown values der er i hvert datasæt og sammenligner med størrelsen af datasættet. Hvis Unknown rows er under 0.5% af datasættet, sletter jeg dem bare. 
I de kolonner, hvor unknown værdierne er mere end grænseværdien omdannes det til at blive en normalfordeling af hvad resten af dataen er.
Dette gøres så vi ikke behøver at slette en hel række pga. et manglende datapunkt.
To do this, the 'dfStr2dfFloat' function must have been run. """
havariData_KBH_Sorted, rørData_KBH_Jord = ChangeUnknowns(havariData_KBH_Sorted), ChangeUnknowns(rørData_KBH_Jord) 
havariData_Brøns_WGIS, rørData_Brøns_WGIS = ChangeUnknowns(havariData_Brøns_WGIS),  ChangeUnknowns(rørData_Brøns_WGIS)

#%% 
# ---------  Adds length and joints to the data set   ---------
# Code for calculating the lengths and joints are found in the GISdata.py file. 

# Adds length as a feature to the data set: 
havariData_KBH_withLength = havariData_KBH_Sorted.copy()
rørData_KBH_withLength = rørData_KBH_Jord.copy()
havariData_Brøns_withLength = havariData_Brøns_WGIS.merge(pipeLength_havari,how='left',left_on="ID",right_on="ID")
rørData_Brøns_withLength = rørData_Brøns_WGIS.merge(pipeLength_rør,how='left',left_on="ID",right_on="ID")

# Adds joints as a feature to the data set:
havariData_KBH_withJoints = havariData_KBH_withLength.copy()
rørData_KBH_withJoints = rørData_KBH_withLength.copy()
havariData_Brøns_withJoints = havariData_Brøns_withLength.merge(nJoints_havari,how='left',left_on="ID",right_on="ID")
rørData_Brøns_withJoints = rørData_Brøns_withLength.merge(nJoints_rør,how='left',left_on="ID",right_on="ID")

# Analyse af Hofors samlingsdata: 
samlingerTilRør = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\Samlinger_Rør_Brøns(Samling-rør-forbindelse).xlsx".format(folderLoc))
samlingerTilRør = samlingerTilRør.drop_duplicates(subset='System ID', keep='first')
samlingsCount = CountCopies(pd.DataFrame(samlingerTilRør["SYSTEM_ID"]))
compareSamlingsCount0 = pd.merge(rørData_Brøns_withJoints[["ID",'nJoints']],samlingsCount,how='inner',left_on='ID',right_on='SYSTEM_ID')
compareSamlingsCount0.drop_duplicates(subset='ID', keep='first')
compareSamlingsCount=compareSamlingsCount0.iloc[:,[1,-1]]
diff = compareSamlingsCount.iloc[:,0] - compareSamlingsCount.iloc[:,1]
# print(np.std(diff))
# print(np.mean(diff))
# print(np.mean(compareSamlingsCount.iloc[:,0]),np.mean(compareSamlingsCount.iloc[:,1]))
# print(np.sum(compareSamlingsCount.iloc[:,0]),np.sum(compareSamlingsCount.iloc[:,1]))


# %%
# ---------  Allocates faults to pipes  ---------
"""The goal here is to allocate each fault to a pipe, meaning the need for a havari-data set is obsolete. Each pipe can 
have more than 1 fault allocated. """

FaultToPipeDM_Brøns = pd.read_excel("{}\\HOFOR\\Data Broenshoej\\Fault2PipeDistanceMatrix_Brøns.xlsx".format(folderLoc)).iloc[:,[1,0]]
pipeFaults_Brøns = CalcNFaultsPrPipe(FaultToPipeDM_Brøns,rørData_Brøns_withJoints,list(havariData_Brøns_withJoints.iloc[:,0]))
pipeFaults_Brøns.columns = ["ID","nFaults"]

havariData_KBH_withnFaults = havariData_KBH_withJoints.copy()
rørData_KBH_withnFaults = rørData_KBH_withJoints.copy()

rørData_Brøns_withnFaults = rørData_Brøns_withJoints.merge(pipeFaults_Brøns,how='left',left_on="ID",right_on="ID") # Adds the number of faults to the pipes.

havariData_Brøns_withnFaults = havariData_Brøns_withJoints.copy()
havariData_Brøns_withnFaults["nFaults"] = 0 


# %%
# ---------  Operational Data  ---------
"""The goal is to implement the operational data, thu sallocating a water temperature and pressure to each pipe in the network. 
""" 

def CalcWeightedAverage(df0, cWeights:str='Length', cValues:str='InnerDia'):
    """Calculate the weighted average of the values in the column 'cValues' dependent on the weights in 'cWeights'
    Input: 
    df0: dataframe with the columns 'cWeights' and 'cValues'

    Output: Float describing the weighted average. 
    """
    weighted_avg = (df0[cValues] * df0[cWeights]).sum() / df0[cWeights].sum()
    return weighted_avg

def CalcTemperatureLoss(supplyDistances,T_supply:float=90,T_ambient:float=8.8,lambda_i:float=0.027,D_i_avg:float=213,D_p_avg:float=55.6): 
    """Calculates the water temperature in every pipe in the network depending on the distance from the supply point, and the temperature 
    of the supply water. 
    
    Input: 
    supplyDistances: df describing the supply distances for each pipe in [m]. Must have the columns 'ID' and 'SupplyDist'.
    T_supply: The temperature of the water at the suppyl point in [degC]
    T_ambient: The temperature of the surrounding soil in [degC]
    lambda_i: Coefficient of heat conduction of the insulation material in [W/(K*m)]
    D_i: Diameter of the insulation and inner metal pipe in [mm]
    D_p: Diameter of the internal metal pipe in the DH pipe in [mm]
    
    output: 

    """
    q_loss = (T_supply-T_ambient)*2*np.pi*lambda_i/(np.log(D_i_avg/D_p_avg)) # Unit: [W/m]
    energyLossPipes = q_loss * supplyDistances['SupplyDist'] # Unit: [W]
    c_water70 = 3925.2 # Unit: [J/(kg*K)] TODO: Exchange this with a formula depending on the Temp
    rho_water70 = 977.63 # Unit: [kg/m^3] TODO: Exchange this with a formula depending on the Temp
    dV = 1*np.pi * (D_p_avg/2)**2 * 1/1000**2 # Unit: [m^3] Unit volume of water used to convert J -> K
    dm = dV * rho_water70 # Unit [kg]
    dT = energyLossPipes/(c_water70*dm) # Unit: [degC] Change in temperature from the supply point
    waterTemp = T_supply - dT 
    supplyDistances["WaterTemp"] = waterTemp

    dic = {'Smørumvej':{
        'ID':316964704,
        'cTemp':'Smørumvej 197  [°C]',
        'cPressure':'Smørumvej 197 [bar]'
        },
        'Frederikssundsvej':{
        'ID':2410285,
        'cTemp':'Frederikssundsvej 203 [°C]',
        'cPressure':'Frederikssundsvej 203 [bar]'
        },
        'Muldager':{
        'ID':333054174,
        'cTemp':'Muldager 20 [°C]',
        'cPressure':'Muldager 20 [bar]'
        }
    }
    


    return supplyDistances.copy()

def HandleOperationalData(operationalData0): 
    """Removes all data where the network was not running as it should have. This includes
    any instance, where the added heat to the network is 0, as the network likely got 
    energy from another source, which do not have access to. 

    Input: 
    operationalData0: df bescribing the temperature and pressure at the supply points as
    well as 3 measurement points. Additionally, the heat added to the network by the 
    supply point is shown in the column 'Delivered heat [MW]'. 
    
    Output: 
    operationalData0
    """
    operationalData0 = operationalData0[~operationalData0["Delivered heat [MW]"].isin([-0.1,0.0,0])]
    operationalData0 = operationalData0.dropna() # Removes blancks
    operationalData0 = operationalData0[~operationalData0["Delivered heat [MW]"].apply(lambda x: isinstance(x, str))] # Removes all str values from Delivered Heat as they are above 1000MW in June which makes no sense. 

    return operationalData0.copy()

def SplitOperationalData(OpData0):
    """
    Splits the input DataFrame into four new DataFrames containing data from each quarter.
    Adds a row at the end of each DataFrame showing the average of each column.
    
    Input:
    OpData0 (pd.DataFrame): Input DataFrame containing a 'Timestamp' column with datetime data in the format 'dd-mm-yyyy HH:MM:SS'.
                       The rest of the columns contain float values.
    
    Output:
    dict: A dictionary with keys 'Q1', 'Q2', 'Q3', 'Q4' and values as DataFrames for each quarter.
    """
    # Convert 'Timestamp' column to datetime
    OpData0['Timestamp'] = pd.to_datetime(OpData0['Timestamp'], format='%d-%m-%Y %H:%M:%S')
    
    # Extract the quarter from the 'Timestamp' column
    OpData0['Quarter'] = OpData0['Timestamp'].dt.quarter
    
    # Iterate over each quarter (1 to 4)
    dfs = {}
    for quarter in range(1, 5):
        # Filter the dataframe for the current quarter
        quarter_df = OpData0[OpData0['Quarter'] == quarter].copy()
        
        # Drop the 'Quarter' column
        quarter_df.drop(columns=['Quarter'], inplace=True)
        
        # Calculate the average for each column except 'Timestamp' and append it as a new row
        averages = quarter_df.drop(columns=['Timestamp']).mean()
        averages['Timestamp'] = 'Average'
        quarter_df = pd.concat([quarter_df, pd.DataFrame([averages])],axis=0, ignore_index=True)
        
        # Store the dataframe in the dictionary
        dfs[f'Q{quarter}'] = quarter_df
    
    return dfs
#%%
def DetermineTopography(topographyData,cMAX='MAXKOTE', supplyHeight:float=17.5): 
    """Takes a df with the column 'MAXKOTE' and returns a new df which describes the height difference
    between the supply point and the pipe. 
    Negative height diffs means the pipe is located lower than the supply point. """
    heightDiffs = topographyData[cMAX] - supplyHeight
    dfout = pd.concat([topographyData.iloc[:,0],heightDiffs],axis=1)
    dfout.columns = ['ID','Height_Diff']
    return dfout



rørHeightsDiffs = DetermineTopography(rørTopography_Brøns)


def LinearPressureApprox(supplyDist_rør,supplyDist_havari,endPipeData,dP=2.0,supplyPressure=4.5):
    """Makes a linear approximation of the pressure drop in the pipes (which is what is expected). 
    The Linear approximation is performed based on the average supply distance to the consumers, as 
    well as the total pressure drop over the network and the absolute supply pressure.  

    Input:
    supplyDistData: df with the columns: "ID" of the pipe, "SupplyDist" describin the the distance 
    from all pipes to the supply point through the network, 'X_UTM', 'Y_UTM' 
    endPipeData: df of all the pipes, describin the number of joints that pipe has due to 
    connections to other pipes. If nJoints=1, then it must be a pipe located at a consumer. Must
    have the columns: 'ID' and 'nJoints' 
    dP: Total supplied pressure to the network from the supply station. 
    supplyPressure: Absolute pressure measured at the supply station. 

    Return: The absolute pressure approximations for rør and havari. 
    """
    endPipeIDs = endPipeData[endPipeData['nJoints'] == 1]['ID']
    endPipes = pd.merge(endPipeIDs,supplyDist_rør,on='ID',how='inner')
    endPipes.to_excel("{}\\HOFOR\\Data Broenshoej\\EndPipes.xlsx".format(folderLoc),index=False)
    averageSupply2ConsumerDist = np.mean(endPipes["SupplyDist"])
    dPpermeter = dP/averageSupply2ConsumerDist
    rørAbsolutePressure_brøns, havariAbsolutePressure_brøns = supplyDist_rør.copy() , supplyDist_havari.copy()
    rørAbsolutePressure_brøns["Pressure"] = supplyPressure - rørAbsolutePressure_brøns["SupplyDist"] * dPpermeter
    havariAbsolutePressure_brøns["Pressure"] = supplyPressure - havariAbsolutePressure_brøns["SupplyDist"] * dPpermeter
    return rørAbsolutePressure_brøns, havariAbsolutePressure_brøns

### Tilføjer Pressure til data sættet ud fra 

rørAbsolutePressure_brøns, havariAbsolutePressure_brøns = LinearPressureApprox(rørSupplyDist_brøns,havariSupplyDist_brøns,endPipesData, dP=2,supplyPressure=4.5)
rørAbsolutePressure_brøns.to_excel("{}\\HOFOR\\Data Broenshoej\\Rør_AbsolutePressures_brøns_Split.xlsx".format(folderLoc),index=False)
rørData_Brøns_withP = pd.merge(rørData_Brøns_withnFaults,rørAbsolutePressure_brøns,how='left',on='ID')
havariData_Brøns_withP = pd.merge(havariData_Brøns_withnFaults,havariAbsolutePressure_brøns,how='left',on='ID')


#%%
# averageInnerDia = CalcWeightedAverage(rørData_Brøns_withnFaults)

# operationalData_handled = dfStr2dfFloat(operationalData)
# operationalData_handled = HandleOperationalData(operationalData)

# operationalData_Quarters = SplitOperationalData(operationalData_handled.copy())

# StationIDs = [316964704,2410285,333054174] # In order from station 1-3

# rørSupplyDist_brøns_withT = CalcTemperatureLoss(rørSupplyDist_brøns,T_supply=76.9)
# havariSupplyDist_brøns_withT = CalcTemperatureLoss(havariSupplyDist_brøns,T_supply=76.9)
# rørSupplyDist_brøns_withT.to_excel("{}\\HOFOR\\Data Broenshoej\\rørSupplyDist_brøns_withT.xlsx".format(folderLoc),index=False)
# havariSupplyDist_brøns_withT.to_excel("{}\\HOFOR\\Data Broenshoej\\havariSupplyDist_brøns_withT.xlsx".format(folderLoc),index=False)
# TODO: 
# - Afvent Hofors svar 


#%% 
# ---------  Combines the data set  ---------
# Latest used data: 
rør_KBH_SemiFinal = rørData_KBH_withnFaults
havari_KBH_SemiFinal = havariData_KBH_withnFaults
rør_Brøns_SemiFinal = rørData_Brøns_withP
havari_Brøns_SemiFinal = havariData_Brøns_withP

# Add a "Did it Break" column, which will function as the target column
havariData_KBH_Withy = pd.concat([havari_KBH_SemiFinal,pd.DataFrame({"FaultOccurance" : np.ones(havari_KBH_SemiFinal.shape[0])})],axis=1)
rørData_KBH_Withy = pd.concat([rørData_KBH_withnFaults,pd.DataFrame({"FaultOccurance" : np.zeros(rørData_KBH_withnFaults.shape[0])})],axis=1)
havariData_Brøns_Withy = pd.concat([havari_Brøns_SemiFinal,pd.DataFrame({"FaultOccurance" : np.ones(havari_Brøns_SemiFinal.shape[0])})],axis=1)
rørData_Brøns_Withy = pd.concat([rør_Brøns_SemiFinal,pd.DataFrame({"FaultOccurance" : np.zeros(rør_Brøns_SemiFinal.shape[0])})],axis=1)

# Samler de forskellige datasæt: 
dataTotal_KBH = pd.concat([havariData_KBH_Withy,rørData_KBH_Withy],axis=0,ignore_index=True)
dataTotal_Brøns = pd.concat([havariData_Brøns_Withy,rørData_Brøns_Withy],axis=0,ignore_index=True)

# Gemmer datasæt: 
dataTotal_KBH.to_excel("{}\\HOFOR\\DataTotal_KBH.xlsx".format(folderLoc),index=False)
dataTotal_Brøns.to_excel("{}\\HOFOR\\DataTotal_Broenshoej.xlsx".format(folderLoc),index=False)



# %%
