"""The goal of this file is to handle and manipulate the geographical data and ready it for further handling in QGIS. The datafiles are saved as CSV-files. """

#%%
import pandas as pd
import numpy as np

folderLoc = "C:\\Users\\tvrj\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Arbejde\\Data"
MappedDataLoc = "{}\\Kortdata".format(folderLoc)
data_WaterLevel = pd.read_csv("{}\\Samples and Measurements, Water levels\\WaterLevels.csv".format(MappedDataLoc))
data_WaterLevel_Broens = pd.read_excel("{}\\Samples and Measurements, Water levels\\GWL_KunBroens.xlsx".format(MappedDataLoc))
# data_WaterCompoundGroup = pd.read_csv("{}\\Groundwater, Compounds\\GW_CompoundGroups.csv".format(MappedDataLoc))
# data_WaterCompoundGroup_Excel = pd.read_excel("{}\\Groundwater, Compounds\\GW_CompoundGroups_ExcelData.xlsx".format(MappedDataLoc))
data_SoilCG_Metals = pd.read_excel("{}\\Samples and Measurements, Soailair samples from boreholes\\SoilSamples_Metals.xlsx".format(MappedDataLoc))
# data_SoilCG_OrganicPol = pd.read_excel("{}\\Samples and Measurements, Soailair samples from boreholes\\SoilSamples_OrganicMicroPolution.xlsx".format(MappedDataLoc))
data_GWCG = pd.read_excel("{}\\GWCompoundGroups v2.0\\GWCG.xlsx".format(MappedDataLoc))
# data_WaterCompound = pd.read_csv("{}\\Groundwater, Compounds\\GW_Compounds.xlsx".format(MappedDataLoc))

#%% 
def FilterGWLbyMonths(df0,cdate:str,cGWL:str,keeplist:list,input:str="csv"): 
    """Filters the ground water level data to only contain data from the months with the highest ground water (November-March)
    input: pd.df
    output: Saves the filtered data as a csv-file, ready to be loaded in QGIS"""
    if input == "Excel": 
        df0[cdate] = pd.to_datetime(df0[cdate], format='%Y-%m-%d')
        unviableGWlevels = ''
    elif input == "csv": 
        df0[cdate] = pd.to_datetime(df0[cdate], format='%Y/%m/%d')
        unviableGWlevels = '###.## m'
    # Bruger kun data efter "startYear"
    startYear= 1990
    viableYears = df0[cdate].dt.year >= startYear 
    df0 = df0[viableYears] 
    # Fjerner alle dårlige værdier
    df0 = df0[df0[cGWL] != unviableGWlevels] 
    # Makes the text in the column into viable depth measurements 
    if input == "Excel": 
        df0[cGWL] = df0[cGWL]*-1 # This should be removed, next time GWL is updated. IT reverses the sign compared to what we want. 
        # df0[cGWL] = df0[cGWL].str.replace(",",".").astype(float)
    elif input == "csv": 
        df0[cGWL] = df0[cGWL].str[:-1] 
        df0[cGWL] = df0[cGWL].astype(float) 
        df0[cGWL] = df0[cGWL]*-1
    # Saves the entire dataset. 
    df0.rename(columns={cGWL:"GWLevel"},inplace=True) 
    keepList = keeplist + [cdate,"GWLevel"]
    df0 = df0[keepList]
    df0.to_csv("{}\\Samples and Measurements, Water levels\\GWLevels_Sorted.csv".format(MappedDataLoc))
    # Splits the data into the four seasons (Not that it is used in the current model)
    winterMonths = [12,1,2]
    springMonths = [3,4,5]
    summerMonths = [6,7,8]
    fallMonths   = [9,10,11]
    seasonList   = [winterMonths, springMonths, summerMonths,fallMonths]
    seasonNames  = ["GWLWinter","GWLSpring","GWLSummer","GWLFall"]
    for i, monthList in enumerate(seasonList): 
        seasonData = df0[df0[cdate].dt.month.isin(monthList)]
        seasonColumn = seasonData["GWLevel"]
        seasonColumn.rename(seasonNames[i],inplace=True) # Renaming a series and NOT a df
        dfout = pd.concat([df0,seasonColumn],axis=1)
        keepListSeasons = keeplist + [seasonNames[i]]
        dfout = dfout[keepListSeasons]
        dfout = dfout[~dfout[seasonNames[i]].isna()]
        dfout.to_csv("{}\\Samples and Measurements, Water levels\\{}.csv".format(MappedDataLoc,seasonNames[i]))
# FilterGWLbyMonths(data_WaterLevel,"dato_foers","label_hidd",["xutm32eure","yutm32eure"])
FilterGWLbyMonths(data_WaterLevel_Broens,"Dato seneste","Vsp mtu seneste",["Xutm32euref89","Yutm32euref89"],input="Excel")

def ColorMap(df0,ccolor): 
    colormap = {"255 51 255":"LightPurple","153 0 153":"Purple","102 0 102":"DarkPurple","0 255 0":"Green","0 0 255":"Blue","255 125 0":"Orange","255 255 0":"Yellow","255 0 0":"Red"}
    df0[ccolor] = df0[ccolor].map(colormap) # Rewrites rgb codes to color. 
    return df0

# FilterSoilCG foer ændring fra grupperede kontamineringsgrupper til at hver kontaminering er for sig. 
# def FilterSoilCG(df0,cjord:str="Art",calder:str="Alder",ccolor:str="Rgb",cstofgruppe:str="Stofgruppe",celement:str="Stof",relevantelements:list = [],keeplist:list=[]): 
#     """ Filtrerer jordproevedataen, således at det kun er relevante stoffer der bliver taget med i modellen og de bliver sorteres efter hvor slemt de er kontamineret
#     input: pd.df, strings med kolonnenavne, liste med relevante kolonner, liste med relevante elementer"""
#     # Mergers the two datasets, so only the data points with all the relevant information are used: 
#     df0 = df0[df0[cjord] == "BJORD"] # Vil kun have jord undersoegelser med
#     df0 = df0[df0[calder] <= 24] # Vil ikke have for gamle data med 
#     contaminationMap = {"Orange":1,"Red":2,"LightPurple":3,"Purple":4,"DarkPurple":5} # Defines badness level of contamination, 5 is worst
#     df0 = ColorMap(df0,ccolor) # Rewrites all rgb codes to color
#     df0[ccolor] = df0[ccolor].map(contaminationMap) # Rewrites colors to badness level  
#     if relevantelements != []: 
#         df0 = df0[df0[celement].isin(relevantelements)] 
#     df0 = df0.rename(columns = {ccolor:'ContaminationLevel',"xutm32eure":"X_UTM","yutm32eure":"Y_UTM","Id":"ID","Dgu nr":"DGU nr",celement:"Compound",cstofgruppe:"CompoundGroup","Alder":"Age"})
#     if keeplist != []: 
#         df0 = df0[keeplist]
#     return df0
# SoilCG_Metals = FilterSoilCG(data_SoilCG_Metals, relevantelements=["Bly","Kviksoelv","Zink"],keeplist=["ID","DGU nr","Age","CompoundGroup","Compound","ContaminationLevel","X_UTM","Y_UTM"])
# # SoilCG_OrganicPol = FilterSoilCG(data_SoilCG_OrganicPol)
# SoilCG_Metals.to_csv("{}\\Samples and Measurements, Soailair samples from boreholes\\SoilCG_Metals.csv".format(MappedDataLoc))
# # SoilCG_OrganicPol.to_csv("{}\\Groundwater, Compounds\\SoilCG_OrganicMicroPolution.csv".format(MappedDataLoc))

# Hold alle kontaminants for sig pånær sæbe  
# Indfoer Kontaminaeringsværdier fromfor de eksisterende "niveauer" 
def FilterSoilCG(df0,cjord:str="Art",calder:str="Alder",ccolor:str="Rgb",cstofgruppe:str="Stofgruppe",celement:str="Stof",relevantelements:list = [],keeplist:list=[]): 
    """ Filtrerer jordproevedataen, således at det kun er relevante stoffer der bliver taget med i modellen. Each contaminant is its own feature. 
    input: pd.df, strings med kolonnenavne, liste med relevante kolonner, liste med relevante elementer"""
    # Merges the two datasets, so only the data points with all the relevant information are used: 
    df0 = df0[df0[cjord] == "BJORD"] # Vil kun have jord undersoegelser med
    df0 = df0[df0[calder] <= 34] # Vil ikke have for gamle data med 
    
    contaminationMap = {"Orange":1,"Red":2,"LightPurple":3,"Purple":4,"DarkPurple":5} # Defines badness level of contamination, 5 is worst
    df0 = ColorMap(df0,ccolor) # Rewrites all rgb codes to color
    df0[ccolor] = df0[ccolor].map(contaminationMap) # Rewrites colors to badness level  
    if relevantelements != []: 
        df0 = df0[df0[celement].isin(relevantelements)] 
    df0 = df0.rename(columns = {ccolor:'ContaminationLevel',"Max maengde":"Contamination Amount","xutm32eure":"X_UTM","yutm32eure":"Y_UTM","Id":"ID","Dgu nr":"DGU nr",celement:"Compound",cstofgruppe:"CompoundGroup","Alder":"Age"})
    if keeplist != []: 
        df0 = df0[keeplist]
    for i, element in enumerate(relevantelements): 
        dfout = df0[df0["Compound"] == element]
        dfout.to_excel("{}\\Samples and Measurements, Soailair samples from boreholes\\SoilCG_{}.xlsx".format(MappedDataLoc,element))
    return df0
SoilCG_Metals = FilterSoilCG(data_SoilCG_Metals, relevantelements=["Bly","Kviksoelv","Zink"],keeplist=["ID","DGU nr","Age","CompoundGroup","Compound","ContaminationLevel","Contamination Amount","X_UTM","Y_UTM"])
# SoilCG_OrganicPol = FilterSoilCG(data_SoilCG_OrganicPol)
SoilCG_Metals.to_excel("{}\\Samples and Measurements, Soailair samples from boreholes\\SoilCG_Metals.xlsx".format(MappedDataLoc))
# SoilCG_OrganicPol.to_csv("{}\\Groundwater, Compounds\\SoilCG_OrganicMicroPolution.csv".format(MappedDataLoc))

def CalcAge(df0,cage:str="dato"): 
    dates = pd.DatetimeIndex(df0[cage].copy())
    years = dates.year
    ages = 2024 - years
    df0 = pd.concat([df0,pd.DataFrame({"Age":np.int64(ages)})],axis=1)
    df0 = df0.drop(cage,axis=1)
    # print(np.int64(ages))
    return df0
# a = CalcAge(data_GWCG,"dato_senes")

def FilterGWCG(df0,cstatus:str="status_ind",cdato:str="dato",cstofgruppe:str="stofgrup_1",celement:str="dominerend",camount:str="maengde",climit:str="graensevae",cx:str="X_UTM",cy:str="Y_UTM",relevantelements:list = [],keeplist:list=[]): 
    """ Filtrerer GW proevedataen, således at det kun er relevante stoffer der bliver taget med i modellen og de bliver sorteres efter hvor slemt de er kontamineret
    input: pd.df, strings med kolonnenavne, liste med relevante elementer, liste med relevante kolonner"""
    df0[cdato] = df0[cdato].fillna(df0["dato_senes"]) # Some dato values are empty, so these are filled with dato values from another column
    df0 = df0[~df0[cdato].isna()]
    df0 = CalcAge(df0,cdato) # Beregner alderen og ændrer column navnet til "Age"
    df0 = df0[df0["Age"] <= 34] # Vil ikke have for gamle data med 
    df0.rename(columns = {cstatus:'ContaminationLevel',camount:"Contamination Amount","xutm32eure":"X_UTM","yutm32eure":"Y_UTM","id":"ID",celement:"Compound",cstofgruppe:"CompoundGroup"},inplace=True)
    df0 = df0[keeplist]
    # Sorts the corrosive compound groups into distinct files, which later will be used in QGIS
    CGdic = {"Uorganiske sporstoffer":["274 (Bly)","207 (Cyanid, total)"],
             "Pesticider, nedbrydningsprodukter og beslaegtede stoffer":"All", # THere are many different detergents, but they are for now kept as one, as I dont know which are dangerous to the 
             "Detergenter (saebe)":"All"} # There are only 1 type in the data. 
    # Makes a df all the 0 measurements 
    df0["Contamination Amount"] = df0["Contamination Amount"].fillna(0) # Blanck element rows are rows no contaminants were found
    noContdf = df0[df0["Contamination Amount"] == 0.0].copy()
    # Makes a df for each compound
    for i,compoundgroup in enumerate(CGdic.keys()):
        dfCG = df0[df0["CompoundGroup"] == compoundgroup].copy()
        noContForCG = noContdf[noContdf["CompoundGroup"] == compoundgroup].copy() # Creates a df of all the 0-measurements for that specific CG. 
        if CGdic[compoundgroup] != "All": 
            for j, compound in enumerate(CGdic[compoundgroup]): 
                dfCG0 = dfCG[dfCG["Compound"].isin([compound])].copy()
                dfCG0 = pd.concat([dfCG0,noContForCG],axis=0)
                dfCG0.to_csv("{}\\GWCompoundGroups v2.0\\GW_CG_{}.csv".format(MappedDataLoc,compound))
                continue
        dfCG = pd.concat([dfCG,noContForCG],axis=0)
        dfCG.to_csv("{}\\GWCompoundGroups v2.0\\GW_CG_{}.csv".format(MappedDataLoc,compoundgroup))
FilterGWCG(data_GWCG,keeplist=["ID","X_UTM","Y_UTM","Age","CompoundGroup","Compound","ContaminationLevel","Contamination Amount"])

# FilterGWCG foer ændring fra grupperede kontamineringsgrupper til at hver kontaminering er for sig. 
# def FilterGWCG(df0,cstatus:str="status_ind",cdato:str="dato",cstofgruppe:str="stofgrup_1",celement:str="dominerend",camount:str="maengde",climit:str="graensevae",cx:str="X_UTM",cy:str="Y_UTM",relevantelements:list = [],keeplist:list=[]): 
#     """ Filtrerer GW proevedataen, således at det kun er relevante stoffer der bliver taget med i modellen og de bliver sorteres efter hvor slemt de er kontamineret
#     input: pd.df, strings med kolonnenavne, liste med relevante elementer, liste med relevante kolonner"""
#     df0 = df0[~df0[cdato].isna()]
#     df0 = CalcAge(df0,cdato) # Beregner alderen og ændrer column navnet til "Age"
#     df0 = df0[df0["Age"] <= 24] # Vil ikke have for gamle data med 
#     df0 = df0[~df0[celement].isna()] # Removes blanck element rows
#     # Sorts the corrosive compound groups into distinct files, which later will be used in QGIS
#     CGdic = {"Uorganiske sporstoffer":["274 (Bly)","207 (Cyanid, total)"],
#              "Pesticider, nedbrydningsprodukter og beslægtede stoffer":"All",
#              "Detergenter (sæbe)":"All"}
#     # Saves all the contamination data in a single dataframe.
#     dfCGTot = pd.DataFrame({}) 
#     for i,compoundgroup in enumerate(CGdic.keys()):
#         dfCG = df0[df0[cstofgruppe] == compoundgroup]
#         if CGdic[compoundgroup] != "All": 
#             dfCG = dfCG[dfCG[celement].isin(CGdic[compoundgroup])]
#         dfCGTot = pd.concat([dfCGTot,dfCG],axis=0) 

#     # if relevantelements != []: 
#     #     df0 = df0[df0[cstofgruppe].isin(relevantelements)] 
#     dfCGTot = dfCGTot.rename(columns = {cstatus:'ContaminationLevel',"xutm32eure":"X_UTM","yutm32eure":"Y_UTM","id":"ID",celement:"Compound",cstofgruppe:"CompoundGroup"})
#     dfCGTot = dfCGTot[keeplist]
#     dfCGTot.to_csv("{}\\GWCompoundGroups v2.0\\GW_CompoundGroups_Sorted.csv".format(MappedDataLoc))
#     return dfCGTot
# a = FilterGWCG(data_GWCG,keeplist=["ID","X_UTM","Y_UTM","Age","CompoundGroup","Compound","ContaminationLevel"])

def MatchingID(df1,cid1:str, df2,cid2:str): 
    # Convert arrays to sets
    array1 = np.array(df1[cid1])
    array2 = np.array(df2[cid2])
    set1 = set(array1)
    set2 = set(array2)

    # Find the intersection of the two sets
    matching_ids = set1.intersection(set2)
    matching_ids = np.array([i for i in matching_ids])
    if matching_ids.size == 0: 
        print("No matching IDs found :(") 
    # else: 
    #     print(matching_ids)
    return matching_ids
# matchingIDs = MatchingID(data_WaterCompoundGroup,"dgu_nr",data_WaterCompoundGroup_Excel,"Dgu nr")
# matchingIDs = MatchingID(data_WaterCompoundGroup,"id",data_WaterCompoundGroup_Excel,"Id")
# print(len(set(matchingIDs)))






