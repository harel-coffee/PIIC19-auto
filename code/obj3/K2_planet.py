"""
 K2 candidate data by NexSCI:  https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2candidates
 bulk data by: https://archive.stsci.edu/missions/hlsp/k2sff/
"""
import pandas as pd
import numpy as np


df_all = pd.read_csv("/work/work_teamEXOPLANET/K2_mission/k2candidates_2020.04.18_21.34.57.csv")
df_all.drop_duplicates(subset = ["epic_candname", "pl_name"],inplace=True)

df_all["k2_campaign"] = df_all["k2_campaign"].fillna(-5).astype("int").astype("str")

lines = []
for i,value in df_all.iterrows():
    id_v = value["epic_name"].replace("EPIC ","")
    c_number = "c"+ value["k2_campaign"].zfill(2)

    if c_number[1:] == "10":
        c_number += "2"
    #if c_number[1:] == "11": #está dividida en 2...
    #    c_number += "2" # hay solo 1 dato

    url = "http://archive.stsci.edu/missions/hlsp/k2sff/"
    url += c_number+ "/"
    url += id_v[:4]+ "00000" + "/"
    url += id_v[4:] + "/"
    url += "hlsp_k2sff_k2_lightcurve_"+id_v+"-"+c_number+"_kepler_v1_llc-default-aper.txt"       

    lines.append(url)

time = []
lc_norm = []
for line in lines:
    print("Leyendo un archivo de url: ",line)
    
    try:
        aux = pd.read_csv(line,index_col=False)
        time.append(aux['BJD - 2454833'].values)
        lc_norm.append(aux[" Corrected Flux"].values)
    except:
        print("No existe")
        time.append([np.nan])
        lc_norm.append([np.nan])
time = np.asarray(time)
lc_norm = np.asarray(lc_norm)

np.save("/work/work_teamEXOPLANET/K2_mission/K2_PLANET_time.npy",time)
np.save("/work/work_teamEXOPLANET/K2_mission/K2_PLANET_lc_detr.npy",lc_norm)
