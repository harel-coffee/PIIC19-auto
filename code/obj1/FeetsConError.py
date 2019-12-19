import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random 
import time
import feets.datasets.base as lc
import feets
from scipy import stats
import warnings
import os,sys, gc
warnings.filterwarnings('ignore')

folder_lc = "/work/work_teamEXOPLANET/KOI_LC/"
time_kepler = np.load(folder_lc+"npy/KOI_LC_time.npy")
lc_kepler = np.load(folder_lc+"npy/KOI_LC_init.npy" )
err_kepler = np.load(folder_lc+"npy/KOI_LC_init_err.npy" )
#process_lc = np.load(folder_lc+'/cleaned/LC_kepler_processed.npy')
N, T = time_kepler.shape
print((N,T))

error_limit=3
std_limit=5

coupled_lc = []
coupled_time = []
coupled_err = []
for i in range(N):
    #borrar nans... arreglo variable
    mask_nan_aux = np.isnan(lc_kepler[i])
    coupled_lc.append(lc_kepler[i][~mask_nan_aux])
    coupled_time.append(time_kepler[i][~mask_nan_aux])
    coupled_err.append(err_kepler[i][~mask_nan_aux])

    #remove noise.. Points within ‘std_limit’ standard deviations from the mean and
    #               with errors greater than ‘error_limit’ times the error mean are
    #               considered as noise and thus are eliminated.
    mask_noise = ( np.abs(coupled_lc[i] - coupled_lc[i].mean()) > std_limit*coupled_lc[i].std() ) #std criteria
    mask_noise &= (coupled_err[i] > error_limit) #error criteria

    coupled_lc[i] = coupled_lc[i][~mask_noise]
    coupled_time[i] = coupled_time[i][~mask_noise]
    coupled_err[i] = coupled_err[i][~mask_noise]
    
coupled_lc = np.asarray(coupled_lc)
coupled_time = np.asarray(coupled_time)
coupled_err = np.asarray(coupled_err)

df_meta = pd.read_csv("/users/mbugueno/PIIC19/KOI_Data/kepler_dataset.csv")
kois=df_meta['KOI Name'].values

for i in range(N): 
    print ("Trabajando en KOI Name:", kois[i])
    name=str(kois[i])
    times=coupled_time[i]
    mags=coupled_lc[i]
    errs=coupled_err[i]
    
    fs = feets.FeatureSpace(data=['time','magnitude','error'])
    features, values = fs.extract(time=times,magnitude=mags, error=errs)
    
    if i==0:
        resumen=pd.DataFrame(columns =['KOI Name']+list(features))
    
    df_res=pd.DataFrame([[name]+list(values.T)], columns=['KOI_name']+list(features))
    resumen= resumen.append(df_res)

    
resumen.to_csv('ResumenFeets_ConError.csv', index=False)
