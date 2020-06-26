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

import multiprocessing 
cores = min(8, multiprocessing.cpu_count())
print("Codigo ejecutado sobre %d cores"%cores)

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

    #remove noise.. Points within -std_limit- standard deviations from the mean and
    #               with errors greater than -error_limit- times the error mean are
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


import csv, time
def safe_write(save_f, value):
    while(1):
        try:
            with open(save_f, "a") as file:
                writer = csv.writer(file)
                writer.writerow(value)
            break 
        except:
            time.sleep(1)
            
def feets_parallel(i, times, mags, errs,  koi_n, save_f):
    resumen = pd.read_csv(save_f)
    if koi_n in resumen["KOI Name"].values:
        print ("Se omite curva %d: %s "%(i,koi_n))
    else:
        print ("Trabajando en curva %d: %s... "%(i,koi_n), end='')

        fs = feets.FeatureSpace(data=['time','magnitude','error'])
        features, values = fs.extract(time=times,magnitude=mags, error=errs)

        res = [koi_n]+list(values.T)
        safe_write(save_f, res) #read and save
        print("Terminado!")


name_saved_file = "Feets_Features/ResumenFeets_conError.csv"
if not os.path.isfile(name_saved_file):
    fs = feets.FeatureSpace(data=['time','magnitude','error'])
    resumen=pd.DataFrame(columns =['KOI Name']+list(fs.features_as_array_))
    resumen.to_csv(name_saved_file, index=False)

start_i = pd.read_csv(name_saved_file).shape[0] #leido desde archivo
if start_i % cores != 0:
    start_i = int(start_i/cores)*cores
if start_i == N:
    sys_out.write("Ya se realizó!")
    assert False
print("Comienza ejecucion en ",start_i)
    
for s in np.arange(start_i, N, cores):
    #va ejecutando cada #cores simultaneamente
    pool = multiprocessing.Pool(processes=cores)  
    for i in range(s, min(s+cores, N) ): #ejecutar cantidad de cores
        pool.apply_async(feets_parallel, args=(i,
                                                     coupled_time[i],
                                                     coupled_lc[i], 
                                                     coupled_err[i],
                                                     kois[i], 
                                                     name_saved_file))
    pool.close()
    pool.join()
    print("%d codigos correctamente ejecutados"%cores)
print("Ejecutando todos")
