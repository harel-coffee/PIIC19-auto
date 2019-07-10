
# coding: utf-8

# In[ ]:


import time
import numpy as np
from IPython.display import display, Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import pandas as pd
from hmmlearn import hmm
import numpy as np
import seaborn as sns
from scipy.stats import norm
import itertools
import matplotlib.pyplot as plt
from hmmlearn.hmm import GMMHMM
from  sklearn.metrics import mean_squared_error as mse
from astropy.io import fits
import matplotlib.pyplot as plt
import os, sys


def generate_trasition(lc_kepler,df,class_type,path):
    #df = pd.read_csv(csv) 
    index_df = df.loc[df['NExScI Disposition'] == class_type]
    npy_name = list(df['KOI Name'])
    raw_list = [lc_kepler[i] for i in list(index_df.index)]
    total_archivos = 0
    for i in range(len(raw_list)):
        tt = time.time()
        if not os.path.isfile(path_save+str(npy_name[i])+'.npy'):
            lc_our_detrend = raw_list[i]
            lc_wind_nan = []
            lengths = []
            sublist = []
            for value in lc_our_detrend:
                if np.isnan(value) and len(sublist) != 0:
                    lc_wind_nan.append(np.asarray(sublist))
                    lengths.append(len(sublist))
                    sublist = []
                elif np.isnan(value) and len(sublist) == 0:
                    continue
                else: 
                    sublist.append(value) 
            if len(sublist) != 0:   
                lc_wind_nan.append(np.asarray(sublist))
                lengths.append(len(sublist))
            lc_wind_nan = np.asarray(lc_wind_nan)   
            lengths = np.asarray(lengths)   
            lc_wind_nan = np.concatenate(lc_wind_nan).reshape(-1,1)
            lc_wind_nan = lc_wind_nan/np.abs(np.min(lc_wind_nan)) 

            n_sta=15
            markov_model = hmm.GaussianHMM(n_components=n_sta, n_iter=50)
            markov_model.fit(lc_wind_nan , lengths)
            np.save(path+str(npy_name[i]),markov_model.transmat_)
            total_archivos +=1
            print('Tiempo de ejecución: %i -- Total Archivos Generados: %i'%(time.time()-tt,total_archivos))
        else:
            print('LC %s ya transformada'%(str(npy_name[i])))
        
class_type = 'CANDIDATE'
path_save = "/work/work_teamEXOPLANET/MTF_gabo/candidatos/"
folder_lc = "/work/work_teamEXOPLANET/KOI_LC/"
#Clean Light Curves
lc_kepler = np.load(folder_lc+"cleaned/LC_kepler_processed.npy" )        
#Class type dataframe
df_sets = pd.read_csv(folder_lc+'csv/kepler_dataset.csv')
confirmed_df = df_sets.loc[df_sets['NExScI Disposition'] == class_type]
#Save Markov as npy
generate_trasition(lc_kepler,confirmed_df,class_type,path_save)


# In[1]:




