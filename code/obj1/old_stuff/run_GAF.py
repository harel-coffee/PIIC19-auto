
# coding: utf-8

from IPython.display import display, Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys ,gc, math, time
from PIL import Image
from keras.layers import *
from keras.models import Sequential
from keras import backend as K

from optparse import OptionParser
op = OptionParser()
op.add_option("-s", "--size", type="string", default="20", help="size of matrix to generate")
op.add_option("-p", "--pool", type="int", default=2, help="pool size")
(opts, args) = op.parse_args()
winds = opts.size.lower().split("/")
pool = opts.pool

dirpath = os.getcwd().split("code")[0]+"code/"
sys.path.append(dirpath)
from pre_process import clean_LC
our_process = np.load('/work/work_teamEXOPLANET/KOI_LC/cleaned/LC_kepler_processed.npy')

#### importante:
def prepare_lc_GAF(fluxs): #rango entre -1 y 1
    max_f = np.nanmax(fluxs)
    min_f = np.nanmin(fluxs)
    return (2*fluxs- max_f - min_f)/(max_f-min_f)

class GAF(object):
    def __init__(self, fluxs):
        self.X_cos = np.asarray(fluxs).astype('float32')
        I = np.ones(self.X_cos.shape[0], dtype='float32')
        self.X_sin = np.sqrt(np.clip(I - self.X_cos**2, 0, 1))
    
    def transform(self, method='sum'):
        if method == 'sum':
            return  np.tensordot(self.X_cos, self.X_cos, axes=-1) - np.tensordot(self.X_sin, self.X_sin, axes=-1)
        elif method == "diff":
            return  np.tensordot(self.X_sin, self.X_cos, axes=-1) - np.tensordot(self.X_cos, self.X_sin, axes=-1)
        
#T = our_process.shape[1]
#reduc = T/float(wind) ##########OJO 
#repeat = math.floor(np.log(reduc)/np.log(pool))

#model = Sequential()
#model.add(InputLayer(input_shape=(T,T)))
#model.add(Lambda(lambda x: K.expand_dims(x, axis=-1)))
#for _ in range(repeat):
#    model.add(AveragePooling2D(pool))
#model.add(Lambda(lambda x: K.squeeze(x, axis=-1)))  


def resize_image(img, w):
    # return model.predict(img[None,:,:])[0]
    aux = Image.fromarray(img)
    to_return = aux.resize((w,w), Image.NEAREST)
    aux.close()
    return np.asarray(to_return)


#file_name_S = "/work/work_teamEXOPLANET/GAF/GASF_%s.npy"%wind
#file_name_D = "/work/work_teamEXOPLANET/GAF/GADF_%s.npy"%wind

print("ATENCION, ESTA EJECUTANDO MULTIPLES SIZE DE RESHAPE, LEERA TODOS LOS DATOS SECUENCIALES, COMO PUNTO DE PARTIDA EL ULTIMO SIZE QUE PUSO EN LOS ARCHIVOS GUARDADOS!! NO EJECUTE MULTIPLES SIZE EN DIFERENTES CODIGOS")

file_aux = "/work/work_teamEXOPLANET/GAF/GASF_%s.npy"%winds[-1]
if os.path.isfile(file_aux):
    i = np.load(file_aux).shape[0]
else:
    i=0
print("Starting in >>> ",i)
    
for lc_our_detrend in our_process[i:]:
    print ("Procesando curva",i)
    fluxs = prepare_lc_GAF(lc_our_detrend)  #scale -1 y 1
    
    #mask_nans = np.isnan(fluxs)
    #fluxs[mask_nans] = 0 #or mask value
    
    model_gaf = GAF(fluxs)

    X_gasf = model_gaf.transform(method='sum')
    #X_gasf[mask_nans,:] = 0
    #X_gasf[:,mask_nans] = 0
    
    X_gadf = model_gaf.transform(method='diff')
    #X_gadf[mask_nans,:] = 0
    #X_gadf[:,mask_nans] = 0
        
        
    to_save_S = []
    to_save_D = []
    for wind in winds:
        print(">>> size ** %s ** ..."%wind,end='')
        file_name_S = "/work/work_teamEXOPLANET/GAF/GASF_%s.npy"%wind
        file_name_D = "/work/work_teamEXOPLANET/GAF/GADF_%s.npy"%wind
        X_gasf_res = resize_image(X_gasf, int(wind))

        X_gadf_res = resize_image(X_gadf, int(wind))
        
        if i ==0:
            X_total_gasf = X_gasf_res[None,:,:]
        else:
            X_total_gasf = np.load(file_name_S)[:i] #avoid write/read errors
            X_total_gasf = np.append(X_total_gasf, X_gasf_res[None,:,:], axis=0)
        to_save_S.append(X_total_gasf)
        
        if i ==0:
            X_total_gadf = X_gadf_res[None,:,:]
        else:
            X_total_gadf = np.load(file_name_D)[:i] #avoid write/read errors
            X_total_gadf = np.append(X_total_gadf, X_gadf_res[None,:,:], axis=0)
        to_save_D.append(X_total_gadf)
        del X_gasf_res, X_gadf_res, X_total_gadf, X_total_gasf
        print(" completado")
    del X_gasf, X_gadf #borrar originales y trabajar con las chicas solamente
    
    print(">>> Comienza a guardar archivos", end='')
    for wind, value_S, value_D  in zip(winds,to_save_S,to_save_D):
        file_name_S = "/work/work_teamEXOPLANET/GAF/GASF_%s.npy"%wind
        file_name_D = "/work/work_teamEXOPLANET/GAF/GADF_%s.npy"%wind
        np.save(file_name_S, value_S)
        np.save(file_name_D, value_D)
    print(" todos archivos guardados con exito!")
        
    del model_gaf, fluxs, to_save_S, to_save_D
    gc.collect()
    i+=1
print("TERMINADO!!!!!!!!")
