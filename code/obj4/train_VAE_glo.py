import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os,sys, gc, keras, time
dirpath = os.getcwd().split("code")[0]+"code/"
sys.path.append(dirpath)
from pre_process import clean_LC,generate_representation
from evaluation import calculate_metrics, evaluate_metadata, evaluate_metadata_raw

from optparse import OptionParser
op = OptionParser()
op.add_option("-T", "--times", type="int", default=200, help="CANTIDAD DE PUNTOS EN LA CURVA GLOBAL")
op.add_option("-D", "--latentdim", type="int", default=32, help="Latent dim")
op.add_option("-l", "--lambd", type="float", default=0.01, help="KL weight")
op.add_option("-e", "--epochs", type="int", default=50, help="epochs to train model")
op.add_option("-p", "--path", type="string", default="../../", help="path to koi data")

(opts, args) = op.parse_args()
Tim = opts.times
latent_dim = opts.latentdim
l = opts.lambd
EPOCHS = opts.epochs

folder = opts.path+"/KOI_Data/"
folder_lc = "/work/work_teamEXOPLANET/KOI_LC/"

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import *
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
def impute_on_pandas(df):
    return df.fillna(df.median(),inplace=False)

df_meta = pd.read_csv(folder+"/kepler_dataset.csv")
all_labels = df_meta["NExScI Disposition"]
mask_conf = (all_labels=="CONFIRMED").values
mask_fp = (all_labels=="FALSE POSITIVE").values
mask_cand = (all_labels=="CANDIDATE").values

metadata_used = ["Period", "Duration", "Time of Transit Epoch", "r/R", "a/R",
                 "Inclination", "Limb Darkening Coeff1", "Limb Darkening Coeff2",
                 "Teq", "Fitted Stellar Density",
                "Teff","Stellar Radius", "Stellar Mass"]
#dividir duration por 24??
df_meta["Duration"] = df_meta["Duration"]/24
df_meta_obj = df_meta[metadata_used]
mask_nan = pd.isna(df_meta_obj)
df_meta_obj = impute_on_pandas(df_meta_obj)

df_sets = pd.read_csv(folder+"/koi_sets.csv") 
mask_train = (df_sets["Set"] == "Train").values
mask_test = (df_sets["Set"] == "Test").values
mask_unlabeled = (df_sets["Set"] == "Unlabeled").values


time_kepler = np.load(folder_lc+"npy/KOI_LC_time.npy")
#lc_kepler = np.load(folder_lc+"npy/KOI_LC_init.npy" )
process_lc = np.load(folder_lc+'/cleaned/LC_kepler_processed.npy')
N, T = time_kepler.shape
print((N,T))

#borrar nans... arreglo variable
coupled_lc = []
coupled_time = []
delta_time = []
for i in range(N):
    mask_nan_aux = np.isnan(process_lc[i])
    coupled_lc.append(process_lc[i][~mask_nan_aux])
    
    time_a = time_kepler[i][~mask_nan_aux]
    # calculate delta time --> this could be done after padding is done..
    delta_time.append(np.hstack([[0],np.diff(time_a)]))
    coupled_time.append(time_a)
    
coupled_lc = np.asarray(coupled_lc)
coupled_time = np.asarray(coupled_time)
delta_time = np.asarray(delta_time)

coupled_lc_scaled = coupled_lc

### data augm
flip_coupled_lc_scaled = []
flip_delta_time = []
flip_time = []
for i in range(coupled_lc.shape[0]):    
    flip_coupled_lc_scaled.append( coupled_lc_scaled[i][::-1] )
    flip_delta_time.append(  np.hstack([[0], delta_time[i][1:][::-1]]) )
    flip_time.append( coupled_time[i][0] + np.cumsum(flip_delta_time[-1]))
    
flip_coupled_lc_scaled = np.asarray(flip_coupled_lc_scaled)
flip_delta_time = np.asarray(flip_delta_time)
flip_time = np.asarray(flip_time)

delta_time = np.concatenate([delta_time, flip_delta_time])
coupled_lc_scaled = np.concatenate([coupled_lc_scaled, flip_coupled_lc_scaled])
coupled_time = np.concatenate([coupled_time, flip_time])

del flip_delta_time, flip_coupled_lc_scaled, flip_time
gc.collect()

from fold import *

import os, sys
dirpath = os.getcwd().split("code")[0]+"code/"
sys.path.append(dirpath)
from pre_process import *
import pysyzygy as ps
T_aux = 800
mask_train = np.zeros(N, dtype='bool')
for n in range(N):
    nflag, ds, sflag = df_meta[["Transit Flag","Disposition Score","Secondary Flag"]].values[n]

    if not nflag and not sflag: #borrar seq eclipse??
        if ds > 0.55: #mayor o igual a 0.3 minimo  (o 0.55)
            mask_train[n] = True
        else:
            #usar Mandel-Agol
            per, t0, dur = df_meta_obj[["Period", "Time of Transit Epoch","Duration"]].values[n]
            t = coupled_time[n]
            y = coupled_lc_scaled[n]

            t_fold, val_fold = phase_fold_and_sort_light_curve(t, y, period= per, t0= t0)
            val_glo = global_view(t_fold, val_fold, period=per, num_bins=T_aux, bin_width_factor=1 / T_aux)
            
            metadata_n = df_meta.iloc[n]
            period = metadata_n["Period"]
            duration = metadata_n["Duration"]/24
            a_R = metadata_n["a/R"]
            r_R = metadata_n["r/R"]
            inclination = metadata_n["Inclination"]
            u1 = metadata_n["Limb Darkening Coeff1"]
            u2 = metadata_n["Limb Darkening Coeff2"]
            
            t_glo = np.cumsum(np.concatenate([[0], np.tile(per/T_aux, T_aux-1)]) )
            t_0 = t_glo[int(np.round(T_aux/2))]
            model = ps.Transit(per = period, RpRs = r_R, t0 = t_0, u1=u1,u2=u2, aRs= a_R, maxpts=65000) 
            try:
                lc_simulated = model(t_glo)-1 #primer error/contratiempo encontrado (TIEMPO)
            except:
                print("ERROR, CURVA ELIMINADO DE INMEDIATO")
                continue

            mse_resid = np.sqrt(np.nanmean(np.square(val_glo - lc_simulated)))
            mse_resid /= np.std(val_glo)
            if mse_resid < 1:
                mask_train[n] = True
    else:
        pass #ya se elimina
print("Cantidad de datos dejados, ",np.sum(mask_train))

def std_scaler(x):
    mu = np.nanmean(x, axis=-1, keepdims=True)
    std = np.nanstd(x, axis=-1, keepdims=True)+1e-9
    return (x- mu)/std

def minmax_scaler(x, max_v,min_v):
    scale = 1 / (max_v - min_v)
    return scale * x - min_v * scale

#pre-process light curve
def set_scaler(x):
    #return x #o +1
    return std_scaler(x)

X_fold_lc = []
X_fold_lc_aug = [] #con curvas invertidas
X_fold_time = []
for n in range(N):
    if n%500 == 0:
        print("Va en el, ",n)
        
    if mask_train[n]: 
        per, t0, dur = df_meta_obj[["Period", "Time of Transit Epoch","Duration"]].values[n]
        t = coupled_time[n]
        y = coupled_lc_scaled[n]

        t_fold, val_fold = phase_fold_and_sort_light_curve(t, y, period= per, t0= t0)

        val_glo = global_view(t_fold, val_fold, period=per, num_bins=Tim, bin_width_factor=1 / Tim)
        #preprocess global..
        val_glo = set_scaler(val_glo)

        X_fold_lc.append(val_glo)

        #dos opciones para revertir
        t_fold_aug = t_fold#*-1
        val_fold_aug = val_fold[::-1]

        val_glo = global_view(t_fold_aug, val_fold_aug, period=per, num_bins=Tim, bin_width_factor=1 / Tim)
        #preprocess global..
        val_glo = set_scaler(val_glo)

        X_fold_lc_aug.append(val_glo)

        #t_glo = np.linspace(0, per, T)
        t_glo_diff = np.concatenate([[0], np.tile(per/Tim, Tim-1)])  #np.diff(t_glo)
        X_fold_time.append(t_glo_diff) #differences values
    
X_fold_lc = np.asarray(X_fold_lc)
X_fold_lc_aug = np.asarray(X_fold_lc_aug)
X_fold_time = np.asarray(X_fold_time)

mask_train = np.random.rand(X_fold_lc.shape[0])<0.75
mask_test = ~mask_train

X_train = np.concatenate([X_fold_lc[mask_train], X_fold_lc_aug[mask_train]], axis=0)
X_train_t = np.concatenate([X_fold_time[mask_train], X_fold_time[mask_train]], axis=0)

X_test = X_fold_lc[mask_test]
X_test_t = X_fold_time[mask_test]

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

X_train_t = np.expand_dims(X_train_t, axis=-1)
X_test_t = np.expand_dims(X_test_t, axis=-1)

print("X train shape: ",X_train.shape)
print("X train time shape: ",X_train_t.shape)
print("X test shape: ",X_test.shape)
_, channels = X_train.shape[1:]


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0., stddev=1.)
    return z_mean + K.exp(0.5*z_log_var) * epsilon

def KL_loss(y_true, y_pred):
    return - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

#### IF THIS IS USED,, KL WEIGHT NEEDED TO BE LOW ######
def MSE_loss(y_true, y_pred):
    v = K.mean( K.square( y_true - y_pred ), axis=1) #on time axis..    
    return K.flatten(v)

#### HAS TO BE USED IN ORDER TO LEARN #######
def SSE_loss(y_true, y_pred):
    v = K.sum( K.square( y_true - y_pred ), axis=1) #on time axis..    
    return K.flatten(v)

def vae_loss(y_true, y_pred):
    return SSE_loss(y_true, y_pred) + l*KL_loss(y_true, y_pred)


x_lc = Input(shape = (Tim, channels) , name="LC_inp")
x_t = Input(shape = (Tim, channels) , name ="T_inp")

x = Concatenate(axis=-1)([x_t, x_lc]) #add time encode

f1 = Bidirectional(GRU(64, return_sequences=True))(x) #bidirectional
f1 = Bidirectional(GRU(64, return_sequences=False))(f1) #bidirectional

z_mean = Dense(latent_dim,activation='linear')(f1)
z_log_var = Dense(latent_dim,activation='linear')(f1)

encoder = Model([x_t, x_lc], z_mean, name="encoder")
z = Lambda(sampling, output_shape=(latent_dim,), name='sample')([z_mean, z_log_var])
samp_encoder = Model([x_t, x_lc], z, name="encoder_sampling")


decoder_input = Input(shape=(latent_dim,))

#decode = Dense(latent_dim, activation='relu')(decoder_input) ## o sacar??

decode = RepeatVector(Tim)(decoder_input)

decode = Concatenate(axis=-1)([x_t, decode]) #add time decode

decode = Bidirectional(GRU(64, return_sequences=True))(decode) #bidirectional
decode = Bidirectional(GRU(64, return_sequences=True))(decode) #bidirectional

decode = TimeDistributed(Dense(1, activation='linear'))(decode)
generator = Model([x_t, decoder_input], decode, name="generator")

# instantiate VAE model
out = generator([x_t, samp_encoder([x_t, x_lc]) ])
vae = Model([x_t, x_lc], out)

vae.compile(optimizer='adam', loss=vae_loss, metrics = [KL_loss, MSE_loss])
vae.summary()

start_time = time.time()
batch_size = 64
vae.fit([X_train_t,X_train], X_train, epochs=EPOCHS, batch_size=batch_size, 
        validation_data=([X_test_t,X_test], X_test))

vae.save_weights("./"+str(Tim)+"T_"+str(latent_dim)+"D_"+str(l)+"l_"+str(EPOCHS)+"e_VAE_glo.h5")

print("Trained for %d epochs on %f hours"%(EPOCHS,(time.time()-start_time)/3600))
