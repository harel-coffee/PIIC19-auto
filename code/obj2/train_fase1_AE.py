import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os,sys, gc, keras, time
dirpath = os.getcwd().split("code")[0]+"code/"
sys.path.append(dirpath)
from pre_process import clean_LC,generate_representation

from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Dense, Flatten, MaxPool1D, Reshape, UpSampling1D, Lambda, AveragePooling1D
from keras.layers import GlobalAveragePooling1D,GlobalMaxPool1D, TimeDistributed, GRU,LSTM, RepeatVector
from keras.layers import BatchNormalization, Dropout, ZeroPadding1D, ZeroPadding2D, Cropping1D, Cropping2D, Conv2D, Conv2DTranspose, MaxPool2D,UpSampling2D
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
def conv_bloq(it, filters, kernel_s, pool, conv_pool=False, BN=False, drop=0,padding='same',dil_r=1, act='relu'):
    f1 = Conv1D(filters, kernel_s, strides=1, padding=padding, activation=act,dilation_rate=dil_r)(it)
    if BN:
        f1 = BatchNormalization()(f1)
    f1 = Conv1D(filters, kernel_s, strides=1, padding=padding, activation=act,dilation_rate=dil_r)(f1)
    if BN:
        f1 = BatchNormalization()(f1)
    if pool!= 0 and dil_r ==1:
        if conv_pool:
            f1 = Conv1D(filters, kernel_s, strides=pool, padding='valid')(f1)
            if BN:
                f1 = BatchNormalization()(f1)
        else:
            #f1 = MaxPool1D(pool_size=pool, strides=pool, padding='valid')(f1)
            f1 = AveragePooling1D(pool_size=pool, strides=pool, padding='valid')(f1)
    if drop != 0:
        f1 = Dropout(drop)(f1)
    return f1

def rnn_bloq(it, units, layers=1, bid=False, gru=True, drop=0):
    f1 = it
    for i in range(layers):
        if gru:
            layer_rec = GRU(units, return_sequences=True)#(i < layers - 1))
        else:
            layer_rec = LSTM(units, return_sequences=True)#(i < layers - 1))
        if bid:
            f1 = Bidirectional(layer_rec)(f1)
        else:
            f1 = layer_rec(f1)
        if drop > 0.0 and i > 0:  # skip these for first layer for symmetry
            f1 = Dropout(drop)(f1)
    return f1
        
def deconv_bloq(it, filters, kernel_s, pool, conv_pool=False, BN=False, drop=0,padding='same',dil_r=1,act='relu'):
    f1 = it
    if pool != 0 and dil_r ==1:
        if conv_pool:
            f1 = Conv2DTranspose(filters, (kernel_s,1), strides=(pool,1), padding=padding)(f1)
            if BN:
                f1 = BatchNormalization()(f1)
        else:
            f1 = Lambda(lambda x: K.expand_dims(x, axis=-2))(f1) #along channel axis-to process with conv2d
            f1 = UpSampling2D((pool,1))(f1)
            f1 = Lambda(lambda x: K.squeeze(x, axis=-2))(f1)
            
    f1 = Conv1D(filters, kernel_s, strides=1, padding=padding, activation=act, dilation_rate=dil_r)(f1)
    if BN:
        f1 = BatchNormalization()(f1)
    f1 = Conv1D(filters, kernel_s, strides=1, padding=padding, activation=act, dilation_rate=dil_r)(f1)
    if BN:
        f1 = BatchNormalization()(f1)
    if drop != 0:
        f1 = Dropout(drop)(f1)
    return f1

def encoder_model_CNN1D(input_dim, channels, L=1, filters=8,kernel_s =10,
                        pool=5,BN=False,conv_pool=False,drop=0,padding='same',dil_r=1,time=False  ): #parametros estructurales
    it = Input(shape=(input_dim,channels))  #fixed length..
    f1 = it
  
    for l in range(L):
        f1 = conv_bloq(f1, filters, kernel_s, pool,BN=BN,conv_pool=conv_pool, drop=drop,padding=padding,dil_r=dil_r) 
        
        if not time:
            filters = int(filters*2)
        if dil_r != 1:
            dil_r = int(dil_r*2)
    return Model(inputs=it, outputs=f1)


def decoder_model_CNN2D(input_dim, channels, T=0, L=1, filters=8,kernel_s=10,
                    pool=5,BN=False,conv_pool=False,drop=0,padding='same',dil_r=1): #parametros estructurales  
    it = Input(shape=input_dim)
    f1 = it
    filters = int(filters*2**(L-1))
    for _ in range(L):
        f1 = deconv_bloq(f1, filters,kernel_s, pool,conv_pool=conv_pool,BN=BN,drop=drop,padding=padding,dil_r=dil_r)
        
        filters = int(filters/2)
        if dil_r != 1:
            dil_r = int(dil_r*2) #o al igual que filtros dividir??
    out_x = Conv1D(channels, kernel_s, strides=1, padding=padding, activation='linear')(f1)
    
    if T != 0:
        T_model = K.int_shape(out_x)[1]
        delta_T = T - T_model
        padd_len = int(np.abs(delta_T/2))
        if np.abs(delta_T) % 2 ==0:
            left_pad = right_pad = padd_len                
        else:
            left_pad = padd_len+1
            right_pad = padd_len
            
        if delta_T > 0:
            out_x = ZeroPadding1D((0, delta_T))(out_x)
        elif delta_T < 0:
            out_x = Cropping1D((left_pad, right_pad))(out_x)
    return Model(inputs=it, outputs=out_x)

def mse_masked(y_true, y_pred):
    """ Masked on 0 value.."""
    mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    e = (y_true - y_pred)*mask
    return K.sum(K.square(e), axis=-1)  #dividir por el error??

def train_model(model,X,y,epochs=1,batch_size=32, initial_epoch=0):
    return model.fit(X,y, epochs=epochs, batch_size=batch_size,initial_epoch=initial_epoch)


folder = "../../KOI_Data/"
folder_lc = "/work/work_teamEXOPLANET/KOI_LC/"

time_kepler = np.load(folder_lc+"npy/KOI_LC_time.npy")
lc_kepler = np.load(folder_lc+"npy/KOI_LC_init.npy" )
process_lc = np.load(folder_lc+'/cleaned/LC_kepler_processed.npy')
N, T = lc_kepler.shape
for i in range(N):
    time_kepler[i], process_lc[i] = generate_representation(time_kepler[i], process_lc[i], plot=False)

#borrar nans... arreglo variable
coupled_lc = []
delta_time = []
lens_lc = []
for i in range(N):
    mask_nan = np.isnan(process_lc[i])
    coupled_lc.append(process_lc[i][~mask_nan])
    delta_time.append(np.hstack([[0],np.diff(time_kepler[i][~mask_nan])]))
    lens_lc.append(np.sum(~mask_nan))
coupled_lc = np.asarray(coupled_lc)
delta_time = np.asarray(delta_time)

#cada curva de luz dividirla por la desviación estandar..
coupled_lc_scaled = []
for i in range(coupled_lc.shape[0]):
    std_i = coupled_lc[i].std(keepdims=True)
    coupled_lc_scaled.append( coupled_lc[i]/std_i )
coupled_lc_scaled = np.asarray(coupled_lc_scaled)

#flip/mirror cada curva de luz
flip_coupled_lc_scaled = []
flip_delta_time = []
for i in range(coupled_lc.shape[0]):    
    flip_coupled_lc_scaled.append( coupled_lc_scaled[i][::-1] )
    flip_delta_time.append(  np.hstack([[0], delta_time[i][1:][::-1]]) )
flip_coupled_lc_scaled = np.asarray(flip_coupled_lc_scaled)
flip_delta_time = np.asarray(flip_delta_time)

delta_time = np.concatenate([flip_delta_time,delta_time])
coupled_lc_scaled = np.concatenate([flip_coupled_lc_scaled,coupled_lc_scaled])
del flip_delta_time, flip_coupled_lc_scaled
gc.collect()

## do padding with keras..
max_len = np.max(lens_lc)
X_time = keras.preprocessing.sequence.pad_sequences(delta_time,maxlen=max_len, value=0,dtype='float32',padding='post')
X_lc_scaled = keras.preprocessing.sequence.pad_sequences(coupled_lc_scaled,maxlen=max_len, value=0,dtype='float32',padding='post')
print("Shape with padding: ", X_lc_scaled.shape)

#need data with extra dim
X_lc_scaled = np.expand_dims(np.squeeze(X_lc_scaled),axis=-1)
X_time = np.expand_dims(np.squeeze(X_time),axis=-1)

T = X_lc_scaled.shape[1]
channels = 2

def op4(it_time,it_lc, T):
    it_all = keras.layers.Concatenate()([it_time,it_lc])
    
    encoder = encoder_model_CNN1D(T, 2, L=5, filters=8, pool=2,kernel_s=5, drop=0) #model
    encoder_model = Model([it_time,it_lc], encoder(it_all) )
    encoder_model.name = 'encoder'
    print("Parametros modelo: ",encoder_model.count_params())
    
    inp_dim = encoder_model.output_shape[1:] 
    decoder_model = decoder_model_CNN2D(inp_dim, 1, L=5, filters=8, pool=2, kernel_s=5,T=T, drop=0) #model
    decoder_model.name = "decoder"
    print("Parametros modelo: ",decoder_model.count_params())
    return encoder_model, decoder_model

def op4_doubleSimp(it_time,it_lc, T, time=False):
    encoder = encoder_model_CNN1D(T, 1, L=5, filters=8, pool=2,kernel_s=5, drop=0) #model
    encoder.name = "LC_encoder"
    if time:
        encoder_t = encoder_model_CNN1D(T, 1, L=5, filters=64, pool=2,kernel_s=5, drop=0, time=time) #o mas filtros?
    else:
        encoder_t = encoder_model_CNN1D(T, 1, L=5, filters=8, pool=2,kernel_s=5, drop=0, time=time)
    encoder_t.name = "Time_encoder"
    inp_dim = encoder.output_shape[1:] 

    encoder_lc = encoder(it_lc)
    encoder_time = encoder_t(it_time)
    concat_encoder = keras.layers.Concatenate()([encoder_time,encoder_lc]) #or multiply?
    encoder_model = Model(inputs=[it_time,it_lc],outputs=concat_encoder)
    encoder_model.name = "encoder"
    print("Parametros modelo: ",encoder_model.count_params())

    out_dim = encoder_model.output_shape[1:] 
    decoder_model = decoder_model_CNN2D(out_dim, 1, L=5, filters=8, pool=2, kernel_s=5,T=T, drop=0) #model
    decoder_model.name = "decoder"
    print("Parametros modelo: ",decoder_model.count_params())
    return encoder_model, decoder_model

#op 4 con cnn al medio?

def op4_GRU(it_time, it_lc, T, aux_time=False, time=False):
    encoder = encoder_model_CNN1D(T, 1, L=5, filters=8, pool=2,kernel_s=5, drop=0) #model
    encoder.name = "LC_encoder"
    if time:
        encoder_t = encoder_model_CNN1D(T, 1, L=5, filters=32, pool=2,kernel_s=5, drop=0, time=time) #o mas filtros?
    else:
        encoder_t = encoder_model_CNN1D(T, 1, L=5, filters=8, pool=2,kernel_s=5, drop=0, time=time)
    encoder_t.name= "Time_encoder"
    inp_dim = encoder.output_shape[1:] 

    encoder_lc = encoder(it_lc)
    encoder_time = encoder_t(it_time)
    concat_encoder = keras.layers.Concatenate()([encoder_time,encoder_lc]) #or multiply?
    f1_out = rnn_bloq(concat_encoder, units=inp_dim[1], layers=1, bid=False, gru=True, drop=0)
    encoder_model = Model(inputs=[it_time,it_lc],outputs=f1_out)
    encoder_model.name = "encoder"
    print("Parametros modelo: ",encoder_model.count_params())

    out_dim = encoder_model.output_shape[1:] 
    it_embd = Input(shape= out_dim)
    
    if aux_time:
        concat_decoder = keras.layers.Concatenate()([encoder_time, it_embd]) #or multiply?
        out_dim = (out_dim[0],out_dim[1] + encoder_t.output_shape[2])
    decoder = decoder_model_CNN2D(out_dim, 1, L=5, filters=8, pool=2, kernel_s=5,T=T, drop=0) #model
    if aux_time:
        decoder_model = Model(inputs=[it_time, it_embd], outputs=decoder(concat_decoder) )
    else:
        decoder_model = Model(inputs=it_embd, outputs=decoder(it_embd))
    decoder_model.name = "decoder"
    print("Parametros modelo: ",decoder_model.count_params())
    return encoder_model, decoder_model


from optparse import OptionParser
op = OptionParser()
op.add_option("-m", "--method", type="string", default='', help="architecture to train")
op.add_option("-e", "--epochs", type="int", default=20, help="epochs to train model")
(opts, args) = op.parse_args()

type_m = opts.method.lower()
epochs = opts.epochs

if os.path.isfile("./models/"+type_m+"_AE.h5"): ## check if file is created
    keras.losses.mse_masked = mse_masked
    autoencoder = load_model("./models/"+type_m+"_AE.h5")
    with open("./models/"+type_m+"_hist.txt","r") as f:
        loss_saved = [float(value.strip()) for value in f.readlines()]
    
else: #if not created
    it_time = Input(shape=X_time.shape[1:], name='time_input')
    it_lc = Input(shape=X_lc_scaled.shape[1:], name='lc_input')

    #if type_m == "op4":
    #    encoder_model, decoder_model = op4(it_time,it_lc, T)
    if type_m == "op4_d":
        encoder_model, decoder_model = op4_doubleSimp(it_time,it_lc, T) 
    elif type_m == "op4_dtsimple":
        encoder_model, decoder_model = op4_doubleSimp(it_time,it_lc, T, time=True) 
    #elif type_m == "op4_gru":
    #    encoder_model, decoder_model = op4_GRU(it_time,it_lc, T) 
    elif type_m == "op4_grut":
        encoder_model, decoder_model = op4_GRU(it_time,it_lc, T, aux_time=True) 
    elif type_m == "op4_grutsimple":
        encoder_model, decoder_model = op4_GRU(it_time,it_lc, T, aux_time=True, time=True) 
        
    loss_saved = []
    
    if "grut" in type_m:
        autoencoder = Model([it_time,it_lc],  decoder_model([it_time, encoder_model([it_time,it_lc])] )) 
    else:
        autoencoder = Model([it_time,it_lc],  decoder_model(encoder_model([it_time,it_lc]))) 
    autoencoder.compile(loss=[mse_masked],optimizer='adam')

## build autoencoder
init_ep = len(loss_saved)
print("Parametros modelo: ",autoencoder.count_params())

start_time = time.time()
hist = train_model(autoencoder, [X_time,X_lc_scaled], X_lc_scaled, 
                    batch_size=256, epochs=epochs+init_ep, initial_epoch=init_ep) 

autoencoder.save("./models/"+type_m+"_AE.h5")

new_loss_saved = hist.history["loss"]
with open("./models/"+type_m+"_hist.txt","a") as f:
    for value in new_loss_saved:
        f.write(str(value) + '\n')
        
print("Trained for %d epochs on %f hours"%(epochs,(time.time()-start_time)/3600))
