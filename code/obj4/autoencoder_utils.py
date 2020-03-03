import numpy as np

class Recon_eval():
    ### evaluar.. residuo + errores en reconstruccion
    def __init__(self):
        return
    
    def get_res(self,x,xhat):
        x = np.squeeze(x)
        xhat = np.squeeze(xhat)
        return x - xhat

    def RMSE_LC(self,x, xhat, res=[]):
        if len(res) == 0:
            res = self.get_res(x, xhat)
        return np.mean( np.sqrt(np.mean(np.square(res), axis=-1)) )

    def MSE_LC(self,x, xhat, res=[]):
        if len(res) == 0:
            res = self.get_res(x, xhat)
        return np.mean(np.mean( np.square(res), axis=-1))

    def MAE_LC(self,x, xhat, res=[]):
        if len(res) == 0:
            res = self.get_res(x, xhat)
        return np.mean( np.mean( np.abs(res), axis=-1))

    
    def autocorr_func(self,x):
        #https://en.wikipedia.org/wiki/Autocorrelation
        #https://stats.stackexchange.com/questions/24607/how-to-measure-smoothness-of-a-time-series-in-r
        #FATS: https://arxiv.org/pdf/1506.00010.pdf
        
        if np.sum(x) == 0:
            return 0
        inp = x - np.mean(x)
        result = np.correlate(inp, inp, mode='full')
        return result[len(x)-1:]/np.var(x, ddof=len(x)-1)

    def std_diff(self,x):
        #:::::::lag-one autocorrelation
        # scores near 1 imply a smoothly varying series
        # scores near 0 imply that there's no overall linear relationship between a data point and the following one (that is, plot(x[-length(x)],x[-1]) won't give a scatterplot with any apparent linearity)
        # scores near -1 suggest that the series is jagged in a particular way: if one point is above the mean, the next is likely to be below the mean by about the same amount, and vice
        
        if np.sum(x) == 0:
            return 0
        aux = np.abs(np.diff(x, axis=-1))
        return np.std(aux)#/np.mean(aux)

    def mean_diff(self,x):
        if np.sum(x) == 0:
            return 0
        aux = np.abs(np.diff(x, axis=-1))
        return np.mean(aux)#/np.mean(aux)
    
    def series_entropy(self,x, k=3,which='perm'):
        ### menor entropia indica mas suave (menos frecuencias de patrones)
        ### mayor entropía quiere decir que es puro ruido X= random()
        try:
            import entropy
            if which == 'perm':
                return entropy.perm_entropy(x, order=k, normalize=True)                 # Permutation entropy
            elif which =='spec':
                return entropy.spectral_entropy(x, 100, method='welch', normalize=True) # Spectral entropy
        except:
            raise Exception("You do not have installed entropy package: https://github.com/raphaelvallat/entropy")
    
import keras
from keras import backend as K
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0., stddev=1.)
    return z_mean + K.exp(0.5*z_log_var) * epsilon

def define_samp_model(D):
    mean_inp = keras.layers.Input(shape=(D,))
    log_var_inp = keras.layers.Input(shape=(D,))
    z = keras.layers.Lambda(sampling, output_shape=(D,), name='sample')([mean_inp, log_var_inp])
    return keras.models.Model([mean_inp, log_var_inp], z, name="norm_sampl")

def get_GRU(units, **args):
    ### Return GRU-layer on gpu or normal (compatible)
    try:
        from keras.layers import CuDNNGRU
        import tensorflow as tf
        GPU_AVAIL = tf.test.is_gpu_available() 
    except:
        GPU_AVAIL = False
    
    if GPU_AVAIL:
        layer = CuDNNGRU(units, **args)
    else:
        from keras.layers import GRU
        gru_args = {'reset_after':True, 'recurrent_activation':'sigmoid', 'implementation':2} 
        layer = GRU(units, **gru_args, **args)
    return layer

########## LAYERS ON SCALE ############
def check_dims(a,b):
    a_shape = list(K.int_shape(a))
    b_shape = list(K.int_shape(b))
    delta = len(a_shape) - len(b_shape)
    
    new_a, new_b = a, b
    if delta > 0:
        ### Aumenta la dimension del segundo valor "b"
        b_shape = [-1 if v ==None else v for v in b_shape] #replace None by -1 (in keras)
        new_dim = b_shape + [1]*delta
        new_b = K.reshape(b, new_dim)
        
    elif delta < 0:
        ### Aumenta la dimension del primer valor "a"
        a_shape = [-1 if v ==None else v for v in a_shape] #replace None by -1 (in keras)
        new_dim = a_shape + [1]*np.abs(delta)
        new_a = K.reshape(a, new_dim)
        
    return new_a, new_b

def Mul_L(args):
    x,s = args
    x,s = check_dims(x,s)
    return x*s

def Div_L(args):
    x,s = args
    x,s = check_dims(x,s)
    return x/s

def Norm_L(x, mu, std):
    x = K.log(x) #to normalize diifference on scale
    return (x-mu)/std

def RevertNorm_L(x, mu, std):
    return K.exp(x*std+mu)



###### FEATURES ANALYSIS
def corr_between(a,b):
    a_len = a.shape[1]
    return np.corrcoef(a, b, rowvar=False)[:a_len,a_len:]


from sklearn.feature_selection import mutual_info_regression as MI
def MI_between(A, B, k=3):
    MutI = np.zeros((A.shape[1],B.shape[1]))
    for d in range(B.shape[1]):
        MutI[:,d] = MI(A, B[:,d], n_neighbors=k)
    return MutI

def NMI_between(A, B, k=3, version=1):
    I_AB = MI_between(A,B, k=k)

    H_A = entropy_matrix(A, k=k)
    H_B = entropy_matrix(B, k=k)
    
    return_v = np.zeros((A.shape[1],B.shape[1])) #normalize
    for i in range(A.shape[1]):
        for j in range(B.shape[1]):    
            if version == 1:    
                joint_H = np.max([H_A[i], H_B[j]]) # max{ H(A), H(B)} as wikipedia shows or average, as sklearn shows
            elif version==2:
                joint_H = H_A[i] + H_B[j] - I_AB[i,j]
            return_v[i,j] = I_AB[i,j]/joint_H
    return return_v

def entropy_matrix(X, k=3):
    return np.asarray([ MI(X[:,column][:,None], X[:,column], n_neighbors=k) for column in range(X.shape[1])])    