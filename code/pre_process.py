"""
Clean light curves files
"""

from scipy.signal import medfilt, savgol_filter
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import TimeDelta
import math, os,sys

def median_filter(flux, width=25):
    """
        Calcula el "moving median filter" a través de la ventana width, steps de a 1
    """
    median_filter = np.zeros_like(flux)
    for i in range(flux.shape[0]):
        lim_inf = int(    i -(width/2))
        lim_sup = int((i+1) + (width/2))
        if lim_inf <0:
            lim_sup += np.abs(lim_inf)
            lim_inf = 0
        if lim_sup > flux.shape[0]:
            lim_inf -= (lim_sup-flux.shape[0])
            lim_sup = flux.shape[0]
            
        median_filter[i] = np.nanmedian(flux[ lim_inf: lim_sup ])
        if np.isnan(median_filter[i]):
            median_filter[i] = 0 #to not get nans
    return median_filter
    
def SavGol(y, win=151, return_filter=False):
    """
    Ajusta un filtro de polinomio de grado 2 sobre la curva y lo resta:
    sirve para quitar la tendencia
    """
    if len(y) <= win:
        print("BIG ERROR! short light curve")
        return
    y_used = np.asarray(y).copy()
    to_return = np.asarray(y).copy()
    mask_null = np.isnan(y)
    
    aux_y =y_used[~mask_null] #saltarse los nans
    #aux_y = np.nan_to_num(y)
    
    """ APPLIED FILTER and get residual"""
    filter_savgol = savgol_filter(aux_y, win, 2)
    aux_y = aux_y - filter_savgol + np.nanmedian(aux_y)
      
    to_return[~mask_null] = aux_y#[~mask_null]
    to_return[mask_null] = np.nan
    
    if return_filter:
        to_return2 = to_return.copy()
        to_return2[~mask_null] = filter_savgol#[~mask_null]
        to_return2[mask_null] = np.nan
        return to_return, to_return2
    return to_return


def clean_LC(flux, kernel_median=25, kernel_pol=151, savgol=True, median_t='subtract',plot=True):
    """
        1- Detrend light curve
        2- Remove outliers
        
        width debe ser definido en base al sample rate (10=5 horas)
    """
    med = np.nanmedian(flux)
    if  np.abs(med - 1) < 0.1:
        flux -= 1
    #elif med > 1:
    else:
        flux = flux/med - 1 #revisar en otros dataset.. si hacerlo despues de lc_savgol se realice..
        
    if kernel_median%2 ==0:
        kernel_median += 1
    if kernel_pol%2 == 0:
        kernel_pol +=1
    
    if savgol: #detrend encontrado + subtract
        flux = process_found(flux, kernel_median=kernel_median, kernel_pol=kernel_pol, plot_show=plot)
    
    if median_t=='subtract':
        lc_process = flux - median_filter(flux, width=kernel_median)
    elif median_t =='divide':
        lc_process = flux/median_filter(flux, width=kernel_median)
    elif median_t == 'smooth':
        lc_process = median_filter(flux, width=kernel_median)
    else:
        lc_process = flux

    return remove_outliers(lc_process)
    
def process_found(flux, kernel_median=25, kernel_pol=151, plot_show=True, divided=False):
    """ Proceso que se realiza en kepler, segun: Kepler: A search for extraterrestral planets """
    lc_SavGol, filter_app = SavGol(flux, win=kernel_pol, return_filter=True)
    if plot_show:
        plt.plot(flux[:1000], label= "Real LC")
        plt.plot(filter_app[:1000], label= "SavGol Filter -fited")
        plt.legend()
        plt.plot()
        plt.show()

        plt.plot(lc_SavGol[:1000])
        plt.title("First step, remove polynomial fit (detrend)")
        plt.plot()
        plt.show()
    #if not divided:
    #    lc_SavGol = lc_SavGol - median_filter(lc_SavGol,width=kernel_median)
    #else:
    #    lc_SavGol = lc_SavGol/median_filter(lc_SavGol,width=kernel_median)
    return lc_SavGol
    
def remove_outliers(f, sigm_up = 5, sigm_low=40, with_MAD=False, plot=True):
    '''
        Performs iterative sigma clipping to get outliers.
    '''
    f_clean = np.asarray(f).copy() #is already the residuals (value-median filter)
    
    do_clean = True
    values_cleaned = 0
    while do_clean and values_cleaned <= len(f_clean)*0.6: #do not remove more than 60% of data..
        if with_MAD:
            med = np.nanmedian(f_clean)
            MAD = 1.4826 * np.nanmedian(np.abs(f_clean - med)) 
        else:
            med = np.nanmean(f_clean)
            MAD = np.nanstd(f_clean)

        mask_nan = (f_clean - med < -sigm_low * MAD) | (f_clean - med > sigm_up * MAD)
        inds = np.where( mask_nan )[0]
        #remove outliers
        f_clean[inds] = np.nan
        
        if len(inds) == 0:
            do_clean = False #stop cleaning
        values_cleaned += len(inds)
    #if plot:
    print("Clean done (remove outliers iterativetly), erase %d values"%(values_cleaned))
    return f_clean

def mask_values(time, flux, mask=np.nan, plot=True):
    """"
        Toma una curva de luz sampleada a través de distintos timesteps y genera una serie continua (con máscara de nans)
        a través de un suevo sampling rate constante para cada tiempo. Calculado actualmente como el mínimo.
    """
    if plot:
        print("***************Mask values to get uniform sampling rate is being done..")
    flux = np.asarray(flux).copy()
    time = np.asarray(time).copy()
    #sample_rate = [ time[i+1] -time[i] for i in range(time.shape[0]-1) if not np.isnan(time[i+1] -time[i]) ]
    sample_rate = np.diff(time)
    
    idx_mins = np.argsort(sample_rate)
    new_samp_ra = sample_rate[idx_mins[0]]
    i = 1
    while new_samp_ra == 0:
        new_samp_ra = sample_rate[idx_mins[i]] #el mas pequeño
        i+=1   
        
    if sample_rate[idx_mins[-1]] - new_samp_ra < 0.000695: #menor que 1 min (60 sec)
        new_samp_ra = (time[-1] - time[0])/len(time) #si la diferencia es minima considerar que viene uniforme
    if plot:
        print("New sampling rate: %f (JD) --- %f (mins)"%(new_samp_ra,new_samp_ra*24*60))
    
        print("Old length: %d"%len(time))
    new_time = np.arange(time[0], time[-1], new_samp_ra)
    if plot:
        print("New length: %d"%len(new_time))
    if len(time) == len(new_time):
        if plot:
            print("Assuming uniform sampling")
        return new_time, flux
    
    #interpolar time
    #indices = np.arange(0,len(time))
    #not_nan = ~np.isnan(time)
    #time = np.interp(indices, indices[not_nan], time[not_nan])
    
    #definir cual sera al indice donde se colocara..
    new_flux = np.tile(mask, len(new_time))
    indx_hold = np.zeros(time.shape[0],dtype='int')
    i = 0
    j = 0
    while j < len(time):
        value = new_time[i]
        if  value - new_samp_ra < time[j] <= value + new_samp_ra: #quizas acá agregar el tema de la mediana..
            indx_hold[j] = i
            j+=1
        i+=1
    new_flux[indx_hold] = flux
    if plot:
        print("percentaje nulls/nans: %f"%( (len(new_flux) -len(flux))/len(new_flux) ))
    return new_time, new_flux


def median_view(x, y, bin_width, num_bins=None, x_min=None, x_max=None, plot=True):
    """Computes the median y-value in uniform intervals (bins) along the x-axis.
    The interval [x_min, x_max) is divided into num_bins uniformly spaced
    intervals of width bin_width. The value computed for each bin is the median
    of all y-values whose corresponding x-value is in the interval.
    """
    print("***************Median view is being done...")
    x_min = x_min if x_min is not None else x[0]
    x_max = x_max if x_max is not None else x[-1]
    x_len = len(x)
    if plot:
        print("Old length: %d"%x_len)
    
    num_bins = num_bins if num_bins is not None else int(round( (x_max - x_min) / bin_width))
    #bin_width = bin_width if bin_width is not None else (x_max - x_min) / num_bins
    bin_spacing = (x_max - x_min - bin_width) / (num_bins - 1)
    if plot:
        print("New length: %d"%num_bins)
    if x_len - num_bins <= 1:
        if plot:
            print("Nothing to be done") #que hacer en ese caso???? oversampling has to be done...
        return x,y

    # Bins with no y-values will fall back to the global median.
    result_y = np.repeat(np.median(y), num_bins)
    result_x = np.interp(np.arange(num_bins), np.arange(x_len), x) #np.repeat(0, num_bins)
    
    # Find the first element of x >= x_min. This loop is guaranteed to produce
    # a valid index because we know that x_min <= x[-1].
    x_start = 0
    while x[x_start] < x_min:
        x_start += 1

    # The bin at index i is the median of all elements y[j] such that
    # bin_min <= x[j] < bin_max, where bin_min and bin_max are the endpoints of bin i.
    bin_min = x_min  # Left endpoint of the current bin.
    bin_max = x_min + bin_width  # Right endpoint of the current bin.
    j_start = x_start  # Inclusive left index of the current bin.
    j_end = x_start  # Exclusive end index of the current bin.

    for i in range(num_bins):
        # Move j_start to the first index of x >= bin_min.
        while j_start < x_len and x[j_start] < bin_min:
            j_start += 1

        # Move j_end to the first index of x >= bin_max (exclusive end index).
        while j_end < x_len and x[j_end] < bin_max:
            j_end += 1

        if j_end > j_start:
            # Compute and insert the median bin value.
            result_y[i] = np.nanmedian(y[j_start:j_end])
            result_x[i] = np.mean(x[j_start:j_end])
        # Advance the bin.
        bin_min += bin_spacing
        bin_max += bin_spacing
    return result_x, result_y

def generate_representation(time, flux, sample_time = 0, kepler_view = False, plot=True): 
    """
        Transform light curve (time and flux) on a uniform sampling based on the lower value on delta time
        sample time has to be on BJD/JD
    """
    #GET UNIFORM SAMPLING
    new_time, new_flux = mask_values(time, flux, plot=plot)
  
    if kepler_view:
        sample_time = 0.020433599999961416  #GET VIEW OF KEPLER (30 min sampling rate)
        
    if sample_time != 0:
        new_time, new_flux = median_view(new_time, new_flux, bin_width=sample_time, plot=plot) # el de kepler
    return new_time, new_flux