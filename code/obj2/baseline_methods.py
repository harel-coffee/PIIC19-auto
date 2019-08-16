import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os,sys, gc, time
from optparse import OptionParser
sys_out = sys.stdout

op = OptionParser()
op.add_option("-p", "--path", type="string", default="./", help="path to KOI metadata")
op.add_option("-t", "--Pprecision", type="float", default=1e-1, help="precision used on period ")
op.add_option("-s", "--Pstart", type="float", default=0.01, help="Period range start")
op.add_option("-e", "--Pend", type="float", default=2000, help="Period range end")
op.add_option("-m", "--method", type="string", default='', help="method choosed to calculate metadata (LS/BLS)")
op.add_option("-a", "--autom", type="int", default=0, help="automatic find period grid?")

(opts, args) = op.parse_args()
method = opts.method.lower()
autom = opts.autom==1
precision_needed = opts.Pprecision #1e-2 #ya es suficientemente pequeño 1e-3
periodstart = opts.Pstart
periodend = opts.Pend

folder = opts.path #"../../KOI_Data/"
folder_lc = "/work/work_teamEXOPLANET/KOI_LC/"

# Choose a period grid
periods = np.arange(periodstart, periodend, precision_needed )
ang_freqs = 2 * np.pi / periods
freqs = 1 / periods
sys_out.write("Period range, between %f - %f , %d values to explore\n"%(periods[0],periods[-1],periods.shape[0]))

dirpath = os.getcwd().split("code")[0]+"code/"
sys.path.append(dirpath)
from pre_process import clean_LC,generate_representation
from evaluation import calculate_metrics, evaluate_metadata, evaluate_metadata_raw
def impute_on_pandas(df):
    return df.fillna(df.median(),inplace=False)


####### READ METADATA
df_meta = pd.read_csv(folder+"/kepler_dataset.csv")
metadata_used = ["Period", "Duration", "Time of Transit Epoch",
                 "Inclination", "Semi-major Axis", "r/R",
                 "Teq", "Transit Number", "Limb Darkening Coeff1", "Limb Darkening Coeff2"]
df_meta_obj = df_meta[metadata_used]
df_meta_obj = impute_on_pandas(df_meta_obj)
mask_conf = (df_meta["NExScI Disposition"]=="CONFIRMED").values

df_sets = pd.read_csv(folder+"/koi_sets_unb.csv") 
mask_test = (df_sets["Set"] == "Test").values

mask_test_conf = mask_conf & mask_test
KOI_names_test = df_meta["KOI Name"].values[mask_test_conf]

###### READ LIGHT CURVE
time_kepler = np.load(folder_lc+"npy/KOI_LC_time.npy")
process_lc = np.load(folder_lc+'/cleaned/LC_kepler_processed.npy')

time_kepler = time_kepler[mask_test_conf]
process_lc = process_lc[mask_test_conf]
N, T = time_kepler.shape
sys_out.write("Datos: "+str(N)+","+str(T)+"\n")

for i in range(N):
    time_kepler[i], process_lc[i] = generate_representation(time_kepler[i], process_lc[i], plot=False)
coupled_lc = []
coupled_time = []
for i in range(N):
    mask_nan_aux = np.isnan(process_lc[i])
    coupled_lc.append(process_lc[i][~mask_nan_aux])    
    coupled_time.append(time_kepler[i][~mask_nan_aux])
X_lc_test = np.asarray(coupled_lc)
X_time_test = np.asarray(coupled_time)

if (method == "ls" or method == "lombscargle") and autom:
    name_saved_file = "./"+method+"_baseline_A.txt"
else:
    name_saved_file = "./"+method+"_baseline_"+str(precision_needed)+".txt"
if not os.path.isfile(name_saved_file):
    with open(name_saved_file, 'w') as saved_file:
        if method == 'ls' or method =='lombscargle':
            saved_file.write("KOI Name,Period,Time (sec)\n")
        elif method == 'bls' or method =='boxleast':
            saved_file.write("KOI Name,Period,Duration,Depth,Transit Time,Time (sec)\n")
        elif method == 'tls' or method =='transitleast':
            saved_file.write("KOI Name,Period,Duration,Depth,Time of Transit Epoch,r/R,Transit Number,Time (sec)\n")       
    #else otros metadatos
sys_out.flush()

aux_file = pd.read_csv(name_saved_file)
start_line = aux_file.shape[0]
if start_line == N:
    sys_out.write("Already done!")
    assert False
sys_out.write(("Starting on file with %d N, Executing "%(start_line)))

from astropy.timeseries import BoxLeastSquares, LombScargle
from transitleastsquares import transitleastsquares as TLS
from scipy.signal import lombscargle

if method == "ls" or method == "lombscargle":
    sys_out.write("Lomb-Scargle method\n")
    sys_out.flush()
    with open(name_saved_file, 'a', buffering=1) as saved_file:
        for i in range(start_line,N):
            sys_out.write("Do in :"+str(i)+"\n")
            iter_time = time.time()
            meta_pred = []

            ### PERIOD ESTIMATE --LOMB-SCARGLE Method
            if autom:
                model = LombScargle(X_time_test[i], X_lc_test[i], center_data=False)
                freqs_t, periodogram = model.autopower(method='fast', minimum_frequency=freqs.min(), maximum_frequency=freqs.max())
                periods = 1/freqs_t
            else:
                periodogram = lombscargle(X_time_test[i], X_lc_test[i], ang_freqs, normalize=False) 
            
            period_pred = periods[periodogram.argmax()] #periods[indx_max[0]] 

            save_data = [KOI_names_test[i], period_pred, time.time()-iter_time]
            save_data = map(str,save_data)
            save_data = ','.join(save_data)
            saved_file.write(save_data)
            saved_file.write("\n")
            sys_out.flush()
        sys_out.write("Completed prediction on test set\n")

elif method == "bls" or method == "boxleast":
    sys_out.write("Box LeastSquares method\n")
    sys_out.flush()
    with open(name_saved_file, 'a', buffering=1) as saved_file:
        for i in range(start_line,N):
            sys_out.write("Do in :"+str(i)+"\n")
            iter_time = time.time()
            
            ## Chose duration grid for BLS 
            dur_BLS = np.arange(periods.min()/100, periods.max()/10, precision_needed)
            sys_out.write("Cantidad de valores de Duration a explorar %d \n"%(dur_BLS.shape[0]))
            sys_out.flush()
         
            model = BoxLeastSquares(X_time_test[i], X_lc_test[i])
            results = model.autopower(dur_BLS) #already automatic
            periodogram = results.power
     
            period_pred = periods[periodogram.argmax()]
            transit_t_pred = results.transit_time[periodogram.argmax()]
            depth_pred = results.depth[periodogram.argmax()]
            dur_pred = results.duration[periodogram.argmax()]

            save_data = [KOI_names_test[i], period_pred, dur_pred,depth_pred,transit_t_pred,time.time()-iter_time]
            save_data = map(str,save_data)
            save_data = ','.join(save_data)
            saved_file.write(save_data)
            saved_file.write("\n")
            sys_out.flush()
        sys_out.write("Completed prediction on test set\n")
        
elif method == "tls" or method == "transitleast":
    sys_out.write("Transit LeastSquare method\n")
    sys_out.flush()
    with open(name_saved_file, 'a', buffering=1) as saved_file:
        for i in range(start_line,N):
            sys_out.write("Do in :"+str(i)+"\n")
            iter_time = time.time()
            
            model = TLS(X_time_test[i], X_lc_test[i]+1 ) #already automatic
            results = model.power(period_min=periods.min(), period_max=periods.max(), show_progress_bar=False)
            periodogram = results["power"]

            period_pred = results["periods"][periodogram.argmax()]
            dur_pred = results["duration"]
            depth_pred = results["depth_mean"][0]
            t0_pred = results["T0"]
            ratio_pred = results["rp_rs"]
            t_N_pred = results["transit_count"]

            save_data = [KOI_names_test[i], period_pred, dur_pred,depth_pred,t0_pred,ratio_pred,t_N_pred,time.time()-iter_time]
            save_data = map(str,save_data)
            save_data = ','.join(save_data)
            saved_file.write(save_data)
            saved_file.write("\n")
            sys_out.flush()
        sys_out.write("Completed prediction on test set\n")

    ### faltae l bayesiano
