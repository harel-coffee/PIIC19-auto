try:
    import pysyzygy as ps 
    PS_AVAIL=True
except:
    print("Cannot import pysyzygy library ")
    PS_AVAIL=False
    
try:
    import pytransit 
    PYTRANSIT_AVAIL=True
except:
    print("Cannot import pytransit library ")
    PYTRANSIT_AVAIL=False
    
try:
    import batman 
    BATMAN_AVAIL=True
except:
    print("Cannot import batman library ")
    BATMAN_AVAIL=False
    
try:
    import pylightcurve as plc
    PYLC_AVAIL=True
except:
    print("Cannot import pylightcurve library ")
    PYLC_AVAIL=False
    
#import everest #-- basado en pysyzygy
#import ktransit

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('module://ipykernel.pylab.backend_inline') 

def get_MA_sim_names():
    return ["Period","T0","r/R", "a/R", "Inclination", "Impact Parameter", "Limb Darkening Coeff1","Limb Darkening Coeff2"]

MA_sim_names = get_MA_sim_names()
def simulate_MA(t, sim_dat, plot_m=True, lib = 'ps'):
    lib = lib.lower().strip()
    if plot_m:
        display(pd.DataFrame(sim_dat, index=[0], columns=MA_sim_names))
    
    per = sim_dat["Period"]
    a_R = sim_dat["a/R"]
    r_R = sim_dat["r/R"]
    t_0 = sim_dat["T0"] 
    
    u1 = sim_dat["Limb Darkening Coeff1"]
    u2 = sim_dat["Limb Darkening Coeff2"]
    
    #https://slideplayer.com/slide/6301069/21/images/7/Finally%2C+finding+transit+duration..jpg
    impact_p = sim_dat["Impact Parameter"] #bcirc: circular impact parameter
    inc = sim_dat["Inclination"]

    if lib == "ps":
        if PS_AVAIL:
            model = ps.Transit(per = per, RpRs = r_R, t0 = t_0, u1=u1,u2=u2, aRs= a_R, bcirc=impact_p,
                               maxpts=65000)
            try:
                lc_simulated = model(t) 
            except Exception as e:
                print("ERROR GENERACION DE CURVA: ",e)
                display(pd.DataFrame(sim_dat, index=[0], columns=MA_sim_names))
                lc_simulated = np.ones(len(t))
        else:
            raise Exception('You dont have available pysyzygy library! check: github.com/rodluger/pysyzygy') 
    
    elif lib =='pytransit':
        if PYTRANSIT_AVAIL:
            model = pytransit.QuadraticModel()
            model.set_data(t)

            lc_simulated = model.evaluate_ps(k = r_R, t0=t_0, p=per, a=a_R, i=inc*np.pi/180, ldc=[u1, u2])

            v = np.sum(lc_simulated)
            if np.isnan(v) or np.isinf(v):
                print("ERROR GENERACION DE CURVA: nans-generation(%s) // infs-generation(%s)"%(np.isnan(v),np.isinf(v)))
                display(pd.DataFrame(sim_dat, index=[0], columns=MA_sim_names))
                lc_simulated = np.ones(len(t))
        else:
            raise Exception('You dont have available pytransit library! check: github.com/hpparvi/PyTransit') 
            
    elif lib=='batman':
        if BATMAN_AVAIL:
            params = batman.TransitParams()       #object to store transit parameters
            params.t0 = t_0                        #time of inferior conjunction
            params.per = per                       #orbital period
            params.rp = r_R                      #planet radius (in units of stellar radii)--rp/rs
            params.a = a_R                     #semi-major axis (in units of stellar radii)
            params.inc = inc                      #orbital inclination (in degrees)
            params.limb_dark = "quadratic"        #limb darkening model
            params.u = [u1, u2]      #limb darkening coefficients [u1, u2, u3, u4]
            #dont known in kepler
            params.ecc = 0.                       #eccentricity
            params.w = 90.                        #longitude of periastron (in degrees)
            model = batman.TransitModel(params, t, nthreads = 1)    #initializes model

            lc_simulated = model.light_curve(params)
        else:
            raise Exception('You dont have available batman library! check: github.com/lkreidberg/batman') 
    
    elif lib=='pylc':
        if PYLC_AVAIL:
            lc_simulated = plc.transit('quad',  #claret son 4 coeficientes.. 'quad' or teh 'sqrt'
                               limb_darkening_coefficients = [u1,u2], 
                               rp_over_rs=r_R, 
                               period= per, 
                               sma_over_rs = a_R, 
                               inclination=inc, 
                               mid_time=t_0,
                               eccentricity=0,
                               periastron=90.,
                               time_array=t)

            v = np.sum(lc_simulated)
            if np.isnan(v) or np.isinf(v):
                print("ERROR GENERACION DE CURVA: nans-generation(%s) // infs-generation(%s)"%(np.isnan(v),np.isinf(v)))
                display(pd.DataFrame(sim_dat, index=[0], columns=MA_sim_names))
                lc_simulated = np.ones(len(t))
        else:
            raise Exception('You dont have available pylightcurve library! check: github.com/ucl-exoplanets/pylightcurve') 
            
    elif lib=='everest': #USA DURATION y depth
        dur = sim_dat["Duration"] 
        depth = sim_dat["Transit Depth"]
        lc_simulated = everest.transit.Transit(t, t0=t_0, per=per, dur=dur, depth=depth, 
                                               #todos los de ps
                                               #RpRs = r_R, 
                                               u1=u1,u2=u2, aRs= a_R, bcirc=impact_p,maxpts=65000)
           
    elif lib=='ktransit':
        model = ktransit.LCModel()
        model.add_star(ld1=u1,ld2=u2) # if only ld1 and ld2 are non-zero then a quadratic limb darkening law is used
        model.add_planet(T0=t_0,period=per, impact=impact_p, rprs=r_R)
        
        model.add_data(time=t)
        lc_simulated = model.transitmodel+1 # the out of transit data will be 0.0 unless you specify zpt

    return lc_simulated

def get_available_MA_methods():
    methods = []
    if PS_AVAIL:
        methods.append("ps")
    if PYTRANSIT_AVAIL:
        methods.append("pytransit")
    if BATMAN_AVAIL:
        methods.append("batman")
    if PYLC_AVAIL:
        methods.append("pylc")
    return methods