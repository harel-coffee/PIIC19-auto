import numpy as np

############# DEFINE GRID #############
def det_state(a, b, n_sta):
    topes=np.linspace(a, b, n_sta+1)
    estados= []
    ind=1
    for top in topes[:-1]:
        estados.append((top,topes[ind]))
        ind+=1
    return estados
def det_state_2ways(a, b, n_sta_up, n_sta_low):
    estados_up = det_state(a, 0, n_sta_up)
    estados_low = det_state(0, b, n_sta_low)
    return estados_up + estados_low
############# DEFINE GRID #############


from numba import njit, jit
@njit(parallel=False, cache=True, fastmath=False)
def det_celda(num, estados): 
    for celda, (est_low, est_up) in enumerate(estados):
        if num<est_low and num>=est_up:
            return celda
        
    if (num >= estados[0][0]):
        return 0 #si se "sale" por arriba
    elif (num <= estados[-1][1]):
        return len(estados)-1 #si se "sale" por abajo
    else:
        return len(estados)

@njit(parallel=False, cache=True, fastmath=False)  #no se pudo paralelizar por el acceso a lista "ind -1" y el if
def add_transitions_gN(fluxs, transition_m, states, n):
    for ind in range(n, len(fluxs)):
        f_prev = fluxs[ind-n] # f(t-n)
        f_next = fluxs[ind]   # f(t)
        if (np.isnan(f_prev) or np.isnan(f_next)):
            continue
        init_s = det_celda(f_prev, states)
        fin_s = det_celda(f_next, states)
        transition_m[init_s, fin_s, n-1] += 1
    
@njit(parallel=True, cache=False, fastmath=True)
def count_MTF_gN(fluxs, states, N=1):
    #fluxs con nan donde no hay mediciones...
    n_sta = len(states)
    transition_m = np.zeros((n_sta, n_sta, N))
    for n in range(1,1+N): #a cada grado de "salto"
        add_transitions_gN(fluxs, transition_m, states, n)
    return transition_m
    
    



def create_MTF_gN(fluxs, n_up, n_down,N=1):
    states_values = det_state_2ways(1,-1, n_sta_up=n_up, n_sta_low=n_down) 

    transition_m = count_MTF_gN(fluxs, states_values, N=N)
    transition_m += 1 #priors
    transition_m = transition_m/transition_m.sum(axis=1,keepdims=True) #normalize
    return transition_m


