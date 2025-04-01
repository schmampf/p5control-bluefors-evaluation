import numpy as np
from tqdm import tqdm
from torch.nn import Upsample
from torch import from_numpy

def bin_y_over_x(x, y, x_bins, upsampling=None):
    if upsampling is not None:
        k = np.full((2, len(x)), np.nan)
        k[0,:] = x
        k[1,:] = y
        m = Upsample(mode='linear', scale_factor=upsampling)
        big = m(from_numpy(np.array([k])))
        x = np.array(big[0,0,:])
        y = np.array(big[0,1,:])
    else:
        pass

    # Apply binning based on histogram function
    x_nu = np.append(x_bins, 2*x_bins[-1]-x_bins[-2])
    x_nu = x_nu - (x_nu[1] - x_nu[0])/2
        # Instead of N_x, gives fixed axis.
        # Solves issues with wider ranges, than covered by data
    _count, _ = np.histogram(x,
                             bins = x_nu,
                             weights=None)
    _count = np.array(_count, dtype='float64')
    _count[_count==0] = np.nan

    _sum, _ = np.histogram(x,
                           bins = x_nu,
                           weights = y)    
    return _sum/_count

def get_keys(
        file, 
        mkey, 
        i1=0, 
        i2=None, 
        to_pop=None,
        ):
    skeys=list(file["measurement"][mkey].keys())
    if to_pop is not None:
        skeys.remove(to_pop) # Remove no_field
    skeys = np.array(skeys, dtype='S20')
    float_keys = np.zeros(len(skeys))
    for i, k in enumerate(skeys):
        if i2 is not None:
            float_keys[i] = float(k[i1:i2])
        else:
            float_keys[i] = float(k[i1:])
    sorted_index = np.argsort(float_keys)
    skeys = list(skeys[sorted_index])
    y = float_keys[sorted_index]
    return y, skeys

def IV_mapping(
    file,
    mkey,
    skeys,
    y,
    V_min=-1e-3,
    V_max=1e-3,
    N_bins=1e3,
    V1_AMP=1e3,
    V2_AMP=1e3,
    R_REF=53e3,
    upsampling=None,
    ):    
    
    f_keyed = file["measurement"][mkey]
    V = np.linspace(V_min, V_max, int(N_bins)+1)

    I_up   = np.full((np.shape(y)[0], np.shape(V)[0]), np.nan, dtype='float64')
    I_down = np.full((np.shape(y)[0], np.shape(V)[0]), np.nan, dtype='float64')
    t      = np.full((np.shape(y)[0], 2),              np.nan, dtype='float64')

    for i, k in enumerate(tqdm(skeys)):
        offset= f_keyed[k]["offset"]["adwin"]
        off_v1 = np.nanmean(offset["V1"])
        off_v2 = np.nanmean(offset["V2"])

        sweep = f_keyed[k]["sweep"]["adwin"]

        t_temp = np.array(sweep['time'], dtype=('float', 'float'))
        t[i] = (t_temp[0], t_temp[-1])

        trigger = np.array(sweep['trigger'], dtype='int')
        v1 = np.array(sweep['V1'], dtype='float64')
        v2 = np.array(sweep['V2'], dtype='float64')

        v_raw = (v1 - off_v1) / V1_AMP
        i_raw = (v2 - off_v2) / V2_AMP / R_REF

        v_raw_up = v_raw[trigger==1]
        v_raw_down = v_raw[trigger==2]

        i_raw_up = i_raw[trigger==1]
        i_raw_down = i_raw[trigger==2]

        i_up   = bin_y_over_x(v_raw_up,   i_raw_up,   V, upsampling=upsampling)
        i_down = bin_y_over_x(v_raw_down, i_raw_down, V, upsampling=upsampling)

        I_up[i,:] = i_up
        I_down[i,:] = i_down

    dIdV_up   = np.gradient(I_up,   V, axis=1)
    dIdV_down = np.gradient(I_down, V, axis=1)

    
    return V, I_up, I_down, dIdV_up, dIdV_down, t

def T_mapping(
        t,
        file=None,
        temp_t=None,
        temp_T=None,
):
    if temp_t is None and temp_T is None:
        mcbj = file['status']['bluefors']['temperature']['MCBJ']
        temp_t = np.array(mcbj['time'])
        temp_T = np.array(mcbj['T'])
    else:
        temp_t = np.array(temp_t, dtype='float64')
        temp_T = np.array(temp_T, dtype='float64')
    
    T   = np.full(np.shape(t)[0], np.nan)
    N_T = np.full(np.shape(t)[0], np.nan)

    for i, t0 in enumerate(tqdm(t)): 
        i0 = np.argmin(np.abs(temp_t-t0[0]))
        i1 = np.argmin(np.abs(temp_t-t0[1]))
        T[i] = np.nanmean(temp_T[i0:i1])
        N_T[i] = i1-i0
        
    return T, N_T

"""
DECRAPTED
"""


def IV_T_mapping(
    file,
    mkey,
    skeys,
    y,
    V_min=-1e-3,
    V_max=1e-3,
    N_bins=1e3,
    V1_AMP=1e3,
    V2_AMP=1e3,
    R_REF=53e3,
    upsampling=None,
    ):    
    
    f_keyed = file["measurement"][mkey]
    V = np.linspace(V_min, V_max, int(N_bins)+1)

    I_up = np.zeros((np.shape(y)[0], np.shape(V)[0]), dtype='float64')
    I_down = np.zeros((np.shape(y)[0], np.shape(V)[0]), dtype='float64')
    t_start = np.zeros(np.shape(y)[0])
    t_stop = np.zeros(np.shape(y)[0])

    for i, k in enumerate(tqdm(skeys)):
        offset= f_keyed[k]["offset"]["adwin"]
        off_v1 = np.nanmean(offset["V1"])
        off_v2 = np.nanmean(offset["V2"])

        sweep = f_keyed[k]["sweep"]["adwin"]

        t = np.array(sweep['time'])
        t_start[i] = t[0]
        t_stop[i]  = t[-1]

        trigger = np.array(sweep['trigger'], dtype='int')
        v1 = np.array(sweep['V1'], dtype='float64')
        v2 = np.array(sweep['V2'], dtype='float64')

        v_raw = (v1 - off_v1) / V1_AMP
        i_raw = (v2 - off_v2) / V2_AMP / R_REF

        v_raw_up = v_raw[trigger==1]
        v_raw_down = v_raw[trigger==2]

        i_raw_up = i_raw[trigger==1]
        i_raw_down = i_raw[trigger==2]

        i_up   = bin_y_over_x(v_raw_up,   i_raw_up,   V, upsampling=upsampling)
        i_down = bin_y_over_x(v_raw_down, i_raw_down, V, upsampling=upsampling)

        I_up[i,:] = i_up
        I_down[i,:] = i_down

    dIdV_up   = np.gradient(I_up,   V, axis=1)
    dIdV_down = np.gradient(I_down, V, axis=1)

    mcbj = file['status']['bluefors']['temperature']['MCBJ']
    temp_t = np.array(mcbj['time'])
    temp_T = np.array(mcbj['T'])
    
    T_y = np.full(np.shape(y)[0], np.nan)
    N_T_y = np.full(np.shape(y)[0], np.nan)
    for i, t0 in enumerate(tqdm(t_start)): 
        i0 = np.argmin(np.abs(temp_t-t0))
        i1 = np.argmin(np.abs(temp_t-t_stop[i]))
        T_y[i] = np.nanmean(temp_T[i0:i1])
        N_T_y[i] = i1-i0
    
    return V, I_up, I_down, dIdV_up, dIdV_down, T_y, N_T_y




"""
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import nonparametric
from torch.nn import Upsample
from torch import from_numpy

def VI_binning(V, I, V_binned, frac = 1/500, plotter = True):

    # Apply LOWESS filter (performs weighted local linear fits) 
        # in V over I to get smooth and accurate data, else random extra features.
        # https://en.wikipedia.org/wiki/Local_regression
        # https://www.statsmodels.org/dev/examples/notebooks/generated/lowess.html
        # ATTENTION: long calculation time
    lowess = nonparametric.lowess
    z = lowess(V, I, frac=frac, return_sorted=True)

    # Apply fancy deep-learning upsampling method
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.upsample.html
        # upsampling factor 100
    k = np.array([[z]])
    m = Upsample(mode='bicubic', size = (k.shape[2]*100, 2))
    big = m(from_numpy(k))
    V_big = np.array(big[0,0,:,1])
    I_big = np.array(big[0,0,:,0])

    # Apply binning based on histogram function
    V_nu = np.append(V_binned, 2*V_binned[-1]-V_binned[-2])
    V_nu = V_nu - (V_nu[1] - V_nu[0])/2
        # Instead of N_x, gives fixed axis.
        # Solves issues with wider ranges, than covered by data
    _count, _ = np.histogram(V_big,
                             bins = V_nu,
                             weights=None)
    _count = np.array(_count, dtype='float64')
    _count[_count==0] = np.nan

    _sum, _ = np.histogram(V_big,
                           bins = V_nu,
                           weights = I_big)        
    I_binned = _sum/_count

    # Showcase the different processing steps
    if plotter:
        plt.figure(100)
        plt.plot(V, I, 'r.', ms=3, label='raw data')
        plt.plot(z[:,1], z[:,0], 'go', ms=5, label='lowess')
        plt.plot(V_big, I_big, 'b.', ms=1, label='upsampled')
        plt.plot(V_binned, I_binned, 'k+', ms = 5, label='binned')
        plt.grid()
        plt.legend()
        plt.xlabel('V [V]')
        plt.ylabel('I [A]')

    return I_binned
"""