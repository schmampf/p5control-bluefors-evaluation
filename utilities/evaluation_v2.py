import numpy as np
from tqdm import tqdm
from torch.nn import Upsample
from torch import from_numpy

from  h5py import File
import matplotlib.pyplot as plt
from datetime import datetime

import warnings, os, platform

from .corporate_design_colors_v4 import cmap

"""
Pre-Definition of Functions
"""

def linfit(x):
    # time binned over voltage is not equally spaced and might have nans
    # this function does a linear fit to uneven spaced array, that might contain nans
    # gives back linear and even spaced array
    nu_x = np.copy(x)
    nans = np.isnan(x)
    not_nans = np.invert(nans)
    xx = np.arange(np.shape(nu_x)[0])
    poly = np.polyfit(xx[not_nans], 
                      nu_x[not_nans], 1)
    fit_x = xx * poly[0] + poly[1]
    return fit_x


def bin_y_over_x(
            x, 
            y,
            x_bins,
            upsampling=None,
        ):
        # gives y-values over even-spaced and monoton-increasing x
        # incase of big gaps in y-data, use upsampling, to fill those.
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
        return _sum/_count, _count

def bin_z_over_y(
        y,
        z,
        y_binned,
        ):
    N_bins = np.shape(y_binned)[0]
    counter = np.full(N_bins, 0)
    result  = np.full((N_bins, np.shape(z)[1]), 0, dtype='float64')
    
    # Find Indizes of x on x_binned
    dig = np.digitize(y, bins=y_binned)

    # Add up counter, I & dIdV
    for i, d in enumerate(dig):
        counter[d-1]   += 1
        result[d-1,:]  += z[i,:]
    
    # Normalize with counter, rest to np.nan
    for i,c in enumerate(counter):
        if c > 0:
            result[i,:] /= c
        elif c== 0:
            result[i,:] *= np.nan
    
    
    # Fill up Empty lines with Neighboring Lines
    for i,c in enumerate(counter):
        if c == 0: # In case counter is 0, we need to fill up
            up, down = i, i # initialze up, down
            while counter[up] == 0 and up < N_bins - 1: 
                # while up is still smaller -2 and counter is still zero, look for better up
                up += 1
            while counter[down] == 0 and down >= 1: 
                # while down is still bigger or equal 1 and coutner still zero, look for better down
                down -= 1

            if up == N_bins - 1 or down == 0:
                # Just ignores the edges, when c == 0
                result[i,:] *= np.nan
            else:
                # Get Coordinate System
                span = up - down
                relative_pos = i - down
                lower_span = span * .25
                upper_span = span * .75

                # Divide in same as next one and intermediate
                if 0 <= relative_pos <= lower_span:
                    result[i,:] =result[down,:]
                elif lower_span < relative_pos < upper_span:
                    result[i,:] = (result[up,:] +result[down,:]) / 2
                elif upper_span <= relative_pos <= span:
                    result[i,:] =result[up,:]
                else:
                    warnings.warn('something went wrong!')
    return result, counter

class EvaluationScript_v2():
    def __init__(self):
        
        system = platform.system()
        if system == 'Darwin':
            self.file_directory = '/Users/oliver/Documents/measurement_data/24 07 OI-24d-10/unbroken/'
        elif system == 'Linux':
            self.file_directory = '/home/oliver/Documents/measurement_data/24 07 OI-24d-10/unbroken/'
        else:
            warnings.warn(f'{system} needs a user path')
        self.file_name = ''
        self.file_folder = ''
        self.mkey = ''
        self.complete_file_name = ''

        self.V1_AMP = None
        self.V1_AMP = None
        self.R_REF = 51.689e3
        self.G_0 = 7.7481e-5 # Siemens, 1/Ohm

        self.V_min  = -1.8e-3
        self.V_max  = +1.8e-3
        self.N_bins = 900

        self.trigger_up = 1
        self.trigger_down = 2

        self.upsampling = None

        self.title = ''
        self.fig_nr = 0
        self.dpi = 300
        self.pdf = False
        self.cmap = 'viridis'
        self.cmap = cmap(color='seeblau', bad='gray')
        self.fig_folder = 'figures'

        self.indices = {
            'temperatures': [7, -3, 1e-6, 'no_heater'],
            'temperatures_up': [7, -2, 1e-6, 'no_heater'],
        }
    
    def show_measurements(self):
        file = File(f'{self.file_directory}{self.file_folder}{self.file_name}', 'r')
        print('Available Measurements:')
        liste = list(file['measurement'].keys())
        for li in liste:
            print(f"'{li}'")

    def set_measurement(self, mkey:str):
        try: 
            self.mkey = mkey
            self.complete_file_name = f'{self.file_directory}{self.file_folder}{self.file_name}'
            file = File(self.complete_file_name, 'r')
            self.keys = list(file['measurement'][self.mkey])
            print(f"Measurement: '{mkey}'")
        except KeyError:
            self.mkey = ''
            self.complete_file_name = ''
            self.keys = []

            print("ERROR: No such measurement.")

    def show_amplifications(self):
        if self.complete_file_name != '':
            print('Showing Femto Amplifications according to Status below.')
            file = File(self.complete_file_name, 'r')
            femto = file['status']['femto']
            time = femto['time']
            amp_A = femto['amp_A']
            amp_B = femto['amp_B']
            fig = plt.figure(1000, figsize=(6,2))
            plt.semilogy(time, amp_A, '-',  label='V1_AMP')
            plt.semilogy(time, amp_B, '--', label='V2_AMP')
            plt.legend()
            plt.title('Femto Amplifications according to Status')
            plt.xlabel('time')
            plt.ylabel('Amplification')

            print('TODO: implement datetime, probably two axes')

    def set_amplifications(
            self, 
            V1_AMP:float, 
            V2_AMP:float
            ):
        self.V1_AMP = V1_AMP
        self.V2_AMP = V2_AMP
        print(f'Set Amplification to {self.V1_AMP} & {self.V2_AMP}.')

    def show_keys(
            self
    ):
        print('Keys are for example: ')
        print(self.keys[:4])
        print(self.keys[:-5:-1])

    def pop_key(
            self,
            to_pop:str,
    ):
        if len(self.keys) != 0:
            self.keys.remove(to_pop) # Remove no_field etc.
            print(f"Remove: '{to_pop}'.")
        
    def get_y(
            self,
            index=None,
    ):
        if index is None:
            index = self.indices
        if index[self.mkey][3] in self.keys:
            self.keys.remove(index[self.mkey][3])

        y = []
        for i, key in enumerate(self.keys):
            temp = key[index[self.mkey][0]:index[self.mkey][1]]
            temp = float(temp) * index[self.mkey][2]
            y.append(temp)
        y = np.array(y)
        self.y_unsorted = y
        print('y-values are for example: ')
        print(y[:8])

    def set_V(
            self,
            V_abs = np.nan,
            V_min = np.nan,
            V_max = np.nan,
            N_bins = np.nan,
            ):
        if not np.isnan(V_abs):
            V_min = -V_abs
            V_max = +V_abs
        if not np.isnan(V_min):
            self.V_min = V_min
        if not np.isnan(V_max):
            self.V_max = V_max
        if not np.isnan(N_bins):
            self.N_bins = N_bins
        print(f'Set V_min = {self.V_min}, V_max = {self.V_max} & N_bins = {self.N_bins}.')

    def get_mapping_done(
            self,
    ):
        print('Start with mapping. Progress:')
        # Calculate new V-Axis
        self.V = np.linspace(
            self.V_min, 
            self.V_max, 
            int(self.N_bins)+1,
            )
        
        # Make shorter file name
        if self.complete_file_name != '':
            file = File(self.complete_file_name, 'r')
            f_keyed = file["measurement"][self.mkey]
        else:
            warnings.warn('no file yet!')
            return

        len_V = np.shape(self.V)[0]
        len_y = np.shape(self.y_unsorted)[0]

        # Initialize all values
        self.I_up = np.full((len_y, len_V), np.nan, dtype='float64')
        self.I_down = np.full((len_y, len_V), np.nan, dtype='float64')
        self.time_up = np.full((len_y, len_V), np.nan, dtype='float64')
        self.time_down = np.full((len_y, len_V), np.nan, dtype='float64')
        self.T_all_up = np.full((len_y, len_V), np.nan, dtype='float64')
        self.T_all_down = np.full((len_y, len_V), np.nan, dtype='float64')
        
        self.t_up_start = np.full(len_y, np.nan, dtype='float64')
        self.t_up_stop = np.full(len_y, np.nan, dtype='float64')
        self.t_down_start = np.full(len_y, np.nan, dtype='float64')
        self.t_down_stop = np.full(len_y, np.nan, dtype='float64')
        self.T_up   = np.full(len_y, np.nan, dtype='float64')
        self.T_down = np.full(len_y, np.nan, dtype='float64')
        self.off_V1 = np.full(len_y, np.nan, dtype='float64')
        self.off_V2 = np.full(len_y, np.nan, dtype='float64')
    
        # Iterate over Keys
        for i, k in enumerate(tqdm(self.keys)):
            # Retrieve Datasets
            offset= f_keyed[k]["offset"]["adwin"]
            sweep = f_keyed[k]["sweep"]["adwin"]
            temperature = f_keyed[k]["sweep"]["bluefors"]

            # Calculate Offsets
            self.off_V1[i] = np.nanmean(offset["V1"])
            self.off_V2[i] = np.nanmean(offset["V2"])

            # Get Voltage Readings of Adwin
            trigger = np.array(sweep['trigger'], dtype='int')
            time = np.array(sweep['time'], dtype='float64')
            v1 = np.array(sweep['V1'], dtype='float64')
            v2 = np.array(sweep['V2'], dtype='float64')

            # Calculate V, I
            v_raw = (v1 - self.off_V1[i]) / self.V1_AMP
            i_raw = (v2 - self.off_V2[i]) / self.V2_AMP / self.R_REF

            # Divide into up- and downsweep
            v_raw_up   = v_raw[trigger == self.trigger_up]
            v_raw_down = v_raw[trigger == self.trigger_down]
            i_raw_up   = i_raw[trigger == self.trigger_up]
            i_raw_down = i_raw[trigger == self.trigger_down]
            t_up    =  time[trigger == self.trigger_up]
            t_down  =  time[trigger == self.trigger_down]

            # Calculate Timepoints
            self.t_up_start[i]   = t_up[0]
            self.t_up_stop[i]    = t_up[-1]
            self.t_down_start[i] = t_down[0]
            self.t_down_stop[i]  = t_down[-1]

            # Bin that stuff
            i_up, _   = bin_y_over_x(v_raw_up,   i_raw_up,   self.V)
            i_down, _ = bin_y_over_x(v_raw_down, i_raw_down, self.V)
            time_up, _   = bin_y_over_x(v_raw_up,   t_up,    self.V)
            time_down, _ = bin_y_over_x(v_raw_down, t_down,  self.V)

            # Take care of time and Temperature
            temp_t = temperature['time']
            temp_T = temperature['Tsample']
            temp_t_up   = linfit(time_up)
            if temp_t_up[0] > temp_t_up[1]:
                temp_t_up = np.flip(temp_t_up)
            temp_t_down = linfit(time_down)
            if temp_t_down[0] > temp_t_down[1]:
                temp_t_down = np.flip(temp_t_down)
            T_up, _   = bin_y_over_x(temp_t, temp_T, temp_t_up,   upsampling=1000)
            T_down, _ = bin_y_over_x(temp_t, temp_T, temp_t_down, upsampling=1000)

            # Save to Array
            self.I_up[i,:]   = i_up
            self.I_down[i,:] = i_down
            self.time_up[i,:]   = time_up
            self.time_down[i,:] = time_down
            self.T_all_up[i,:]   = T_up
            self.T_all_down[i,:] = T_down

        # sorting afterwards, because of probably unknown characters in keys
        indices = np.argsort(self.y_unsorted)
        self.y_sorted = self.y_unsorted[indices]
        self.t_up_start = self.t_up_start[indices]
        self.t_up_stop = self.t_up_stop[indices]
        self.t_down_start = self.t_down_start[indices]
        self.t_down_stop = self.t_down_stop[indices]
        self.off_V1 = self.off_V1[indices]
        self.off_V2 = self.off_V2[indices]

        self.I_up = self.I_up[indices,:]
        self.I_down = self.I_down[indices,:]
        self.time_up = self.time_up[indices,:]
        self.time_down = self.time_down[indices,:]
        self.T_all_up = self.T_all_up[indices,:]
        self.T_all_down = self.T_all_down[indices,:]

        # calculating differential conductance
        self.dIdV_up   = np.gradient(self.I_up,   self.V, axis=1)/self.G_0
        self.dIdV_down = np.gradient(self.I_down, self.V, axis=1)/self.G_0

        # calculates self.T_up, self.T_down
        self.T_up   = np.nanmean(self.T_all_up,   axis=1)
        self.T_down = np.nanmean(self.T_all_down, axis=1)

    def map_over_T(
            self,
            T_min = 0,
            T_max = 2,
            N_bins = 2000+1,
    ):
        self.T_binned = np.linspace(T_min, T_max, N_bins)
        self.I_up_T, self.counter_up     = bin_z_over_y(self.T_up,   self.I_up,   self.T_binned)
        self.I_down_T, self.counter_down = bin_z_over_y(self.T_down, self.I_down, self.T_binned)
        self.dIdV_up_T,   _ = bin_z_over_y(self.T_up,   self.dIdV_up,   self.T_binned)
        self.dIdV_down_T, _ = bin_z_over_y(self.T_down, self.dIdV_down, self.T_binned)
        print('Mapping over T.')

    def show_map(
            self,
            x_key,
            y_key,
            z_key,
            x_lim = None,
            y_lim = None,
            z_lim = None,
            contrast = 1,
    ):  
        try:
            self.plot_keys = {
                'V_bias_up_muV':    [self.V*1e6,        r'$V_\mathrm{Bias}^\rightarrow$ (µV)'],
                'V_bias_up_mV':     [self.V*1e3,        r'$V_\mathrm{Bias}^\rightarrow$ (mV)'],
                'V_bias_up_V':      [self.V*1e0,        r'$V_\mathrm{Bias}^\rightarrow$ (V)'],
                'V_bias_down_muV':  [self.V*1e6,        r'$V_\mathrm{Bias}^\leftarrow$ (µV)'],
                'V_bias_down_mV':   [self.V*1e3,        r'$V_\mathrm{Bias}^\leftarrow$ (mV)'],
                'V_bias_down_V':    [self.V*1e0,        r'$V_\mathrm{Bias}^\leftarrow$ (V)'],
                'heater_power_muW': [self.y_sorted*1e6, r'$P_\mathrm{Heater}$ (µW)'],
                'heater_power_mW':  [self.y_sorted*1e3, r'$P_\mathrm{Heater}$ (mW)'],
                'T_all_up_mK':      [self.T_all_up*1e3, r'$T_{Sample}$ (mK)'],
                'T_all_up_K':       [self.T_all_up*1e0, r'$T_{Sample}$ (K)'],
                'T_up_mK':          [self.T_up*1e3,     r'$T_\mathrm{Sample}$ (mK)'],
                'T_up_K':           [self.T_up*1e0,     r'$T_\mathrm{Sample}$ (K)'],
                'T_binned_up_K':    [self.T_binned,     r'$T_\mathrm{Sample}^\rightarrow$ (K)'],
                'T_binned_down_K':  [self.T_binned,     r'$T_\mathrm{Sample}^\leftarrow$ (K)'],
                'dIdV_up':          [self.dIdV_up,      r'd$I/$d$V$ ($G_0$)'],
                'dIdV_up_T':        [self.dIdV_up_T,    r'd$I/$d$V$ ($G_0$)'],
                'dIdV_down':        [self.dIdV_down,    r'd$I/$d$V$ ($G_0$)'],
                'dIdV_down_T':      [self.dIdV_down_T,  r'd$I/$d$V$ ($G_0$)'],
                'time_up':          [self.time_up,      r'time']
            }
        except AttributeError:
            print('Value not found. Please calculate first!')
            return
        

        try:
            self.x = self.plot_keys[x_key][0]
            self.x_label = self.plot_keys[x_key][1]
            self.y = self.plot_keys[y_key][0]
            self.y_label = self.plot_keys[y_key][1]
            self.z = self.plot_keys[z_key][0]
            self.z_label = self.plot_keys[z_key][1]
        except KeyError:
            print('Key not found. Choose from:')
            for key in self.plot_keys.keys():
                print(f"'{key}")
            return

        self.x_key = x_key
        self.y_key = y_key
        self.z_key = z_key

        if self.z.dtype == np.dtype('int32'):
            warnings.warn("img is integer. Sure?")

        stepsize_x=np.abs(self.x[-1]-self.x[-2])/2
        stepsize_y=np.abs(self.y[-1]-self.y[-2])/2
        if x_lim is None:
            x_ind = [0, -1]
        else:
            if x_lim[0] >= x_lim[1]:
                warnings.warn('First x_lim must be smaller than first one.')
                return
            x_ind = [np.abs(self.x-x_lim[0]).argmin(),
                    np.abs(self.x-x_lim[1]).argmin()]
        if y_lim is None:
            y_ind = [0, -1]
        else:
            if y_lim[0] >= y_lim[1]:
                warnings.warn('First y_lim must be smaller than first one.')
                return
            y_ind = [np.abs(self.y-y_lim[0]).argmin(),
                    np.abs(self.y-y_lim[1]).argmin()]
        ext = [self.x[x_ind[0]]-stepsize_x,
               self.x[x_ind[1]]+stepsize_x,
               self.y[y_ind[0]]-stepsize_y,
               self.y[y_ind[1]]+stepsize_y]
        self.z = self.z[y_ind[0]:y_ind[1],
                        x_ind[0]:x_ind[1]]
        self.x = self.x[x_ind[0]:x_ind[1]]
        self.y = self.y[y_ind[0]:y_ind[1]]

        if z_lim is None:
            z_lim = [np.nanmean(self.z)-np.nanstd(self.z)/contrast, 
                     np.nanmean(self.z)+np.nanstd(self.z)/contrast]

        plt.close(self.fig_nr)
        self.fig, (self.ax_z, self.ax_c) = plt.subplots(
            num=self.fig_nr,
            ncols=2,
            figsize=(6,4),
            dpi=self.dpi,
            gridspec_kw={"width_ratios":[5.8,.2]},
            constrained_layout=True
            )

        im = self.ax_z.imshow(self.z, 
                        extent=ext, 
                        aspect='auto',
                        origin='lower',
                        clim=z_lim,
                        cmap=self.cmap,
                        interpolation='none')
        self.ax_z.set_xlabel(self.x_label)
        self.ax_z.set_ylabel(self.y_label)
        self.ax_z.ticklabel_format(
            axis="both", 
            style="sci", 
            scilimits=(-3,3),
            useMathText=True
        )
        self.ax_z.tick_params(direction='in')

        cbar = self.fig.colorbar(im, label=self.z_label, cax=self.ax_c)
        self.ax_c.tick_params(direction='in')
        lim = self.ax_z.set_xlim(ext[0],ext[1])
        lim = self.ax_z.set_ylim(ext[2],ext[3])

        if self.title is not None:
            plt.suptitle(self.title)
        else:
            plt.suptitle(self.mkey)


    def save_figure(
        self,
        ):
        # Handle figure folder
        check = os.path.isdir(self.fig_folder)
        if not check:
            os.makedirs(self.fig_folder)

        # Handle Title
        title = f"{self.title}"

        # Save Everything
        name = os.path.join(os.getcwd(), self.fig_folder, title)
        self.fig.savefig(f'{name}.png')
        if self.pdf: # save as pdf
            self.fig.savefig(f'{name}.pdf', dpi=600)
        print(f"Figure is saved under: {self.fig_folder}/{title}.png")


    def get_dict_keys(self):   
        ignore = ['fig', 'ax_z', 'ax_c', 'cmap']
        keys = []
        for key in self.__dict__.keys():
            if key not in ignore:
                keys.append(key)
        return keys

    def get_keys(self):
        key = 'self.'+key
        dict_keys = self.get_dict_keys()
        dictionary = {}
        for key in dict_keys:
            dictionary[key] = globals()[key]
        return dictionary
