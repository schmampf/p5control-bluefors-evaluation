import numpy as np
import matplotlib.pyplot as plt

import warnings, os, platform, logging, h5py, pickle

from tqdm import tqdm
from PIL import Image
from scipy import constants
from scipy.signal import savgol_filter

from .corporate_design_colors_v4 import cmap
from .evaluation_helper_v2 import *

from importlib import reload
logger = logging.getLogger(__name__)
reload(logging)
logging.basicConfig(level=logging.INFO, format='%(message)s')

class EvaluationScript_v2():
    def __init__(
            self,
            name = 'eva',
            ):
        self._name = name

        system = platform.system()
        if system == 'Darwin':
            self.file_directory = '/Users/oliver/Documents/measurement_data/'
        elif system == 'Linux':
            self.file_directory = '/home/oliver/Documents/measurement_data/'
        else:
            logger.warning(f'{system} needs a user path')
            self.file_directory = './'

        self.file_name = ''
        self.file_folder = ''
        self.mkey = ''

        self.fig_folder = 'figures/'
        self.data_folder = 'data/'
        self.sub_folder = ''

        self.V1_AMP = None
        self.V1_AMP = None
        self.R_REF = 51.689e3 # Ohm

        self.V_min  = -1.8e-3
        self.V_max  = +1.8e-3
        self.V_bins = 900

        self.T_min = 0
        self.T_max = 2
        self.T_bins = 2000

        self.trigger_up = 1
        self.trigger_down = 2

        self.upsampling = None
        self.window_length = 20
        self.polyorder = 2

        self.title = ''
        self.fig_nr = 0
        self.display_dpi = 100
        self.png_dpi = 600
        self.pdf_dpi = 600
        self.pdf = False
        self.cmap = cmap(color='seeblau', bad='gray')
        self.contrast = 1        

        self.indices = {
            'temperatures': [7, -3, 1e-6, 'no_heater'],
            'temperatures_up': [7, -2, 1e-6, 'no_heater'],
            'gate_voltages': [5, -2, 1e-3, 'no_gate'],
        }

        self.plot_keys = {
                'y_axis':             ['self.y_axis',             r'$y$ (arb. u.)'],
                'V_bias_up_muV':      ['self.V_axis*1e6',         r'$V_\mathrm{Bias}^\rightarrow$ (µV)'],
                'V_bias_up_mV':       ['self.V_axis*1e3',         r'$V_\mathrm{Bias}^\rightarrow$ (mV)'],
                'V_bias_up_V':        ['self.V_axis*1e0',         r'$V_\mathrm{Bias}^\rightarrow$ (V)'],
                'V_bias_down_muV':    ['self.V_axis*1e6',         r'$V_\mathrm{Bias}^\leftarrow$ (µV)'],
                'V_bias_down_mV':     ['self.V_axis*1e3',         r'$V_\mathrm{Bias}^\leftarrow$ (mV)'],
                'V_bias_down_V':      ['self.V_axis*1e0',         r'$V_\mathrm{Bias}^\leftarrow$ (V)'],
                'heater_power_muW':   ['self.y_axis*1e6',         r'$P_\mathrm{Heater}$ (µW)'],
                'heater_power_mW':    ['self.y_axis*1e3',         r'$P_\mathrm{Heater}$ (mW)'],
                'T_all_up_mK':        ['self.T_all_up*1e3',       r'$T_{Sample}$ (mK)'],
                'T_all_up_K':         ['self.T_all_up*1e0',       r'$T_{Sample}$ (K)'],
                'T_up_mK':            ['self.T_mean_up*1e3',      r'$T_\mathrm{Sample}$ (mK)'],
                'T_up_K':             ['self.T_mean_up*1e0',      r'$T_\mathrm{Sample}$ (K)'],
                'T_axis_up_K':        ['self.T_axis',             r'$T_\mathrm{Sample}^\rightarrow$ (K)'],
                'T_axis_down_K':      ['self.T_axis',             r'$T_\mathrm{Sample}^\leftarrow$ (K)'],
                'dIdV_up':            ['self.dIdV_up',            r'd$I/$d$V$ ($G_0$)'],
                'dIdV_up_T':          ['self.dIdV_up_T',          r'd$I/$d$V$ ($G_0$)'],
                'dIdV_down':          ['self.dIdV_down',          r'd$I/$d$V$ ($G_0$)'],
                'dIdV_down_T':        ['self.dIdV_down_T',        r'd$I/$d$V$ ($G_0$)'],
                'uH_up_mT':           ['self.y_axis*1e3',         r'$\mu_0H^\rightarrow$ (mT)'],
                'uH_up_T':            ['self.y_axis',             r'$\mu_0H^\rightarrow$ (T)'],
                'uH_down_mT':         ['self.y_axis*1e3',         r'$\mu_0H^\leftarrow$ (mT)'],
                'uH_down_T':          ['self.y_axis',             r'$\mu_0H^\leftarrow$ (T)'],
                'uH_mT':              ['self.y_axis*1e3',         r'$\mu_0H$ (mT)'],
                'uH_T':               ['self.y_axis',             r'$\mu_0H$ (T)'],
                'V_gate_up_V':        ['self.y_axis',             r'$V_\mathrm{Gate}^\rightarrow$ (V)'],
                'V_gate_down_V':      ['self.y_axis',             r'$V_\mathrm{Gate}^\leftarrow$ (V)'],
                'V_gate_V':           ['self.y_axis',             r'$V_\mathrm{Gate}$ (V)'],
                'V_gate_up_mV':       ['self.y_axis*1e3',         r'$V_\mathrm{Gate}^\rightarrow$ (mV)'],
                'V_gate_down_mV':     ['self.y_axis*1e3',         r'$V_\mathrm{Gate}^\leftarrow$ (mV)'],
                'V_gate_mV':          ['self.y_axis*1e3',         r'$V_\mathrm{Gate}$ (mV)'],
                'time_up':            ['self.time_up',            r'time'],
            } 
        
        self.ignore = []

        logger.info('(%s) ... initialized.', self._name)

    def showAmplifications(
            self,
            ):
        """
        Shows amplification over time during whole measurement.
        """
        logger.info('(%s) showAmplifications()', self._name)

        file = h5py.File(f'{self.file_directory}{self.file_folder}{self.file_name}', 'r')
        time  = file['status']['femto']['time']
        amp_A = file['status']['femto']['amp_A']
        amp_B = file['status']['femto']['amp_B']
        fig = plt.figure(1000, figsize=(6,1))
        plt.semilogy(time, amp_A, '-',  label='V1_AMP')
        plt.semilogy(time, amp_B, '--', label='V2_AMP')
        plt.legend()
        plt.title('Femto Amplifications according to Status')
        plt.xlabel('time')
        plt.ylabel('Amplification')

        # TODO: implement datetime, probably two axes

    def setAmplifications(
            self, 
            V1_AMP:float, 
            V2_AMP:float,
            ):
        """
        Sets Amplifications for Calculations.
        """
        logger.info('(%s) setAmplifications(%s, %s)', self._name, V1_AMP, V2_AMP)
        self.V1_AMP = V1_AMP
        self.V2_AMP = V2_AMP
    
    def showMeasurements(self):
        """
        Shows available Measurements in File.
        """
        logger.info('(%s) showMeasurements()', self._name)

        file = h5py.File(f'{self.file_directory}{self.file_folder}{self.file_name}', 'r')
        liste = list(file['measurement'].keys())
        logger.info('(%s) %s', self._name, liste)

    def setMeasurement(
            self, 
            mkey:str
            ):
        """
        Sets Measurement Key. Mandatory for further Evaluation.

        Parameters
        ----------
        mkey : str
            name of measurement
        """
        logger.info("(%s) setMeasurement('%s')", self._name, mkey)
        try: 
            file = h5py.File(f'{self.file_directory}{self.file_folder}{self.file_name}', 'r')
            self.keys = list(file['measurement'][mkey])
            self.mkey = mkey
        except KeyError:
            self.keys = []
            self.mkey = ''
            logger.error("(%s) '%s' found in File.", self._name, mkey)

    def showKeys(
            self
            ):
        """
        Shows available Keys in Measurement.
        """
        logger.info('(%s) showKeys()', self._name)
        show_keys = self.keys[:2] + self.keys[-2:]
        logger.info('(%s) %s', self._name, show_keys)
        
    def setKeys(
            self,
            parameters = None,
            ):
        """
        Sets Keys and calculate y_unsorted. Mandatory for further Evaluation.

        Parameters
        ----------
        parameters : str
            [
            index of first y-value,
            index of last y-value,
            normalization for y-value,
            key to pop,
            ]

        """
        if self.mkey == '':
            logger.warning('(%s) Do setMeasurement() first.', self._name)
            return

        if parameters is None:
            parameters = self.indices[self.mkey]

        logger.info('(%s) setKeys(%s)', self._name, parameters)

        try:
            i0 = parameters[0]
            i1 = parameters[1]
            norm = parameters[2]
            to_pop = parameters[3]
        except IndexError:
            logger.warning('(%s) List of Parameter is incompete.', self._name)
            return
        
        if to_pop in self.keys:
            self.keys.remove(to_pop)
        else:
            logger.warning('(%s) Key to pop is not found.', self._name)

        y = []
        for key in self.keys:
            temp = key[i0:i1]
            temp = float(temp) * norm
            y.append(temp)
        y = np.array(y)

        self.y_unsorted = y

        

    def setV(
            self,
            V_abs:float = np.nan,
            V_min:float = np.nan,
            V_max:float = np.nan,
            V_bins:float = np.nan,
            ):
        """
        Sets V-axis. (Optional)

        Parameters
        ----------
        V_abs : float
            V_min = -V_abs, V_max = +V_abs
        V_min : float
            V_min, is minimum value on V-axis
        V_max : float
            V_min, is minimum value on V-axis
        V_bins : float
            Number of bins minus 1. (float, since default must be np.nan)

        """
        if not np.isnan(V_abs):
            V_min = -V_abs
            V_max = +V_abs
        if not np.isnan(V_min):
            self.V_min = V_min
        if not np.isnan(V_max):
            self.V_max = V_max
        if not np.isnan(V_bins):
            self.V_bins = V_bins
        
        logger.info(
            '(%s) setV(%s, %s, %s)', 
            self._name, 
            self.V_min, 
            self.V_max, 
            self.V_bins
            )
        
    def getMaps(
            self,
            bounds:list = [0, 0],
    ):
        """ getMaps()
        - Calculate I and V and split in up / down sweep
        - Maps I, dIdV, T over linear V-axis
        - Also saves start and stop times
        - As well as offsets
        - sort by y-axis
        """

        logger.info('(%s) getMaps()', self._name)

        # Calculate new V-Axis
        self.V_axis = np.linspace(
            self.V_min, 
            self.V_max, 
            int(self.V_bins)+1,
            )
        
        # Access File
        try:
            file = h5py.File(f'{self.file_directory}{self.file_folder}{self.file_name}', 'r')
            f_keyed = file["measurement"][self.mkey]
        except AttributeError:
            logger.error('(%s) File can not be found!', self._name)
            return
        except KeyError:
            logger.error('(%s) Measurement can not be found!', self._name)
            return

        len_V = np.shape(self.V_axis)[0]
        len_y = np.shape(self.y_unsorted)[0]

        # Initialize all values
        self.I_up         = np.full((len_y, len_V), np.nan, dtype='float64')
        self.I_down       = np.full((len_y, len_V), np.nan, dtype='float64')
        self.time_up      = np.full((len_y, len_V), np.nan, dtype='float64')
        self.time_down    = np.full((len_y, len_V), np.nan, dtype='float64')
        self.T_all_up     = np.full((len_y, len_V), np.nan, dtype='float64')
        self.T_all_down   = np.full((len_y, len_V), np.nan, dtype='float64')
        
        self.t_up_start   = np.full(len_y, np.nan, dtype='float64')
        self.t_up_stop    = np.full(len_y, np.nan, dtype='float64')
        self.t_down_start = np.full(len_y, np.nan, dtype='float64')
        self.t_down_stop  = np.full(len_y, np.nan, dtype='float64')
        self.off_V1       = np.full(len_y, np.nan, dtype='float64')
        self.off_V2       = np.full(len_y, np.nan, dtype='float64')
    
        # Iterate over Keys
        for i, k in enumerate(tqdm(self.keys)):
            # Retrieve Datasets
            offset= f_keyed[k]["offset"]["adwin"]
            sweep = f_keyed[k]["sweep"]["adwin"]
            if "bluefors" in f_keyed[k]["sweep"].keys():
                temperature = f_keyed[k]["sweep"]["bluefors"]
            else:
                temperature = False
                logger.error('(%s) No temperature data available!', self._name)

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

            if self.trigger_up is not None:
                # Get upsweep
                v_raw_up   = v_raw[trigger == self.trigger_up]
                i_raw_up   = i_raw[trigger == self.trigger_up]
                t_up    =  time[trigger == self.trigger_up]
                # Calculate Timepoints
                self.t_up_start[i]   = t_up[0]
                self.t_up_stop[i]    = t_up[-1]
                # Bin that stuff
                i_up, _   = bin_y_over_x(v_raw_up,   i_raw_up,   self.V_axis)
                time_up, _   = bin_y_over_x(v_raw_up,   t_up,    self.V_axis)
                # Save to Array
                self.I_up[i,:]   = i_up
                self.time_up[i,:]   = time_up

            if self.trigger_down is not None:
                # Get dwonsweep
                v_raw_down = v_raw[trigger == self.trigger_down]
                i_raw_down = i_raw[trigger == self.trigger_down]
                t_down  =  time[trigger == self.trigger_down]
                # Calculate Timepoints
                self.t_down_start[i] = t_down[0]
                self.t_down_stop[i]  = t_down[-1]
                # Bin that stuff
                i_down, _ = bin_y_over_x(v_raw_down, i_raw_down, self.V_axis)
                time_down, _ = bin_y_over_x(v_raw_down, t_down,  self.V_axis)
                # Save to Array
                self.I_down[i,:] = i_down
                self.time_down[i,:] = time_down


            # Take care of time and Temperature
            if temperature:
                temp_t = temperature['time']
                temp_T = temperature['Tsample']

                if self.trigger_up is not None:
                    temp_t_up   = linfit(time_up)
                    if temp_t_up[0] > temp_t_up[1]:
                        temp_t_up = np.flip(temp_t_up)
                    T_up, _   = bin_y_over_x(temp_t, temp_T, temp_t_up,   upsampling=1000)
                    self.T_all_up[i,:]   = T_up

                if self.trigger_down is not None:
                    temp_t_down = linfit(time_down)
                    if temp_t_down[0] > temp_t_down[1]:
                        temp_t_down = np.flip(temp_t_down)
                    T_down, _ = bin_y_over_x(temp_t, temp_T, temp_t_down, upsampling=1000)
                    self.T_all_down[i,:] = T_down                
                

        # sorting afterwards, because of probably unknown characters in keys
        indices           = np.argsort(self.y_unsorted)
        self.y_axis       = self.y_unsorted[indices]
        self.off_V1       = self.off_V1[indices]
        self.off_V2       = self.off_V2[indices]

        if self.trigger_up is not None:
            self.I_up         = self.I_up[indices,:]
            self.time_up      = self.time_up[indices,:]
            self.T_all_up     = self.T_all_up[indices,:]
            self.t_up_start   = self.t_up_start[indices]
            self.t_up_stop    = self.t_up_stop[indices]

        if self.trigger_down is not None:
            self.I_down       = self.I_down[indices,:]
            self.time_down    = self.time_down[indices,:]
            self.T_all_down   = self.T_all_down[indices,:]
            self.t_down_start = self.t_down_start[indices]
            self.t_down_stop  = self.t_down_stop[indices]

        if bounds != [0, 0]:
            self.y_axis       = self.y_axis[bounds[0]:bounds[1]]
            self.off_V1       = self.off_V1[bounds[0]:bounds[1]]
            self.off_V2       = self.off_V2[bounds[0]:bounds[1]]

            if self.trigger_up is not None:
                self.I_up         = self.I_up[bounds[0]:bounds[1],:]
                self.time_up      = self.time_up[bounds[0]:bounds[1],:]
                self.T_all_up     = self.T_all_up[bounds[0]:bounds[1],:]
                self.t_up_start   = self.t_up_start[bounds[0]:bounds[1]]
                self.t_up_stop    = self.t_up_stop[bounds[0]:bounds[1]]

            if self.trigger_down is not None:
                self.I_down       = self.I_down[bounds[0]:bounds[1],:]
                self.time_down    = self.time_down[bounds[0]:bounds[1],:]
                self.T_all_down   = self.T_all_down[bounds[0]:bounds[1],:]
                self.t_down_start = self.t_down_start[bounds[0]:bounds[1]]
                self.t_down_stop  = self.t_down_stop[bounds[0]:bounds[1]]

        G_0 = constants.physical_constants['conductance quantum'][0]

        if self.trigger_up is not None:
            # calculating differential conductance
            self.dIdV_up   = np.gradient(self.I_up,   self.V_axis, axis=1)/G_0
            # calculates self.T_mean_up, self.T_mean_down
            self.T_mean_up   = np.nanmean(self.T_all_up,   axis=1)

        if self.trigger_down is not None:
            self.dIdV_down = np.gradient(self.I_down, self.V_axis, axis=1)/G_0
            self.T_mean_down = np.nanmean(self.T_all_down, axis=1)

    def setT(
            self,
            T_min:float = np.nan,
            T_max:float = np.nan,
            T_bins:float = np.nan,
            ):
        """
        Sets T-axis. (Optional)

        Parameters
        ----------
        T_min : float
            T_min, is minimum value on V-axis
        T_max : float
            T_max, is minimum value on V-axis
        T_bins : float
            Number of bins minus 1. (float, since default must be np.nan)

        """
        if not np.isnan(T_min):
            self.T_min = T_min
        if not np.isnan(T_max):
            self.T_max = T_max
        if not np.isnan(T_bins):
            self.T_bins = T_bins
        
        logger.info(
            '(%s) setT(%s, %s, %s)', 
            self._name, 
            self.T_min, 
            self.T_max, 
            self.T_bins,
            )

    def getMapsT(
            self,
        ):
        """ getMapsT()
        - Maps I, dIdV over linear T-axis
        """
        logger.info('(%s) getMapsT()', self._name)

        # Calculate new V-Axis
        self.T_axis = np.linspace(
            self.T_min, 
            self.T_max, 
            int(self.T_bins)+1,
            )
        
        self.I_up_T, self.counter_up     = bin_z_over_y(self.T_mean_up,   self.I_up,   self.T_axis)
        self.I_down_T, self.counter_down = bin_z_over_y(self.T_mean_down, self.I_down, self.T_axis)
        self.dIdV_up_T,   _ = bin_z_over_y(self.T_mean_up,   self.dIdV_up,   self.T_axis)
        self.dIdV_down_T, _ = bin_z_over_y(self.T_mean_down, self.dIdV_down, self.T_axis)

    def showMap(
            self,
            x_key:str = 'V_bias_up_mV',
            y_key:str = 'y_axis',
            z_key:str = 'dIdV_up',
            x_lim:list = [np.nan, np.nan],
            y_lim:list = [np.nan, np.nan],
            z_lim:list = [np.nan, np.nan],
            smoothing:bool = False,
            window_length:float = np.nan,
            polyorder:float = np.nan,
    ):         
        """ showMap()
        - checks for synthax errors
        - get data and label from plot_keys
        - calls plot_map()

        Parameters
        ----------
        x_key : str = 'V_bias_up_mV'
            select plot_key_x from self.plot_keys
        y_key : str = 'y_axis'
            select plot_key_y from self.plot_keys
        z_key : str = 'dIdV_up'
            select plot_key_z from self.plot_keys
        x_lim : list = [np.nan, np.nan]
            sets limits on x-Axis
        y_lim : list = [np.nan, np.nan]
            sets limits on y-Axis
        z_lim : list = [np.nan, np.nan]
            sets limits on z-Axis / colorbar
        """
        
        if x_lim == [np.nan, np.nan]:
            x_lim = self.x_lim
        if y_lim == [np.nan, np.nan]:
            y_lim = self.y_lim
        if z_lim == [np.nan, np.nan]:
            z_lim = self.z_lim
        
        if np.isnan(window_length):
            window_length = self.window_length
        else:
            self.window_length = window_length

        if np.isnan(polyorder):
            polyorder = self.polyorder
        else:
            self.polyorder = polyorder

        logger.info(
            "(%s) showMap(%s, %s, %s)", 
            self._name, 
            [x_key, y_key, z_key], 
            [x_lim, y_lim, z_lim], 
            [smoothing, window_length, polyorder]
            )

        warning = False

        try:
            plot_key_x = self.plot_keys[x_key]
        except KeyError:
            logger.warn('(%s) x_key not found.', self._name)
            warning = True

        try:
            plot_key_y = self.plot_keys[y_key]
        except KeyError:
            logger.warn('(%s) y_key not found.', self._name)
            warning = True

        try:
            plot_key_z = self.plot_keys[z_key]
        except KeyError:
            logger.warn('(%s) z_key not found.', self._name)
            warning = True

        if x_lim[0] >= x_lim[1]:
            logger.warn('(%s) x_lim = [lower_limit, upper_limit].', self._name)
            warning = True

        if y_lim[0] >= y_lim[1]:
            logger.warn('(%s) y_lim = [lower_limit, upper_limit].', self._name)
            warning = True

        if z_lim[0] >= z_lim[1]:
            logger.warn('(%s) z_lim = [lower_limit, upper_limit].', self._name)
            warning = True

        if not warning:
            try:
                x_data = eval(plot_key_x[0])
                y_data = eval(plot_key_y[0])
                z_data = eval(plot_key_z[0])
            except AttributeError:
                logger.warn('(%s) Required data not found. Check if data is calculated and plot_keys!', self._name)
                return
            
            if smoothing:
                z_data = savgol_filter(z_data, window_length=window_length, polyorder=polyorder)

            x_label = plot_key_x[1]
            y_label = plot_key_y[1]
            z_label = plot_key_z[1]

            self.x_label = x_label
            self.y_label = y_label
            self.z_label = z_label

            self.x_lim = x_lim
            self.y_lim = y_lim
            self.z_lim = z_lim

            self.x_key = x_key
            self.y_key = y_key
            self.z_key = z_key

        else:
            logger.warn('(%s) Check Parameter!', self._name)
            try:
                image = Image.open("/home/oliver/Documents/p5control-bluefors-evaluation/utilities/blueforslogo.png", mode='r')
            except FileNotFoundError:
                logger.warn('(%s) Trick verreckt :/', self._name)
                return
            image = np.asarray(image, dtype='float64')
            z_data = np.flip(image[:,:,1], axis=0)
            z_data[z_data >= 80] = .8
            z_data /= np.max(z_data)
            x_data = np.arange(image.shape[1])
            y_data = np.arange(image.shape[0])
            x_label = r'$x_\mathrm{}$ (pxl)'
            y_label = r'$y_\mathrm{}$ (pxl)'
            z_label = r'BlueFors (arb. u.)'
            x_lim = [0., 2000.]
            y_lim = [0., 1000.]
            z_lim = [0., 1.]
        
        fig, ax_z, ax_c, x, y, z, ext = plot_map(
                x = x_data, 
                y = y_data, 
                z = z_data, 
                x_lim = x_lim, 
                y_lim = y_lim, 
                z_lim = z_lim,
                x_label = x_label, 
                y_label = y_label,  
                z_label = z_label, 
                fig_nr = self.fig_nr,
                cmap = self.cmap,
                display_dpi = self.display_dpi,
                contrast = self.contrast,
                )
        
        # self.fig = fig
        # self.ax_z = ax_z
        # self.ax_c = ax_c
        # self.x = x
        # self.y = y
        # self.z = z
        # self.ext = ext

        if warning:
            plt.suptitle('Hier könnte ihre Werbung stehen.')
        elif self.title is not None:
            plt.suptitle(self.title)
        else:
            plt.suptitle(self.mkey)

        self.show_map = {
            'fig': fig,
            'ax_z': ax_z,
            'ax_c': ax_c,
            'ext': ext,
            'x': x,
            'y': y,
            'z': z,
            'x_lim': x_lim,
            'y_lim': y_lim,
            'z_lim': z_lim,
            'x_label': x_label,
            'y_label': y_label,
            'z_label': z_label,
            'x_key': x_key,
            'y_key': y_key,
            'z_key': z_key,
        }

    def saveFigure(
        self,
        ): 
        """ saveFigure()
        - safes Figure to self.fig_folder/self.title
        """
        logger.info("(%s) saveFigure() to %s%s.png", self._name, self.fig_folder, self.title)

        # Handle Title
        title = f"{self.title}"

        # Handle data folder
        folder = os.path.join(os.getcwd(), self.fig_folder, self.sub_folder)
        check = os.path.isdir(folder)
        if not check:
            os.makedirs(folder)

        # Save Everything
        name = os.path.join(folder, title)
        self.fig.savefig(f'{name}.png', dpi=self.png_dpi)
        if self.pdf: # save as pdf
            logger.info("(%s) saveFigure() to %s%s.pdf", self._name, self.fig_folder, self.title)
            self.fig.savefig(f'{name}.pdf', dpi=self.pdf_dpi)

    def saveData(
            self,
            title = None,
            ):
        """ saveData()
        - safes self.__dict__ to pickle
        """
        logger.info("(%s) saveData()", self._name)

        # Handle Title
        if title is None:
            title = f"{self.title}.pickle"

        # Handle data folder
        folder = os.path.join(os.getcwd(), self.data_folder, self.sub_folder)
        check = os.path.isdir(folder)
        if not check and self.data_folder != '':
            os.makedirs(folder)

        # Get Dictionary
        data = {}
        for key in self.__dict__.keys():
            if key not in self.ignore:
                data[key] = self.__dict__[key]

        # save data to pickle
        name = os.path.join(os.getcwd(), self.data_folder, self.sub_folder, title)
        with open(name, 'wb') as file:
            pickle.dump(data, file)

    def loadData(
            self,
            title = None,
            ):
        """ loadData()
        - gets self.__dict__ from pickle
        """
        logger.info("(%s) loadData()", self._name)

        # Handle Title
        if title is None:
            title = f"{self.title}.pickle"
        
        # get data from pickle
        name = os.path.join(os.getcwd(), self.data_folder, self.sub_folder, title)
        with open(name, 'rb') as file:
            data = pickle.load(file)

        # Save Data to self.
        for key in data.keys():
            self.__dict__[key] = data[key]

    def showData(
            self,
            ):
        """ showData()
        - safes self.__dict__ to pickle
        """
        logger.info("(%s) showData()", self._name)

        # Get Dictionary
        data = {}
        for key in self.__dict__.keys():
            if key not in self.ignore:
                data[key] = self.__dict__[key]
        return data
    










'''
Decrapted.


    def getMapsSmooth(
            self,
            window_length = None,
            polyorder = None,
        ):
        """ getMapsSmooth()
        - smoothes available Maps: I, dIdV, up, down, _T
        """
        logger.info('(%s) getMapsSmooth()', self._name)
        
        if window_length is None:
            window_length = self.window_length
        else:
            self.window_length = window_length

        if polyorder is None:
            polyorder = self.polyorder
        else:
            self.polyorder = polyorder

        self.I_up_smooth      = savgol_filter(self.I_up,      window_length=window_length, polyorder=polyorder)
        self.I_down_smooth    = savgol_filter(self.I_down,    window_length=window_length, polyorder=polyorder)
        self.dIdV_up_smooth   = savgol_filter(self.dIdV_up,   window_length=window_length, polyorder=polyorder)
        self.dIdV_down_smooth = savgol_filter(self.dIdV_down, window_length=window_length, polyorder=polyorder)

        if hasattr(self, 'T_axis'):
            self.I_up_T_smooth      = savgol_filter(self.I_up_T,      window_length=window_length, polyorder=polyorder)
            self.I_down_T_smooth    = savgol_filter(self.I_down_T,    window_length=window_length, polyorder=polyorder)
            self.dIdV_up_T_smooth   = savgol_filter(self.dIdV_up_T,   window_length=window_length, polyorder=polyorder)
            self.dIdV_down_T_smooth = savgol_filter(self.dIdV_down_T, window_length=window_length, polyorder=polyorder)

'''