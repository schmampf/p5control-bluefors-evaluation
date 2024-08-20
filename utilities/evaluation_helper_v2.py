import numpy as np
from torch.nn import Upsample
from torch import from_numpy

import matplotlib.pyplot as plt

import warnings

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
    
def plot_map(
    x: np.ndarray, 
    y: np.ndarray, 
    z: np.ndarray, 
    x_lim: list[float] = [-1., 1.], 
    y_lim: list[float] = [-1., 1.], 
    z_lim: list[float] = [ 0., 0.],
    x_label: str = r'$x$-label', 
    y_label: str = r'$y$-label',  
    z_label: str = r'$z$-label', 
    fig_nr: int = 0,
    cmap = cmap(color='seeblau', bad='gray'),
    display_dpi: int = 100,
    contrast: float = 1.,
    ):
    
    if z.dtype == np.dtype('int32'):
        warnings.warn("z is integer. Sure?")

    stepsize_x=np.abs(x[-1]-x[-2])/2
    stepsize_y=np.abs(y[-1]-y[-2])/2
    x_ind = [np.abs(x-x_lim[0]).argmin(),
                np.abs(x-x_lim[1]).argmin()]
    y_ind = [np.abs(y-y_lim[0]).argmin(),
                np.abs(y-y_lim[1]).argmin()]
    
    ext = np.array([x[x_ind[0]]-stepsize_x,
                    x[x_ind[1]]+stepsize_x,
                    y[y_ind[0]]-stepsize_y,
                    y[y_ind[1]]+stepsize_y],
                    dtype = 'float64')
    z = np.array(z[y_ind[0]:y_ind[1], x_ind[0]:x_ind[1]], dtype='float64')
    x = np.array(x[x_ind[0]:x_ind[1]], dtype='float64')
    y = np.array(y[y_ind[0]:y_ind[1]], dtype='float64')

    if z_lim == [0, 0]:
        z_lim = [float(np.nanmean(z)-np.nanstd(z)/contrast), 
                 float(np.nanmean(z)+np.nanstd(z)/contrast)]
        
    if x_lim[0] >= x_lim[1] or y_lim[0] >= y_lim[1] or z_lim[0] >= z_lim[1]:
        warnings.warn('First of xy_lim must be smaller than first one.')

    plt.close(fig_nr)
    fig, (ax_z, ax_c) = plt.subplots(
        num=fig_nr,
        ncols=2,
        figsize=(6,4),
        dpi=display_dpi,
        gridspec_kw={"width_ratios":[5.8,.2]},
        constrained_layout=True
        )

    im = ax_z.imshow(z, 
                    extent=ext, 
                    aspect='auto',
                    origin='lower',
                    clim=z_lim,
                    cmap=cmap,
                    interpolation='none')
    ax_z.set_xlabel(x_label)
    ax_z.set_ylabel(y_label)
    ax_z.ticklabel_format(
        axis="both", 
        style="sci", 
        scilimits=(-9,9),
        useMathText=True
    )
    ax_z.tick_params(direction='in')

    cbar = fig.colorbar(im, label=z_label, cax=ax_c)
    ax_c.tick_params(direction='in')
    lim = ax_z.set_xlim(ext[0],ext[1])
    lim = ax_z.set_ylim(ext[2],ext[3])
    
    return fig, ax_z, ax_c, x, y, z, ext


