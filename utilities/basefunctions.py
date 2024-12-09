"""
Contains the functions:
- linear_fit
- bin_y_over_x
- bin_z_over_y
- get_ext
"""

import logging
import torch

import numpy as np

logger = logging.getLogger(__name__)


def linear_fit(x):
    """
    time binned over voltage is not equally spaced and might have nans
    this function does a linear fit to uneven spaced array, that might contain nans
    gives back linear and even spaced array

    """
    nu_x = np.copy(x)
    nans = np.isnan(x)
    not_nans = np.invert(nans)
    xx = np.arange(np.shape(nu_x)[0])
    poly = np.polyfit(xx[not_nans], nu_x[not_nans], 1)
    fit_x = xx * poly[0] + poly[1]
    return fit_x


def bin_y_over_x(
    x,
    y,
    x_bins,
    upsampling=None,
):
    """
    Description
    # gives y-values over even-spaced and monoton-increasing x
    # incase of big gaps in y-data, use upsampling, to fill those.
    """
    if upsampling is not None:
        k = np.full((2, len(x)), np.nan)
        k[0, :] = x
        k[1, :] = y
        m = torch.nn.Upsample(mode="linear", scale_factor=upsampling)
        big = m(torch.from_numpy(np.array([k])))
        x = np.array(big[0, 0, :])
        y = np.array(big[0, 1, :])
    else:
        pass

    # Apply binning based on histogram function
    x_nu = np.append(x_bins, 2 * x_bins[-1] - x_bins[-2])
    x_nu = x_nu - (x_nu[1] - x_nu[0]) / 2
    # Instead of N_x, gives fixed axis.
    # Solves issues with wider ranges, than covered by data
    _count, _ = np.histogram(x, bins=x_nu, weights=None)
    _count = np.array(_count, dtype="float64")
    _count[_count == 0] = np.nan

    _sum, _ = np.histogram(x, bins=x_nu, weights=y)
    return _sum / _count, _count


def bin_z_over_y(
    y,
    z,
    y_binned,
):
    """
    Desription.
    """
    number_of_bins = np.shape(y_binned)[0]
    counter = np.full(number_of_bins, 0)
    result = np.full((number_of_bins, np.shape(z)[1]), 0, dtype="float64")

    # Find Indizes of x on x_binned
    dig = np.digitize(y, bins=y_binned)

    # Add up counter, I & differential_conductance
    for i, d in enumerate(dig):
        counter[d - 1] += 1
        result[d - 1, :] += z[i, :]

    # Normalize with counter, rest to np.nan
    for i, c in enumerate(counter):
        if c > 0:
            result[i, :] /= c
        elif c == 0:
            result[i, :] *= np.nan

    # Fill up Empty lines with Neighboring Lines
    for i, c in enumerate(counter):
        if c == 0:  # In case counter is 0, we need to fill up
            up, down = i, i  # initialze up, down
            while counter[up] == 0 and up < number_of_bins - 1:
                # while up is still smaller -2 and counter is still zero, look for better up
                up += 1
            while counter[down] == 0 and down >= 1:
                # while down is still bigger or equal 1 and coutner still zero, look for better down
                down -= 1

            if up == number_of_bins - 1 or down == 0:
                # Just ignores the edges, when c == 0
                result[i, :] *= np.nan
            else:
                # Get Coordinate System
                span = up - down
                relative_pos = i - down
                lower_span = span * 0.25
                upper_span = span * 0.75

                # Divide in same as next one and intermediate
                if 0 <= relative_pos <= lower_span:
                    result[i, :] = result[down, :]
                elif lower_span < relative_pos < upper_span:
                    result[i, :] = np.divide(result[up, :] + result[down, :], 2)
                elif upper_span <= relative_pos <= span:
                    result[i, :] = result[up, :]
                else:
                    logger.warning("something went wrong!")
    return result, counter


def get_ext(x: np.ndarray, y: np.ndarray, x_lim: tuple, y_lim: tuple):
    """
    Calculate Extent, X-Limits and Y-Limits from given x, y, x_lim, y_lim
    Takes into account of pixel dimension and Nones in xy_lim.
    """

    pixel_width = np.abs(x[-1] - x[-2])
    pixel_height = np.abs(y[-1] - y[-2])
    ext = (
        x[0] - pixel_width / 2,
        x[len(x) - 1] + pixel_width / 2,
        y[0] - pixel_height / 2,
        y[len(y) - 1] + pixel_height / 2,
    )

    if x_lim[0] is not None:
        new_x_lim_0 = x_lim[0] - pixel_width / 2
    else:
        new_x_lim_0 = None
    if x_lim[1] is not None:
        new_x_lim_1 = x_lim[1] + pixel_width / 2
    else:
        new_x_lim_1 = None
    if y_lim[0] is not None:
        new_y_lim_0 = y_lim[0] - pixel_height / 2
    else:
        new_y_lim_0 = None
    if y_lim[1] is not None:
        new_y_lim_1 = y_lim[1] + pixel_height / 2
    else:
        new_y_lim_1 = None

    new_x_lim = (new_x_lim_0, new_x_lim_1)
    new_y_lim = (new_y_lim_0, new_y_lim_1)

    return ext, new_x_lim, new_y_lim
