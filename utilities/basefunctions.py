"""
Module: Data Binning and Linear Fitting

Description:
    This module provides utility functions for performing linear fitting,
    binning data along different axes, and calculating extent boundaries
    for visualization. These functions are useful for handling unevenly
    spaced data, interpolating missing values, and preparing datasets for
    further analysis.

Functions:
    - linear_fit(x): Performs a linear fit on an unevenly spaced array that
      may contain NaNs and returns a linearly spaced array.
    - bin_y_over_x(x, y, x_bins, upsampling=0): Bins y-values over an evenly
      spaced and monotonically increasing x-axis, with optional upsampling
      for handling gaps in data.
    - bin_z_over_y(y, z, y_binned): Bins z-values over y-binned data, filling
      missing values using neighboring bins.
    - get_ext(x, y, x_lim, y_lim): Computes extent boundaries for data
      visualization, considering pixel dimensions and handling None values
      in axis limits.

Dependencies:
    - logging
    - torch
    - numpy

Author: Oliver Irtenkauf
Date: 2025-04-01
"""

import logging
import torch

import numpy as np

logger = logging.getLogger(__name__)


def linear_fit(x) -> np.ndarray:
    """
    Perform a linear fit on an unevenly spaced array that may contain NaN values.
    The function returns an evenly spaced, NaN-free array with interpolated values.

    Parameters:
        x (np.ndarray): Input array with uneven spacing and potential NaN values.

    Returns:
        np.ndarray: Evenly spaced array with linearly interpolated values.
    """
    nu_x = np.copy(x)
    nans = np.isnan(x)
    not_nans = np.invert(nans)
    xx = np.arange(np.shape(nu_x)[0])
    poly = np.polyfit(xx[not_nans], nu_x[not_nans], 1)
    fit_x = xx * poly[0] + poly[1]
    return fit_x


def bin_y_over_x(x: np.ndarray, y: np.ndarray, x_bins: np.ndarray, upsampling: int = 0):
    """
    Bin y-values over evenly spaced, monotonically increasing x-values.
    If there are large gaps in the y-data, upsampling can be used to fill them.

    Parameters:
        x (np.ndarray): X-values (may be unevenly spaced).
        y (np.ndarray): Y-values corresponding to x-values.
        x_bins (np.ndarray): Bin edges for x-values.
        upsampling (int, optional): Factor to increase data resolution. Defaults to 0.

    Returns:
        tuple: (Binned y-values, count of points in each bin)
    """
    if upsampling:
        k = np.full((2, len(x)), np.nan)
        k[0, :] = x
        k[1, :] = y
        m = torch.nn.Upsample(mode="linear", scale_factor=upsampling)
        big = m(torch.from_numpy(np.array([k])))
        x = np.array(big[0, 0, :])
        y = np.array(big[0, 1, :])

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


# def bin_z_over_y(y: np.ndarray, z: np.ndarray, y_binned: np.ndarray):
#     """
#     Bin z-values over y-values using predefined bins.

#     Parameters:
#         y (np.ndarray): Y-values to be binned.
#         z (np.ndarray): Corresponding Z-values (2D array with shape (N, M)).
#         y_binned (np.ndarray): Bin edges for y-values.

#     Returns:
#         tuple: (Binned z-values, count of points in each bin)
#     """
#     number_of_bins = len(y_binned)
#     counter = np.zeros(number_of_bins, dtype=int)
#     result = np.zeros((number_of_bins, z.shape[1]), dtype=float)

#     dig = np.digitize(y, bins=y_binned) - 1
#     valid_indices = (dig >= 0) & (dig < number_of_bins)

#     np.add.at(counter, dig[valid_indices], 1)
#     np.add.at(result, dig[valid_indices], z[valid_indices])

#     with np.errstate(invalid='ignore'):  # Suppress divide-by-zero warnings
#         result[counter > 0] /= counter[counter > 0, None]
#         result[counter == 0] = np.nan

#     return result, counter


def bin_z_over_y(y: np.ndarray, z: np.ndarray, y_binned: np.ndarray):
    """
    Bin z-values over y-values using predefined bins.

    Parameters:
        y (np.ndarray): Y-values to be binned.
        z (np.ndarray): Corresponding Z-values (2D array with shape (N, M)).
        y_binned (np.ndarray): Bin edges for y-values.

    Returns:
        tuple: (Binned z-values, count of points in each bin)
    """
    number_of_bins = np.shape(y_binned)[0]
    counter = np.zeros(number_of_bins, dtype=int)
    result = np.zeros((number_of_bins, z.shape[1]), dtype=float)

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


# def get_ext(x: np.ndarray, y: np.ndarray, x_lim: tuple, y_lim: tuple):
#     """
#     Calculate Extent, X-Limits and Y-Limits from given x, y, x_lim, y_lim
#     Takes into account of pixel dimension and Nones in xy_lim.
#     """

#     pixel_width = np.abs(x[-1] - x[-2])
#     pixel_height = np.abs(y[-1] - y[-2])
#     ext = (
#         x[0] - pixel_width / 2,
#         x[len(x) - 1] + pixel_width / 2,
#         y[0] - pixel_height / 2,
#         y[len(y) - 1] + pixel_height / 2,
#     )

#     if x_lim[0] is not None:
#         new_x_lim_0 = x_lim[0] - pixel_width / 2
#     else:
#         new_x_lim_0 = None
#     if x_lim[1] is not None:
#         new_x_lim_1 = x_lim[1] + pixel_width / 2
#     else:
#         new_x_lim_1 = None
#     if y_lim[0] is not None:
#         new_y_lim_0 = y_lim[0] - pixel_height / 2
#     else:
#         new_y_lim_0 = None
#     if y_lim[1] is not None:
#         new_y_lim_1 = y_lim[1] + pixel_height / 2
#     else:
#         new_y_lim_1 = None

#     new_x_lim = (new_x_lim_0, new_x_lim_1)
#     new_y_lim = (new_y_lim_0, new_y_lim_1)

#     return ext, new_x_lim, new_y_lim


def get_ext(x: np.ndarray, y: np.ndarray, x_lim: tuple, y_lim: tuple):
    """
    Compute the extent and adjusted limits for an image-like representation of data.
    Takes into account pixel dimensions and handles None values in limits.

    Parameters:
        x (np.ndarray): X-axis values.
        y (np.ndarray): Y-axis values.
        x_lim (tuple): X-axis limits (can contain None values).
        y_lim (tuple): Y-axis limits (can contain None values).

    Returns:
        tuple: (Extent, new x limits, new y limits)
    """
    pixel_width, pixel_height = np.abs(x[-1] - x[-2]), np.abs(y[-1] - y[-2])
    ext = (
        x[0] - pixel_width / 2,
        x[-1] + pixel_width / 2,
        y[0] - pixel_height / 2,
        y[-1] + pixel_height / 2,
    )

    adjust_limit = lambda lim, pixel_size: (
        (lim - pixel_size / 2) if lim is not None else None
    )
    new_x_lim = (
        adjust_limit(x_lim[0], pixel_width),
        adjust_limit(x_lim[1], pixel_width),
    )
    new_y_lim = (
        adjust_limit(y_lim[0], pixel_height),
        adjust_limit(y_lim[1], pixel_height),
    )

    return ext, new_x_lim, new_y_lim
