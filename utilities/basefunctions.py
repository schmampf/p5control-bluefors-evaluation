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
    - get_norm(values): return appropriate norm and corresponding unit string.

Dependencies:
    - logging
    - torch
    - numpy

Author: Oliver Irtenkauf
Date: 2025-04-01
"""

import torch
import warnings

import numpy as np
from numpy.fft import fft, fftfreq
from sympy import isprime


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


def bin_y_over_x(
    x: np.ndarray,
    y: np.ndarray,
    x_bins: np.ndarray,
    upsample: int = 0,
    upsample_method: str = "linear",
):
    """
    Bins y-values over given x-intervals (x_bins), optionally upsampling the (x, y) data beforehand.

    This function is useful for aggregating y-values over x-axis intervals, such as averaging measurements
    over fixed voltage or time bins. If the data is sparse or contains gaps, upsampling with interpolation
    can increase the resolution before binning.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-values (e.g., voltage or time). Does not need to be evenly spaced.
    y : np.ndarray
        1D array of y-values corresponding to x.
    x_bins : np.ndarray
        1D array of bin edges for x.
    upsample : int, optional
        Upsampling factor for increasing resolution before binning. If 0 (default), no upsampling is applied.
    upsample_method : str, optional
        Interpolation method for upsampling. Must be compatible with `torch.nn.Upsample`.
        Common options: "linear", "nearest", "bicubic".

    Returns
    -------
    tuple of np.ndarray
        - Binned y-values (mean per bin)
        - Count of data points (after upsampling) in each bin
    """
    if upsample:
        # Stack x and y as two rows in a 2D array for upsampling
        k = np.full((2, len(x)), np.nan)
        k[0, :] = x
        k[1, :] = y

        # Apply PyTorch interpolation
        m = torch.nn.Upsample(mode=upsample_method, scale_factor=upsample)
        big = m(torch.from_numpy(np.array([k])))
        x = np.array(big[0, 0, :])
        y = np.array(big[0, 1, :])

    # Extend bin edges for histogram: shift by half a bin width for center alignment
    x_nu = np.append(x_bins, 2 * x_bins[-1] - x_bins[-2])  # Add one final edge
    x_nu = x_nu - (x_nu[1] - x_nu[0]) / 2

    # Count how many x-values fall into each bin
    _count, _ = np.histogram(x, bins=x_nu, weights=None)
    _count = np.array(_count, dtype="float64")
    _count[_count == 0] = np.nan

    # Sum of y-values in each bin
    _sum, _ = np.histogram(x, bins=x_nu, weights=y)

    # Return mean y per bin and count
    return _sum / _count, _count


def bin_z_over_y(y: np.ndarray, z: np.ndarray, y_binned: np.ndarray):
    """
    Bin z-values over y-values using predefined bins.
    If a bin receives no data, fill it from the previous valid bin above (top-down fill).

    Parameters
    ----------
    y : np.ndarray
        Y-values to be binned.
    z : np.ndarray
        Corresponding Z-values (2D array with shape (N, M)).
    y_binned : np.ndarray
        Bin edges for y-values.

    Returns
    -------
    tuple
        Binned z-values (2D), and count of points in each bin.
    """
    number_of_bins = len(y_binned)
    counter = np.zeros(number_of_bins, dtype=int)
    result = np.zeros((number_of_bins, z.shape[1]), dtype=float)

    # Assign y-values to bins
    dig = np.digitize(y, bins=y_binned) - 1
    valid_indices = (dig >= 0) & (dig < number_of_bins)

    # Count entries and sum z-values into bins
    np.add.at(counter, dig[valid_indices], 1)
    np.add.at(result, dig[valid_indices], z[valid_indices])

    # Average values where data exists
    with np.errstate(invalid="ignore"):  # Suppress divide-by-zero warnings
        result[counter > 0] /= counter[counter > 0, None]
        result[counter == 0] = np.nan  # Mark empty bins as NaN

    # Top-down fill: fill missing rows from previous valid row above
    last_valid = None
    for i in range(number_of_bins):
        if counter[i] > 0:
            last_valid = result[i].copy()
        elif last_valid is not None:
            result[i] = last_valid

    return result, counter


def get_ext(
    x: np.ndarray,
    y: np.ndarray,
    x_lim: tuple = (None, None),
    y_lim: tuple = (None, None),
):
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


def get_z_lim(z_values: np.ndarray, z_lim: tuple = (None, None), z_contrast: float = 1):
    """
    Determines the z-axis limits based on the given `z_lim` values and contrast factor.

    Parameters
    ----------
    z_values : np.ndarray
        The dataset from which to calculate the z-limits.
    z_lim : tuple, optional
        Tuple specifying the lower and upper limits (default: (None, None)).
        If `None`, limits are computed based on mean and standard deviation.
    z_contrast : float, optional
        Scaling factor for standard deviation to adjust z-limits (default: 1).

    Returns
    -------
    tuple
        A tuple (z_lim_0, z_lim_1) representing the computed or given z-axis limits.
    """
    delta_z = np.nanstd(z_values) * z_contrast
    mean_z = np.nanmean(z_values)

    z_lim_0 = z_lim[0] if z_lim[0] is not None else mean_z - delta_z
    z_lim_1 = z_lim[1] if z_lim[1] is not None else mean_z + delta_z

    return z_lim_0, z_lim_1


def get_norm(values):
    """
    Determines an appropriate normalization factor and corresponding SI prefix
    for a given set of values.

    The function finds the largest absolute value in `values` and selects the
    smallest normalization factor from a predefined list that is still
    smaller than or equal to this maximum value.

    Parameters
    ----------
    values : array-like
        Input numerical values that need to be normalized.

    Returns
    -------
    float
        The selected normalization factor.
    str
        The corresponding SI unit prefix.

    Examples
    --------
    >>> get_norm([0.002, 0.005, 0.009])
    (0.001, 'm')

    >>> get_norm([5e6, 1e7, 3e8])
    (1000000.0, 'M')
    """
    norm_values = [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
    norm_string = ["G", "M", "k", "", "m", "µ", "n", "p", "f"]

    max_value = np.nanmax(np.abs(values))
    where_norm_is_smaller = norm_values <= max_value
    index = np.where(where_norm_is_smaller)[0][0]

    return norm_values[index], norm_string[index]


def get_power(amplitude, R=50):
    """
    Convert a real voltage amplitude (peak) to power in dBm.

    Parameters
    ----------
    amplitude : float or np.ndarray
        Voltage amplitude (in volts).
    R : float, optional
        System impedance in ohms. Defaults to 50 ohms.

    Returns
    -------
    power_dbm : float or np.ndarray
        Corresponding power in dBm.

    Notes
    -----
    Assumes power = V^2 / R, and converts to dBm using:
        dBm = 10 * log10(power in watts / 1e-3)
    """
    power_watts = amplitude**2 / R  # Calculate power in watts
    power_dbm = 10 * np.log10(power_watts / 1e-3)  # Convert to dBm (1 mW reference)
    return power_dbm


def get_amplitude(power_dbm, R=50):
    """
    Convert power in dBm to real voltage amplitude (peak).

    Parameters
    ----------
    power_dbm : float or np.ndarray
        Power in dBm.
    R : float, optional
        System impedance in ohms. Defaults to 50 ohms.

    Returns
    -------
    amplitude : float or np.ndarray
        Corresponding voltage amplitude (in volts).

    Notes
    -----
    Inverts the formula:
        power = V^2 / R,
    after converting dBm to watts.
    """
    power_watts = 10 ** (power_dbm / 10) * 1e-3  # Convert dBm to watts
    amplitude = np.sqrt(power_watts * R)  # Solve for voltage
    return amplitude


def make_even_spaced(input_array: np.ndarray) -> np.ndarray:
    """
    Converts an unevenly spaced 1D numpy array into an evenly spaced array
    with spacing equal to the minimal spacing in the input array, ignoring NaN values.

    Parameters
    ----------
    input_array : np.ndarray
        Unevenly spaced 1D numpy array.

    Returns
    -------
    np.ndarray
        Evenly spaced 1D numpy array with spacing equal to the minimal spacing
        in the input array.
    """
    if len(input_array) < 2:
        raise ValueError("Input array must have at least two elements.")

    # Remove NaN values from the input array
    input_array = input_array[~np.isnan(input_array)]

    if len(input_array) < 2:
        raise ValueError("Input array must have at least two non-NaN elements.")

    # Sort the input array to ensure proper spacing calculation
    sorted_array = np.sort(input_array)

    # Calculate the minimal spacing
    min_spacing = np.min(np.diff(sorted_array))

    # Generate the evenly spaced array
    even_spaced_array = np.arange(
        sorted_array[0], sorted_array[-1] + min_spacing, min_spacing
    )

    return even_spaced_array


def downsample_signals_by_time(
    current: np.ndarray,
    voltage: np.ndarray,
    time: np.ndarray,
    target_frequency: float = 0,
    savety_factor: float = 3,
    prefer_prime: bool = True,
    power_threshold: float = 0.95,
):
    """
    Downsample time-domain current and voltage signals to a uniform time grid.

    If `target_frequency` is 0, a suitable downsampling frequency is automatically
    estimated based on the spectral content of both signals, considering the
    specified `power_threshold`, `savety_factor`, and whether prime numbers are preferred.

    Parameters
    ----------
    current : array-like
        Current signal sampled over time.
    voltage : array-like
        Voltage signal sampled over time.
    time : array-like
        Time vector corresponding to the current and voltage signals.
    target_frequency : float, optional
        Target frequency [Hz] for downsampling. If 0, a safe frequency is estimated
        using `estimate_downsample_freq_dual()` (default is 0).
    savety_factor : float, optional
        Minimum required ratio between input sampling frequency and target downsampling
        frequency to avoid aliasing. Used to enforce Nyquist-compliant downsampling (default is 3).
    prefer_prime : bool, optional
        Whether to prefer prime numbers as target frequencies when automatically estimated
        (default is True).
    power_threshold : float, optional
        Fraction of the total signal power to preserve in the spectral analysis when
        estimating a suitable downsampling frequency (default is 0.95).

    Returns
    -------
    i_down : ndarray
        Downsampled current values (mean per time bin).
    v_down : ndarray
        Downsampled voltage values (mean per time bin).
    t_down : ndarray
        New uniform time grid corresponding to downsampled signals.
    counts : ndarray
        Number of original samples per bin.
    target_frequency : float
        Target downsampling frequency [Hz].

    Raises
    ------
    ValueError
        If the input sampling rate is lower than the target downsampling rate.
    """
    current = np.asarray(current)
    voltage = np.asarray(voltage)
    time = np.asarray(time)

    # Estimate input frequency
    dt = np.median(np.diff(time))
    input_frequency = 1 / dt

    # Auto-estimate if target frequency is zero
    if target_frequency == 0:
        target_frequency = estimate_downsample_freq_dual(
            current,
            voltage,
            time,
            safety_factor=savety_factor,
            prefer_prime=prefer_prime,
            power_threshold=power_threshold,
        )

    if input_frequency < target_frequency:
        raise ValueError(
            f"Input frequency ({input_frequency:.2f} Hz) is lower than target "
            f"downsampling frequency ({target_frequency:.2f} Hz). Upsampling is not allowed."
        )
    elif input_frequency < savety_factor * target_frequency:
        warnings.warn(
            f"Input frequency ({input_frequency:.2f} Hz) is less than 3× the target "
            f"frequency ({target_frequency:.2f} Hz). This may violate the Nyquist "
            f"sampling principle and result in aliasing or signal degradation.",
            RuntimeWarning,
        )

    # Create uniform time grid for downsampling
    t_down = np.arange(np.min(time), np.max(time), 1 / target_frequency)

    # Bin current and voltage over time
    i_down, counts = bin_y_over_x(time, current, t_down)
    v_down, _ = bin_y_over_x(time, voltage, t_down)

    return i_down, v_down, t_down, counts, target_frequency


def estimate_downsample_freq_dual(
    signal1, signal2, time, power_threshold=0.95, safety_factor=3, prefer_prime=True
):
    """
    Estimate a safe downsampling frequency for two signals (e.g., current and voltage).

    This function analyzes the frequency content of both signals and returns a recommended
    downsampling frequency that preserves a given fraction of total power while respecting
    the Nyquist criterion (plus a safety factor). Optionally, the result can be adjusted
    to the nearest lower prime number.

    Parameters
    ----------
    signal1 : array-like
        First input signal (e.g., current).
    signal2 : array-like
        Second input signal (e.g., voltage).
    time : array-like
        Time vector associated with both signals (must be uniformly sampled).
    power_threshold : float, optional
        Fraction of total signal power to retain (default is 0.95).
    safety_factor : float, optional
        Multiplier applied to the dominant frequency to ensure safe downsampling (default is 3).
    prefer_prime : bool, optional
        If True, snap the result to the nearest lower prime number (default is True).

    Returns
    -------
    downsample_freq : float
        Recommended downsampling frequency in Hz.
    """
    f1 = estimate_dominant_freq(signal1, time, power_threshold)
    f2 = estimate_dominant_freq(signal2, time, power_threshold)
    max_freq = max(f1, f2)

    suggested_freq = safety_factor * max_freq

    if prefer_prime:
        return nearest_lower_prime(suggested_freq)
    else:
        return suggested_freq


def estimate_dominant_freq(signal, time, power_threshold=0.95):
    """
    Estimate the frequency below which a specified fraction of the signal's power is contained.

    Parameters
    ----------
    signal : array-like
        Time-domain signal (e.g., current or voltage).
    time : array-like
        Corresponding time vector.
    power_threshold : float, optional
        Fraction of total power to preserve (default is 0.95).

    Returns
    -------
    dominant_freq : float
        Frequency in Hz below which `power_threshold` of total power is contained.
    """
    signal = np.asarray(signal).flatten()
    time = np.asarray(time)
    dt = float(np.median(np.diff(time)))
    n = int(len(signal))

    freqs = fftfreq(n, d=dt)[: n // 2]
    power_spectrum = np.abs(fft(signal)[: n // 2]) ** 2

    total_power = np.sum(power_spectrum)
    cumsum = np.cumsum(power_spectrum) / total_power
    idx = np.searchsorted(cumsum, power_threshold)
    return freqs[idx]


def nearest_lower_prime(n):
    """
    Return the largest prime number less than or equal to the input.

    Parameters
    ----------
    n : float or int
        Target value to snap to a lower prime number.

    Returns
    -------
    prime : int
        Nearest smaller or equal prime number.
    """
    n = int(np.floor(n))
    while n > 2:
        if isprime(n):
            return n
        n -= 1
    return 2


# import logging
# logger = logging.getLogger(__name__)
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
#     number_of_bins = np.shape(y_binned)[0]
#     counter = np.zeros(number_of_bins, dtype=int)
#     result = np.zeros((number_of_bins, z.shape[1]), dtype=float)

#     # Find Indices of x on x_binned
#     dig = np.digitize(y, bins=y_binned)

#     # Add up counter, I & differential_conductance
#     for i, d in enumerate(dig):
#         counter[d - 1] += 1
#         result[d - 1, :] += z[i, :]

#     # Normalize with counter, rest to np.nan
#     for i, c in enumerate(counter):
#         if c > 0:
#             result[i, :] /= c
#         elif c == 0:
#             result[i, :] *= np.nan

#     # Fill up Empty lines with Neighboring Lines
#     for i, c in enumerate(counter):
#         if c == 0:  # In case counter is 0, we need to fill up
#             up, down = i, i  # initialize up, down
#             while counter[up] == 0 and up < number_of_bins - 1:
#                 # while up is still smaller -2 and counter is still zero, look for better up
#                 up += 1
#             while counter[down] == 0 and down >= 1:
#                 # while down is still bigger or equal 1 and counter still zero, look for better down
#                 down -= 1

#             if up == number_of_bins - 1 or down == 0:
#                 # Just ignores the edges, when c == 0
#                 result[i, :] *= np.nan
#             else:
#                 # Get Coordinate System
#                 span = up - down
#                 relative_pos = i - down
#                 lower_span = span * 0.25
#                 upper_span = span * 0.75

#                 # Divide in same as next one and intermediate
#                 if 0 <= relative_pos <= lower_span:
#                     result[i, :] = result[down, :]
#                 elif lower_span < relative_pos < upper_span:
#                     result[i, :] = np.divide(result[up, :] + result[down, :], 2)
#                 elif upper_span <= relative_pos <= span:
#                     result[i, :] = result[up, :]
#                 else:
#                     logger.warning("something went wrong!")
#     return result, counter


# def get_ext(x: np.ndarray, y: np.ndarray, x_lim: tuple, y_lim: tuple):
#     """
#     Compute the extent and adjusted limits for an image-like representation of data.
#     Takes into account pixel dimensions and handles None values in limits.

#     Parameters:
#         x (np.ndarray): X-axis values.
#         y (np.ndarray): Y-axis values.
#         x_lim (tuple): X-axis limits (can contain None values).
#         y_lim (tuple): Y-axis limits (can contain None values).

#     Returns:
#         tuple: (Extent, new x limits, new y limits)
#     """
#     pixel_width, pixel_height = np.abs(x[-1] - x[-2]), np.abs(y[-1] - y[-2])
#     ext = (
#         x[0] - pixel_width / 2,
#         x[-1] + pixel_width / 2,
#         y[0] - pixel_height / 2,
#         y[-1] + pixel_height / 2,
#     )

#     adjust_limit = lambda lim, pixel_size: (
#         (lim - pixel_size / 2) if lim is not None else None
#     )
#     new_x_lim = (
#         adjust_limit(x_lim[0], pixel_width),
#         adjust_limit(x_lim[1], pixel_width),
#     )
#     new_y_lim = (
#         adjust_limit(y_lim[0], pixel_height),
#         adjust_limit(y_lim[1], pixel_height),
#     )

#     return ext, new_x_lim, new_y_lim
