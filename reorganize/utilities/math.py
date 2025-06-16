import math
import numpy as np


def dBmToV(dBm):
    """Convert dBm to voltage in volts."""
    # P = 10 ** ((dBm - 30) / 10)
    R = 50
    V = math.sqrt(R / 1000) * 10 ** (dBm / 20)
    return V


def moving_average(arr, window_size):
    return np.convolve(arr, np.ones(window_size) / window_size, mode="valid")
