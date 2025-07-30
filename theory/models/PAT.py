import numpy as np
from numpy.typing import NDArray

from scipy.special import jv
from scipy.interpolate import RegularGridInterpolator

import sys

sys.path.append("/Users/oliver/Documents/p5control-bluefors-evaluation")

from theory.models.constants import h_e_pVs


def get_I_nA(
    A_mV: NDArray[np.float64],
    V_mV: NDArray[np.float64],
    I_nA: NDArray[np.float64],
    nu_GHz: float,
    N: int = 100,
) -> NDArray[np.float64]:

    nu_mV = nu_GHz * h_e_pVs
    A = A_mV / nu_mV
    n = np.arange(-N, N + 1, 1)
    n_mV = n * nu_mV

    n, A = np.meshgrid(n, A)
    J_n = jv(n, A)
    J_n_2 = J_n * J_n

    I_nA = np.meshgrid(I_nA, n_mV)[0]
    interp = RegularGridInterpolator(
        (n_mV, V_mV),
        I_nA,
        bounds_error=False,
        fill_value=None,
    )

    V_mV, n_mV = np.meshgrid(V_mV, n_mV)
    V_n_mV = V_mV - n_mV
    I_n_nA = interp(np.stack([n_mV, V_n_mV], axis=-1))

    J_n_2 = J_n_2[:, :, np.newaxis]
    I_nA = I_n_nA[np.newaxis, :, :]
    I_nA = J_n_2 * I_nA

    I_nA = np.sum(I_nA, axis=1)
    return I_nA
