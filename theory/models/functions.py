import numpy as np
from numpy.typing import NDArray

from theory.models.constants import V_tol_mV
from theory.models.constants import tau_tol
from theory.models.constants import T_tol_K
from theory.models.constants import Delta_tol_meV
from theory.models.constants import Gamma_tol_meV


# performance binning
def bin_y_over_x(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x_bins: NDArray[np.float64],
) -> NDArray[np.float64]:
    x_nu = np.append(x_bins, 2 * x_bins[-1] - x_bins[-2])
    x_nu = x_nu - (x_nu[1] - x_nu[0]) / 2
    count, _ = np.histogram(x, bins=x_nu, weights=None)
    count = np.where(count == 0, np.nan, count)
    summe, _ = np.histogram(x, bins=x_nu, weights=y)
    return summe / count


# cache hashes
def cache_hash(
    V_max_mV: float,
    dV_mV: float,
    tau: float,
    T_K: float,
    Delta_1_meV: float,
    Delta_2_meV: float,
    Gamma_1_meV: float,
    Gamma_2_meV: float,
    string="HA",
) -> str:
    string += "_"
    string += f"V_max={V_max_mV:.{V_tol_mV}f}mV_"
    string += f"dV={dV_mV:.{V_tol_mV}f}mV_"
    string += f"tau={tau:.{tau_tol}f}_"
    string += f"T={T_K:.{T_tol_K}f}K_"
    string += f"Delta=({Delta_1_meV:.{Delta_tol_meV}f},"
    string += f"{Delta_2_meV:.{Delta_tol_meV}f})meV_"
    string += f"Gamma=({Gamma_1_meV:.{Gamma_tol_meV}f},"
    string += f"{Gamma_2_meV:.{Gamma_tol_meV}f})meV"
    return string


def cache_hash_pbar(
    tau: float,
    T_K: float,
    Delta_1_meV: float,
    Delta_2_meV: float,
    Gamma_1_meV: float,
    Gamma_2_meV: float,
    string="FCS",
) -> str:
    string += "_"
    string += f"tau={tau:.{tau_tol}f}_"
    string += f"T={T_K:.{T_tol_K}f}K_"
    string += f"Delta=({Delta_1_meV:.{Delta_tol_meV}f},"
    string += f"{Delta_2_meV:.{Delta_tol_meV}f})meV_"
    string += f"Gamma=({Gamma_1_meV:.{Gamma_tol_meV}f},"
    string += f"{Gamma_2_meV:.{Gamma_tol_meV}f})meV"
    return string
