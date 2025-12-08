import torch
import numpy as np
from typing import TypeAlias
from numpy.typing import NDArray

from theory.models.constants import V_tol_mV
from theory.models.constants import tau_tol
from theory.models.constants import T_tol_K
from theory.models.constants import Delta_tol_meV
from theory.models.constants import gamma_tol_meV

NDArray64: TypeAlias = NDArray[np.float64]


# performance binning
def bin_y_over_x(
    x: NDArray64,
    y: NDArray64,
    x_bins: NDArray64,
) -> NDArray64:
    x_nu = np.append(x_bins, 2 * x_bins[-1] - x_bins[-2])
    x_nu = x_nu - (x_nu[1] - x_nu[0]) / 2
    count, _ = np.histogram(x, bins=x_nu, weights=None)
    count = np.where(count == 0, np.nan, count)
    summe, _ = np.histogram(x, bins=x_nu, weights=y)
    return summe / count


def oversample(
    x: NDArray64,
    y: NDArray64,
    upsample: int = 100,
    upsample_method: str = "linear",
) -> tuple[NDArray64, NDArray64]:
    if upsample <= 1:
        return x, y  # nothing to do

    # stack as channels: shape (batch=1, channels=2, length=N)
    k = np.stack([x, y])[None, ...]  # shape (1, 2, N)
    k_torch = torch.tensor(k, dtype=torch.float32)

    # interpolate along last dimension
    m = torch.nn.Upsample(
        scale_factor=upsample, mode=upsample_method, align_corners=True
    )
    big = m(k_torch)  # shape (1, 2, N*upsample)

    x_new = big[0, 0, :].numpy().astype(np.float64)
    y_new = big[0, 1, :].numpy().astype(np.float64)

    return x_new, y_new


# cache hashes
def cache_hash(
    V_max_mV: float,
    dV_mV: float,
    tau: float,
    T_K: float,
    Delta_1_meV: float,
    Delta_2_meV: float,
    gamma_1_meV: float,
    gamma_2_meV: float,
    string: str = "HA",
) -> str:
    string += "_"
    string += f"V_max={V_max_mV:.{V_tol_mV}f}mV_"
    string += f"dV={dV_mV:.{V_tol_mV}f}mV_"
    string += f"tau={tau:.{tau_tol}f}_"
    string += f"T={T_K:.{T_tol_K}f}K_"
    string += f"Delta=({Delta_1_meV:.{Delta_tol_meV}f},"
    string += f"{Delta_2_meV:.{Delta_tol_meV}f})meV_"
    string += f"gamma=({gamma_1_meV:.{gamma_tol_meV}f},"
    string += f"{gamma_2_meV:.{gamma_tol_meV}f})meV"
    return string


def cache_hash_pbar(
    tau: float,
    T_K: float,
    Delta_1_meV: float,
    Delta_2_meV: float,
    gamma_1_meV: float,
    gamma_2_meV: float,
    string: str = "FCS",
) -> str:
    string += "_"
    string += f"tau={tau:.{tau_tol}f}_"
    string += f"T={T_K:.{T_tol_K}f}K_"
    string += f"Delta=({Delta_1_meV:.{Delta_tol_meV}f},"
    string += f"{Delta_2_meV:.{Delta_tol_meV}f})meV_"
    string += f"gamma=({gamma_1_meV:.{gamma_tol_meV}f},"
    string += f"{gamma_2_meV:.{gamma_tol_meV}f})meV"
    return string
