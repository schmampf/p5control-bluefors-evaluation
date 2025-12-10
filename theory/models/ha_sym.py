import io
import numpy as np
import os
import subprocess
import sys

from numpy.typing import NDArray
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed


import theory.models.carlosha.ha_sym as ha_sym

from theory.models.bcs import Delta_meV_of_T

from theory.models.functions import cache_hash_sym
from theory.models.functions import bin_y_over_x

from theory.models.constants import G_0_muS
from theory.models.constants import k_B_meV
from theory.models.constants import V_tol_mV
from theory.models.constants import tau_tol
from theory.models.constants import T_tol_K
from theory.models.constants import Delta_tol_meV
from theory.models.constants import gamma_tol_meV

HOME_DIR = "/Users/oliver/Documents/p5control-bluefors-evaluation"
sys.path.append(HOME_DIR)

WORK_DIR = os.path.join(HOME_DIR, "theory/models/carlosha/")
CACHE_DIR = os.path.join(WORK_DIR, ".cache_sym")
os.makedirs(CACHE_DIR, exist_ok=True)


def run_ha_sym(
    V_mV: NDArray[np.float64],
    tau: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    Delta_T_meV = Delta_meV_of_T(Delta_meV, T_K)
    T_Delta = k_B_meV * T_K / Delta_T_meV
    gamma_Delta = gamma_meV / Delta_T_meV
    V_Delta = V_mV / Delta_T_meV

    E_max: float = np.max(
        [
            10.0 * Delta_meV / Delta_T_meV,
            np.max(V_mV) / Delta_T_meV,
        ]
    )

    I_nA = np.array(
        ha_sym.ha_sym_curve(
            tau,
            T_Delta,
            gamma_Delta,
            -E_max,
            E_max,
            V_Delta,
        ),
        dtype=np.float64,
    )
    I_nA *= Delta_T_meV * G_0_muS
    return V_mV, I_nA


def run_multiple_ha_sym(
    V_mV: NDArray[np.float64],
    tau: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
    n_worker: int = 16,
) -> NDArray[np.float64]:

    V_chunks = np.array_split(V_mV, n_worker)
    with ThreadPoolExecutor(max_workers=n_worker) as executor:
        futures: list = []
        for i in range(n_worker):
            futures.append(
                executor.submit(
                    run_ha_sym,
                    V_mV=V_chunks[i],
                    tau=tau,
                    T_K=T_K,
                    Delta_meV=Delta_meV,
                    gamma_meV=gamma_meV,
                )
            )

        I_nA = np.array([], dtype="float64")
        V_mV = np.array([], dtype="float64")

        for future in as_completed(futures):
            v_mV, i_nA = future.result()
            V_mV = np.concatenate((V_mV, v_mV))
            I_nA = np.concatenate((I_nA, i_nA))

        sorting = np.argsort(V_mV)

    return I_nA[sorting]


def get_I_nA(
    V_mV: NDArray[np.float64],
    tau: float = 0.5,
    T_K: float = 0.0,
    Delta_meV: float = 0.18,
    gamma_meV: float = 1e-4,
    n_worker: int = 16,
    caching: bool = True,
) -> NDArray[np.float64]:

    if tau == 0.0:
        return np.zeros_like(V_mV)

    Delta_T_meV = Delta_meV_of_T(Delta_meV, T_K)
    if Delta_T_meV == 0.0:
        return V_mV * G_0_muS * tau

    # voltage axis
    V_0_mV = V_mV
    V_max_mV = np.max(np.abs(V_0_mV))
    dV_mV = np.abs(np.nanmax(V_0_mV) - np.nanmin(V_0_mV)) / (V_0_mV.shape[0] - 1)
    V_mV = np.arange(dV_mV, V_max_mV + dV_mV, dV_mV, dtype="float64")

    cached_file: str = "dump.pyz"

    if caching:
        V_mV = np.round(V_mV, decimals=V_tol_mV)
        tau = np.round(tau, decimals=tau_tol)
        T_K = np.round(T_K, decimals=T_tol_K)
        Delta_meV = np.round(Delta_meV, decimals=Delta_tol_meV)
        gamma_meV = np.round(gamma_meV, decimals=gamma_tol_meV)

        cache_key = cache_hash_sym(
            V_max_mV=V_max_mV,
            dV_mV=dV_mV,
            tau=tau,
            T_K=T_K,
            Delta_meV=Delta_meV,
            gamma_meV=gamma_meV,
            string="ha_sym",
        )
        cached_file = os.path.join(CACHE_DIR, f"{cache_key}.npz")

    if os.path.exists(cached_file) and caching:
        cache_data = np.load(cached_file)
        V_mV = cache_data["V_mV"].astype("float64")
        I_nA: NDArray[np.float64] = cache_data["I_nA"].astype("float64")
    else:
        I_nA = run_multiple_ha_sym(
            V_mV=V_mV,
            tau=tau,
            T_K=T_K,
            Delta_meV=Delta_meV,
            gamma_meV=gamma_meV,
            n_worker=n_worker,
        )

        if caching:
            # save to cache
            np.savez(
                cached_file,
                V_mV=V_mV,
                I_nA=I_nA,
                tau=tau,
                T_K=T_K,
                Delta_meV=Delta_meV,
                gamma_meV=gamma_meV,
            )

    # make symmetric
    V_mV = np.concatenate((V_mV, np.zeros((1)), -V_mV))
    I_nA = np.concatenate((I_nA, np.zeros((1)), -I_nA))

    I_nA = bin_y_over_x(
        x=V_mV,
        y=I_nA,
        x_bins=V_0_mV,
    )
    return I_nA
