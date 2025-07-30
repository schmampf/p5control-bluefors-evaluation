import io
import numpy as np
import os
import subprocess
import sys

from numpy.typing import NDArray
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

HOME_DIR = "/Users/oliver/Documents/p5control-bluefors-evaluation"
sys.path.append(HOME_DIR)

from theory.models.functions import cache_hash
from theory.models.functions import bin_y_over_x

from theory.models.constants import V_tol_mV
from theory.models.constants import tau_tol
from theory.models.constants import T_tol_K
from theory.models.constants import Delta_tol_meV
from theory.models.constants import Gamma_tol_meV

WORK_DIR = os.path.join(HOME_DIR, "theory/models/carlosfcs/")
CACHE_DIR = os.path.join(WORK_DIR, ".cache")
FCS_EXE = os.path.join(WORK_DIR, "fcs")
os.makedirs(CACHE_DIR, exist_ok=True)

# number of maximum charges
from theory.models.constants import m_max
from theory.models.constants import iw
from theory.models.constants import nchi


def run_fcs(
    Vi_mV: float,
    Vf_mV: float,
    dV_mV: float,
    tau: float,
    T_K: float,
    Delta_1_meV: float,
    Delta_2_meV: float,
    Gamma_1_meV: float,
    Gamma_2_meV: float,
) -> NDArray[np.float64]:

    string = ""
    string += f"{tau:.{tau_tol}f}\n"  # [0, 1]
    string += f"{T_K:.{T_tol_K}f}\n"  # K
    string += f"{Delta_1_meV:.{Delta_tol_meV}f} {Delta_2_meV:.{Delta_tol_meV}f}\n"  # mV
    string += f"{Gamma_1_meV:.{Gamma_tol_meV}f} {Gamma_2_meV:.{Gamma_tol_meV}f}\n"  # mV
    string += f"{Vi_mV:.{V_tol_mV}f} {Vf_mV:.{V_tol_mV}f} {dV_mV:.{V_tol_mV}f} \n"  # mV
    string += f"{m_max} {iw} {nchi}"

    proc = subprocess.run(
        [FCS_EXE],
        input=string,
        capture_output=True,
        text=True,
        cwd=WORK_DIR,
        check=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)

    data = np.genfromtxt(io.StringIO(proc.stdout), dtype="float64")
    if data.size == 0:
        raise RuntimeError(
            "Fortran code produced no output. Check input sweep range and step."
        )

    return data


def run_multiple_fcs(
    V_max_mV: float,
    dV_mV: float,
    tau: float,
    T_K: float,
    Delta_1_meV: float,
    Delta_2_meV: float,
    Gamma_1_meV: float,
    Gamma_2_meV: float,
    n_worker: int = 16,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    chunk = int(np.ceil((V_max_mV / dV_mV + 1) / n_worker))
    with ThreadPoolExecutor(max_workers=n_worker) as executor:
        futures = []
        for i in range(n_worker):
            Vi_mV = i * dV_mV * chunk - dV_mV
            Vf_mV = (i + 1) * dV_mV * chunk
            futures.append(
                executor.submit(
                    run_fcs,
                    Vi_mV=Vi_mV,
                    Vf_mV=Vf_mV,
                    dV_mV=dV_mV,
                    tau=tau,
                    T_K=T_K,
                    Delta_1_meV=Delta_1_meV,
                    Delta_2_meV=Delta_2_meV,
                    Gamma_1_meV=Gamma_1_meV,
                    Gamma_2_meV=Gamma_2_meV,
                )
            )

        all_V_mV = np.full((1), np.nan, dtype="float64")
        all_I_nA = np.full((1, m_max + 1), np.nan, dtype="float64")

        for future in as_completed(futures):
            result = future.result()
            all_V_mV = np.concatenate((all_V_mV, result[:, 0]), axis=0)
            all_I_nA = np.concatenate((all_I_nA, result[:, 1:]), axis=0)

    return all_V_mV, all_I_nA


def get_I_nA(
    V_mV: NDArray[np.float64],
    tau: float = 0.5,
    T_K: float = 0.0,
    Delta_meV: NDArray[np.float64] = np.array([2e-3, 2e-3]),
    Gamma_meV: NDArray[np.float64] = np.array([1e-4, 1e-4]),
    n_worker: int = 16,
) -> NDArray[np.float64]:

    if tau == 0.0:
        return np.zeros((V_mV.shape[0], m_max + 1))

    V_mV = np.round(V_mV, decimals=V_tol_mV)
    tau = np.round(tau, decimals=tau_tol)
    T_K = np.round(T_K, decimals=T_tol_K)
    Delta_meV = np.round(Delta_meV, decimals=Delta_tol_meV)
    Gamma_meV = np.round(Gamma_meV, decimals=Gamma_tol_meV)

    # voltage axis
    V_0_mV = V_mV
    V_max_mV = np.max(np.abs(V_0_mV))
    dV_mV = np.abs(np.nanmax(V_0_mV) - np.nanmin(V_0_mV)) / (V_0_mV.shape[0] - 1)

    cache_key = cache_hash(
        V_max_mV=V_max_mV,
        dV_mV=dV_mV,
        tau=tau,
        T_K=T_K,
        Delta_1_meV=Delta_meV[0],
        Delta_2_meV=Delta_meV[1],
        Gamma_1_meV=Gamma_meV[0],
        Gamma_2_meV=Gamma_meV[1],
        string="FCS",
    )
    cached_file = os.path.join(CACHE_DIR, f"{cache_key}.npz")

    if os.path.exists(cached_file):
        cache_data = np.load(cached_file)
        V_mV = np.round(cache_data["V_mV"], decimals=V_tol_mV)
        I_nA = np.round(cache_data["I_nA"], decimals=V_tol_mV)
    else:
        V_mV, I_nA = run_multiple_fcs(
            V_max_mV=V_max_mV,
            dV_mV=dV_mV,
            tau=tau,
            T_K=T_K,
            Delta_1_meV=Delta_meV[0],
            Delta_2_meV=Delta_meV[1],
            Gamma_1_meV=Gamma_meV[0],
            Gamma_2_meV=Gamma_meV[1],
            n_worker=n_worker,
        )

        # save to cache
        np.savez(
            cached_file,
            V_mV=V_mV,
            I_nA=I_nA,
            tau=tau,
            T_K=T_K,
            Delta_1_meV=Delta_meV[0],
            Delta_2_meV=Delta_meV[1],
            Gamma_1_meV=Gamma_meV[0],
            Gamma_2_meV=Gamma_meV[1],
        )

    # make symmetric
    V_mV = np.concatenate((V_mV, -V_mV), axis=0)
    I_nA = np.concatenate((I_nA, -I_nA), axis=0)

    I_na_bin = np.full((V_0_mV.shape[0], m_max + 1), np.nan, dtype="float64")
    for m in range(m_max):
        I_na_bin[:, m] = bin_y_over_x(
            x=V_mV,
            y=I_nA[:, m],
            x_bins=V_0_mV,
        )
    return I_na_bin
