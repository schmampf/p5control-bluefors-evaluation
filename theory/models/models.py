import numpy as np
import numpy.typing as npt
from tqdm import tqdm

import sys

HOME_DIR = "/Users/oliver/Documents/p5control-bluefors-evaluation/"
sys.path.append(HOME_DIR)

# import models
from theory.models.bcs_jnp import get_I_nA as get_I_nA_dynes
from theory.models.ha_asym import get_I_nA as get_I_nA_HA
from theory.models.fcs_pbar import get_I_nA as get_I_nA_FCS_pbar

# number of maximum charges
from theory.models.constants import m_max

# from .dynes_np import get_current_dynes as get_I_nA_dynes_np
# from .full_counting_statistic import get_I_nA as get_I_nA_FCS
# from .PAT import get_I_nA as get_I_nA_PAT
# from .PAMAR import get_I_nA as get_I_nA_PAMAR


def get_I_nA(
    V_mV: npt.NDArray[np.float64],
    tau: int | float | npt.NDArray[np.float64] = 0.5,
    T_K: int | float | npt.NDArray[np.float64] = 0,
    Delta_meV: (
        float | tuple[float, float] | list[float] | npt.NDArray[np.float64]
    ) = 189e-3,
    Gamma_meV: float | tuple[float, float] | npt.NDArray[np.float64] = 10e-3,
    model: str = "dynes",
    Gamma_min_meV: float = 0.0001,
):
    # Typecheck V_mV
    if not (isinstance(V_mV, np.ndarray) and V_mV.dtype == np.float64):
        raise TypeError("V_mV must be a np.array filled with floats.")

    # Typecheck tau
    if isinstance(tau, int | float):
        tau = np.array([tau], dtype="float64")
    if not (isinstance(tau, np.ndarray) and tau.dtype == np.float64):
        raise TypeError("tau must be either a float or an array of floats.")

    # Typecheck T_K
    if isinstance(T_K, int | float):
        T_K = np.array([T_K], dtype="float64")
    if not (isinstance(T_K, np.ndarray) and T_K.dtype == np.float64):
        raise TypeError("T_K must be either a float or an array of floats.")

    # Typecheck Delta_meV
    if isinstance(Delta_meV, float):
        Delta_meV = np.array([Delta_meV, Delta_meV], dtype=np.float64)
    if isinstance(Delta_meV, tuple | list) and len(Delta_meV) == 2:
        Delta_meV = np.array([Delta_meV[0], Delta_meV[1]], dtype=np.float64)
    if not (
        isinstance(Delta_meV, np.ndarray)
        and Delta_meV.dtype == np.float64
        and Delta_meV.shape == (2,)
    ):
        raise TypeError(
            "Delta_meV must be a float or a list, tuple or np.array filled with two floats."
        )

    # Typecheck Gamma_meV
    if isinstance(Gamma_meV, float):
        Gamma_meV = np.array([Gamma_meV, Gamma_meV], dtype=np.float64)
    if isinstance(Gamma_meV, tuple | list) and len(Gamma_meV) == 2:
        Gamma_meV = np.array([Gamma_meV[0], Gamma_meV[1]], dtype=np.float64)
    if not (
        isinstance(Gamma_meV, np.ndarray)
        and Gamma_meV.dtype == np.float64
        and Gamma_meV.shape == (2,)
    ):
        raise TypeError(
            "Gamma_meV must be a float or a list, tuple or np.array filled with two floats."
        )

    match model:
        case "dynes" | "Dynes" | "D":
            I_nA = np.full((T_K.shape[0], V_mV.shape[0]), np.nan, dtype="float64")
            for i, T in enumerate(tqdm(T_K, desc="Dynes: ")):
                I_nA[i, :] = get_I_nA_dynes(
                    V_mV=V_mV,
                    T_K=T,
                    Delta_meV=Delta_meV,
                    Gamma_meV=Gamma_meV,
                    Gamma_min_meV=Gamma_min_meV,
                )
            tau = tau[:, np.newaxis, np.newaxis]
            I_nA = I_nA[np.newaxis, :, :]
            I_nA = tau * I_nA

        case "ha" | "HA" | "hamiltonian" | "hamiltonian approach":
            I_nA = np.full(
                (tau.shape[0], T_K.shape[0], V_mV.shape[0]), np.nan, dtype="float64"
            )
            total = len(tau) * len(T_K)
            for idx, (i, t_i, j, T_j) in enumerate(
                tqdm(
                    (
                        (i, t, j, T)
                        for i, t in enumerate(tau)
                        for j, T in enumerate(T_K)
                    ),
                    total=total,
                    desc="HA: ",
                )
            ):
                I_nA[i, j, :] = get_I_nA_HA(
                    V_mV=V_mV,
                    tau=t_i,
                    T_K=T_j,
                    Delta_meV=Delta_meV,
                    Gamma_meV=Gamma_meV,
                )

        case "fcs" | "FCS" | "full counting" | "full counting statistics":
            I_nA = np.full(
                (tau.shape[0], T_K.shape[0], V_mV.shape[0], m_max + 1),
                np.nan,
                dtype="float64",
            )
            print(f"Iterations: {tau.shape[0]*T_K.shape[0]}")
            for i, tau_i in enumerate(tau):
                for j, T_K_j in enumerate(T_K):
                    I_nA[i, j, :, :] = get_I_nA_FCS_pbar(
                        V_mV=V_mV,
                        tau=tau_i,
                        T_K=T_K_j,
                        Delta_meV=Delta_meV,
                        Gamma_meV=Gamma_meV,
                    )

        case _:
            raise KeyError(
                "model must be either 'dynes' ('d'), 'hamiltonian approach' ('ha'), 'full counting statistics'('fcs')"
            )

    I_nA = np.squeeze(I_nA)
    return I_nA
