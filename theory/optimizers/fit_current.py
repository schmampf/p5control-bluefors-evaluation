"""
Module Doc String
"""

import importlib
import sys
from typing import Optional, Callable, TypeAlias

import numpy as np
from numpy.typing import NDArray, ArrayLike

from theory.optimizers.models import models
from theory.optimizers.optimizers import optimizers

importlib.reload(sys.modules["theory.optimizers.models"])
importlib.reload(sys.modules["theory.optimizers.optimizers"])

NDArray64: TypeAlias = NDArray[np.float64]
ParameterType: TypeAlias = tuple[float, tuple[float, float], bool]
DictType: TypeAlias = dict[str, float | NDArray[np.float64 | np.bool] | str | None]
ModelFunction: TypeAlias = Callable[..., ArrayLike]
ModelType: TypeAlias = tuple[ModelFunction, NDArray[np.bool]]


def fit_current(
    V_mV: NDArray64,
    I_nA: NDArray64,
    tau: ParameterType = (1.0, (0, 10.0), False),
    T_K: ParameterType = (0.2, (0, 1.5), False),
    Delta_mV: ParameterType = (0.195, (0.18, 0.21), False),
    Gamma_mV: ParameterType = (1e-3, (1e-3, 25e-3), False),
    A_mV: ParameterType = (1.0, (0, 10.0), False),
    nu_GHz: ParameterType = (7.8, (1.0, 20.0), False),
    E_mV: Optional[NDArray64] = None,
    sigma: Optional[NDArray64] = None,
    model: str = "dynes",
    optimizer: str = "curve_fit_jax",
    maxfev: Optional[int] = None,
) -> DictType:
    """
    Doc String
    """

    # Define Parameter
    tau_0, (tau_lower, tau_upper), tau_fixed = tau
    T_K_0, (T_K_lower, T_K_upper), T_K_fixed = T_K
    Delta_mV_0, (Delta_mV_lower, Delta_mV_upper), Delta_mV_fixed = Delta_mV
    Gamma_mV_0, (Gamma_mV_lower, Gamma_mV_upper), Gamma_mV_fixed = Gamma_mV
    A_mV_0, (A_mV_lower, A_mV_upper), A_mV_fixed = A_mV
    nu_GHz_0, (nu_GHz_lower, nu_GHz_upper), nu_GHz_fixed = nu_GHz

    guess_full: NDArray64 = np.array(
        [
            tau_0,
            T_K_0,
            Delta_mV_0,
            Gamma_mV_0,
            A_mV_0,
            nu_GHz_0,
        ],
        dtype="float64",
    )

    lower_full: NDArray64 = np.array(
        [
            tau_lower,
            T_K_lower,
            Delta_mV_lower,
            Gamma_mV_lower,
            A_mV_lower,
            nu_GHz_lower,
        ],
        dtype="float64",
    )

    upper_full: NDArray64 = np.array(
        [
            tau_upper,
            T_K_upper,
            Delta_mV_upper,
            Gamma_mV_upper,
            A_mV_upper,
            nu_GHz_upper,
        ],
        dtype="float64",
    )

    fixed: NDArray[np.bool] = np.array(
        [
            tau_fixed,
            T_K_fixed,
            Delta_mV_fixed,
            Gamma_mV_fixed,
            A_mV_fixed,
            nu_GHz_fixed,
        ],
        dtype="bool",
    )

    # get model
    chosen_model: ModelType = models(model=model, E_mV=E_mV)
    function: ModelFunction = chosen_model[0]
    parameter_mask: NDArray[np.bool] = chosen_model[1]

    free_mask = parameter_mask & ~fixed

    guess = guess_full[free_mask]
    lower = lower_full[free_mask]
    upper = upper_full[free_mask]

    def fixed_function(V_mV: NDArray64, *free_params: tuple[float, ...]) -> ArrayLike:
        full_params = guess_full.copy()
        full_params[free_mask] = free_params
        return function(V_mV, *full_params[parameter_mask])

    # optimize with optimizer
    popt, pcov, perr = optimizers(
        optimizer=optimizer,
        function=fixed_function,
        V_mV=V_mV,
        I_nA=I_nA,
        sigma=sigma,
        guess=guess,
        lower=lower,
        upper=upper,
        maxfev=maxfev,
    )

    popt_full: NDArray64 = guess_full.copy()
    popt_full[free_mask] = popt

    pcov_full: NDArray64 = np.zeros((len(fixed), len(fixed)), dtype=np.float64)
    pcov_full[np.ix_(free_mask, free_mask)] = pcov

    perr_full: NDArray64 = np.full_like(fixed, 0.0, dtype=np.float64)
    perr_full[free_mask] = perr

    I_exp_nA: NDArray64 = I_nA
    I_ini_nA: NDArray64 = np.array(fixed_function(V_mV, *guess), dtype=np.float64)
    I_fit_nA: NDArray64 = np.array(fixed_function(V_mV, *popt), dtype=np.float64)

    solution: DictType = {
        "optimizer": optimizer,
        "model": model,
        "V_mV": V_mV,
        "I_exp_nA": I_exp_nA,
        "I_ini_nA": I_ini_nA,
        "I_fit_nA": I_fit_nA,
        "guess": guess_full,
        "lower": lower_full,
        "upper": upper_full,
        "fixed": fixed,
        "popt": popt_full,
        "pcov": pcov_full,
        "perr": perr_full,
        "E_mV": E_mV,
        "sigma": sigma,
        "maxfev": maxfev,
        "tau": popt_full[0],
        "T_K": popt_full[1],
        "Delta_mV": popt_full[2],
        "Gamma_mV": popt_full[3],
        "A_mV": popt_full[4],
        "nu_GHz": popt_full[5],
    }

    return solution
