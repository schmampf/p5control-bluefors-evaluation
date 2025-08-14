"""
document sting
"""

from typing import Callable, Optional, TypeAlias

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.optimize import curve_fit  # type: ignore

NDArray64: TypeAlias = NDArray[np.float64]
ModelFunction: TypeAlias = Callable[..., ArrayLike]
ModelType: TypeAlias = tuple[ModelFunction, NDArray[np.bool]]


def optimizers(
    optimizer: str,
    function: ModelFunction,
    V_mV: NDArray64,
    I_nA: NDArray64,
    sigma: Optional[NDArray64],
    guess: NDArray64,
    lower: NDArray64,
    upper: NDArray64,
    maxfev: Optional[int],
) -> tuple[NDArray64, NDArray64, NDArray64]:
    """
    document sting

    maxfev : int, optional
        The maximum number of calls to the function. If zero, then
        ``100*(N+1)`` is the maximum where N is the number of elements
        in `x0`.
    """
    match optimizer:

        case "curve_fit":
            results: tuple[NDArray64, NDArray64] = curve_fit(
                f=function,
                xdata=V_mV,
                ydata=I_nA,
                sigma=sigma,
                absolute_sigma=True,
                p0=guess,
                bounds=(lower, upper),
                maxfev=maxfev,
            )
            popt: NDArray64 = np.array(results[0], dtype=np.float64)
            pcov: NDArray64 = np.array(results[1], dtype=np.float64)
            perr = np.sqrt(np.diag(pcov))

        # case "curve_fit_jax":
        #     from jaxfit import CurveFit  # type: ignore
        #     jcf = CurveFit()
        #     popt, pcov, *_ = jcf.curve_fit(  # type: ignore
        #         f=function,
        #         xdata=V_mV,
        #         ydata=I_nA,
        #         sigma=sigma,
        #         absolute_sigma=True,
        #         p0=guess,
        #         bounds=(lower, upper),
        #         maxfev=maxfev,
        #     )
        #     perr = np.sqrt(np.diag(pcov))

        case _:
            raise KeyError("Optimizer not found.")
    return popt, pcov, perr
