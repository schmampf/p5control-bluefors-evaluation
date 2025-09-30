from typing import Callable, Optional, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, config
from numpy.typing import NDArray, ArrayLike

from theory.models.PAT import get_I_pat_nA_from_I0_A0
from theory.models.dynes_jnp import G_0_muS_jax, currents, thermal_energy_gap

jax.config.update("jax_enable_x64", True)

NDArray64: TypeAlias = NDArray[np.float64]
ModelFunction: TypeAlias = Callable[..., ArrayLike]
ModelType: TypeAlias = tuple[ModelFunction, NDArray[np.bool]]


def models(
    model: str = "dynes", E_mV: Optional[NDArray64] = None, N: Optional[int] = None
) -> ModelType:

    if E_mV is None:
        E_mV: NDArray64 = np.linspace(-2.0, 2.0, 2001)
    E_mV_jax: Array = jnp.array(E_mV, dtype=jnp.float64)

    if N is None:
        N: int = 100

    @jit
    def get_dynes_jnp(
        V_mV: Array,
        tau: Array,
        T_K: Array,
        Delta_mV: Array,
        Gamma_mV: Array,
    ) -> Array:
        Delta_T_mV: Array = thermal_energy_gap(Delta_meV=Delta_mV, T_K=T_K)
        I_mV: Array = currents(
            V_meV=V_mV,
            E_meV=E_mV_jax,
            T_K=T_K,
            Delta_meV=Delta_T_mV,
            Gamma_meV=Gamma_mV,
        )
        I_nA: Array = I_mV * tau * G_0_muS_jax
        return I_nA

    match model:
        case "dynes":

            def get_dynes(
                V_mV: NDArray64,
                tau: float,
                T_K: float,
                Delta_mV: float,
                Gamma_mV: float,
            ) -> NDArray64:
                I_nA_jax: Array = get_dynes_jnp(
                    V_mV=jnp.array(V_mV, dtype=jnp.float64),
                    tau=jnp.array(tau, dtype=jnp.float64),
                    T_K=jnp.array(T_K, dtype=jnp.float64),
                    Delta_mV=jnp.array(Delta_mV, dtype=jnp.float64),
                    Gamma_mV=jnp.array(Gamma_mV, dtype=jnp.float64),
                )
                I_nA: NDArray64 = np.array(I_nA_jax, dtype=np.float64)
                return I_nA

            # Mask which parameter are used
            parameter_mask: NDArray[np.bool] = np.array(
                [True, True, True, True, False, False]
            )
            return get_dynes, parameter_mask

        case "dynes+pat":

            def get_dynes_pat(
                V_mV: NDArray64,
                tau: float,
                T_K: float,
                Delta_mV: float,
                Gamma_mV: float,
                A_mV: float,
                nu_GHz: float,
            ) -> NDArray64:
                V_mV_jax = jnp.array(V_mV, dtype=jnp.float64)

                I_dynes_jax: Array = get_dynes_jnp(
                    V_mV=V_mV_jax,
                    tau=jnp.array(tau, dtype=jnp.float64),
                    T_K=jnp.array(T_K, dtype=jnp.float64),
                    Delta_mV=jnp.array(Delta_mV, dtype=jnp.float64),
                    Gamma_mV=jnp.array(Gamma_mV, dtype=jnp.float64),
                )
                I_dynes: NDArray64 = np.array(I_dynes_jax, dtype=np.float64)
                I_dynes_pat: NDArray64 = get_I_pat_nA_from_I0_A0(
                    V_mV=V_mV,
                    I_nA=I_dynes,
                    A_mV=A_mV,
                    nu_GHz=nu_GHz,
                    N=N,
                )
                return I_dynes_pat

            # Mask which parameter are used
            parameter_mask: NDArray[np.bool] = np.full((6), True, dtype=np.bool)
            return get_dynes_pat, parameter_mask

        case _:
            raise KeyError("model not found.")
