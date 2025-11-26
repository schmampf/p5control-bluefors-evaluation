import numpy as np

from .functions import NDArray64
from .bcs import get_I_nA as get_I_nA_bcs
from .btk import get_I_nA as get_I_nA_btk

from .tg import get_I_pat_nA_from_I0_A0 as get_I_nA_tg
from .utg import get_I_nA as get_I_nA_utg


def get_I_nA(
    V_mV: NDArray64 = np.linspace(0, 0.6, 101, dtype="float64"),
    A_mV: int | float | NDArray64 = 0.0,
    tau: int | float | list[float] | NDArray64 = 0.0,
    tau_0: int | float = 0.0,
    T_K: int | float = 0,
    Delta_meV: int | float = 189e-3,
    Gamma_meV: int | float = 10e-3,
    nu_GHz: int | float = 10.0,
    Gamma_min_meV: float = 0.0001,
) -> NDArray64:

    if tau == 0.0 and tau_0 == 0.0:
        raise KeyError

    if tau_0 != 0.0:
        I_nA_0 = get_I_nA_bcs(
            V_mV=V_mV,
            Delta_meV=(0.0, Delta_meV),
            tau=tau_0,
            T_K=T_K,
            Gamma_meV=Gamma_meV,
            Gamma_meV_min=Gamma_min_meV,
        )

        if tau == 0.0 and A_mV != 0.0:
            I_nA_0 = get_I_nA_tg(
                A_mV=A_mV,
                V_mV=V_mV,
                I_nA=I_nA_0,
                nu_GHz=nu_GHz,
            )
    else:
        I_nA_0 = np.zeros_like(V_mV)

    if tau != 0.0:
        I_nA = get_I_nA_btk(
            V_mV=V_mV,
            Delta_meV=Delta_meV,
            tau=tau,
            T_K=T_K,
            Gamma_meV=Gamma_meV,
            Gamma_meV_min=Gamma_min_meV,
        )
        I_nA[:, 0] += I_nA_0
        I_nA[:, 1] += I_nA_0

        if A_mV != 0.0:
            I_nA = get_I_nA_utg(
                A_mV=np.array([A_mV], dtype="float64"),
                V_mV=V_mV,
                I_nA=I_nA,
                nu_GHz=nu_GHz,
                M=2,
            )[0, :, :]
    else:
        I_nA = I_nA_0

    return I_nA
