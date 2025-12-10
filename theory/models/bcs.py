import numpy as np
import sys

sys.path.append("/Users/oliver/Documents/p5control-bluefors-evaluation")

from theory.utilities.constants import k_B_meV
from theory.utilities.constants import G_0_muS

from theory.utilities.functions import bin_y_over_x

from theory.utilities.types import NDArray64


def Delta_meV_of_T(Delta_meV: float, T_K: float) -> float:
    """Calculates the energy gap in eV at a given temperature."""

    T_C_K = Delta_meV / (1.76 * k_B_meV)  # Critical temperature in Kelvin
    if T_K < 0:
        raise ValueError("Temperature (K) must be non-negative.")
    if T_K >= T_C_K:
        # warnings.warn(f"Estimated T_C: {T_C_K:.2f} K", category=UserWarning)
        return 0.0
    elif T_K == 0:
        return Delta_meV
    else:
        # BCS theory: Delta(T) = Delta(0) * tanh(1.74 * sqrt(Tc/T - 1))
        return Delta_meV * np.tanh(1.74 * np.sqrt(T_C_K / T_K - 1))


def f_of_E(E_meV: NDArray64, T_K: float) -> NDArray64:
    """Fermi-Dirac distribution at zero and finite temperature."""
    if T_K < 0:
        raise ValueError("Temperature (K) must be non-negative.")
    elif T_K == 0:
        f = np.where(E_meV < 0, 1.0, 0.0)
    else:
        exponent = E_meV / (k_B_meV * T_K)
        exponent = np.clip(exponent, -100, 100)
        f = 1 / (np.exp(exponent) + 1)
    return f


def N_of_E(E_meV: NDArray64, Delta_meV: float, gamma_meV: float) -> NDArray64:
    """Computes the density of states for a superconductor using the Dynes model."""
    if Delta_meV < 0:
        raise ValueError("Energy gap (eV) must be non-negative.")
    if gamma_meV < 0:
        raise ValueError("Dynes parameter (eV) must be non-negative.")

    E_complex_meV = np.asarray(E_meV, dtype="complex128") + 1j * gamma_meV

    Delta_meV_2 = Delta_meV * Delta_meV
    E_complex_meV_2 = np.multiply(E_complex_meV, E_complex_meV)
    denom = np.sqrt(E_complex_meV_2 - Delta_meV_2)
    dos = np.divide(E_complex_meV, denom)
    dos = np.real(dos)

    dos = np.abs(dos, dtype="float64")
    dos[np.isnan(dos)] = 0.0
    dos = np.clip(dos, 0, 100.0)

    return dos


def get_I_nA(
    V_mV: NDArray64,
    Delta_meV: float | tuple[float, float] = (0.18, 0.18),
    G_N: float = 0.5,
    T_K: float = 0.0,
    gamma_meV: float | tuple[float, float] = 0.0,
    gamma_meV_min: float = 1e-4,
) -> NDArray64:

    G_N_muS = G_N * G_0_muS

    # Calculate Current, assuming Ohmic behavior
    I_NN_nA = V_mV * G_N_muS

    # take care of type and asymetric case

    if isinstance(Delta_meV, float):
        Delta1_meV, Delta2_meV = Delta_meV, Delta_meV
    elif isinstance(Delta_meV, tuple):
        Delta1_meV, Delta2_meV = Delta_meV
    else:
        raise KeyError("Delta_meV must be float | tuple[float, float]")

    if isinstance(gamma_meV, float):
        gamma1_meV, gamma2_meV = gamma_meV, gamma_meV
    elif isinstance(gamma_meV, tuple):
        gamma1_meV, gamma2_meV = gamma_meV
    else:
        raise KeyError("gamma_meV must be float | tuple[float, float]")

    Delta1_meV_T = Delta_meV_of_T(Delta_meV=Delta1_meV, T_K=T_K)
    Delta2_meV_T = Delta_meV_of_T(Delta_meV=Delta2_meV, T_K=T_K)

    gamma1_meV = gamma_meV_min if gamma1_meV < gamma_meV_min else gamma1_meV
    gamma2_meV = gamma_meV_min if gamma2_meV < gamma_meV_min else gamma2_meV

    Delta_meV_T_max = max(Delta1_meV_T, Delta2_meV_T)
    if Delta_meV_T_max == 0.0:
        return I_NN_nA

    # Determine stepsize in V and E
    dV_mV = np.abs(np.nanmax(V_mV) - np.nanmin(V_mV)) / (len(V_mV) - 1)
    V_max_mV = np.max(np.abs(V_mV))

    E_max_meV = np.max([Delta_meV_T_max * 10, V_max_mV])
    dE_meV = np.min([dV_mV, gamma_meV_min])

    # create V and E axis
    V_mV_temp = np.arange(0.0, V_max_mV + dV_mV, dV_mV, dtype="float64")
    E_meV = np.arange(-E_max_meV, E_max_meV + dE_meV, dE_meV, dtype="float64")

    # create meshes
    energy_eV_mesh, voltage_eV_mesh = np.meshgrid(E_meV, V_mV_temp / 2)
    energy1_eV_mesh = energy_eV_mesh - voltage_eV_mesh
    energy2_eV_mesh = energy_eV_mesh + voltage_eV_mesh

    # Calculate the Fermi-Dirac Distribution
    f_E_meV = f_of_E(E_meV=E_meV, T_K=T_K)
    f1 = np.interp(energy1_eV_mesh, E_meV, f_E_meV, left=1.0, right=0.0)
    f2 = np.interp(energy2_eV_mesh, E_meV, f_E_meV, left=1.0, right=0.0)
    integrand = f1 - f2

    if Delta1_meV_T > 0.0:
        n1 = N_of_E(
            E_meV=E_meV,
            Delta_meV=Delta1_meV_T,
            gamma_meV=gamma1_meV,
        )
        # Interpolate the shifted DOS
        N1 = np.interp(energy1_eV_mesh, E_meV, n1, left=1.0, right=1.0)
        integrand *= N1

    if Delta2_meV_T > 0.0:
        n2 = N_of_E(
            E_meV=E_meV,
            Delta_meV=Delta2_meV_T,
            gamma_meV=gamma2_meV,
        )
        N2 = np.interp(energy2_eV_mesh, E_meV, n2, left=1.0, right=1.0)
        integrand *= N2

    # Clean up the integrand
    integrand[np.isnan(integrand)] = 0.0

    # Do integration and normalization
    I_meV = np.trapezoid(integrand, E_meV, axis=1)
    I_nA = np.array(I_meV, dtype="float64") * G_N_muS

    # add negative voltage values
    I_nA = np.concatenate((I_nA, -np.flip(I_nA[1:])))
    V_mV_temp = np.concatenate((V_mV_temp, -np.flip(V_mV_temp[1:])))

    # bin over originally obtained V-axis
    I_nA = bin_y_over_x(V_mV_temp, I_nA, V_mV)

    # Fill up values, that are not calculated with dynes
    I_nA = np.where(
        np.abs(V_mV) >= 10 * Delta_meV_T_max,
        I_NN_nA,
        I_nA,
    )

    return I_nA
