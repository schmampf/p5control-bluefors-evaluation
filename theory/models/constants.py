# nature constants
from scipy.constants import e
from scipy.constants import h
from scipy.constants import Boltzmann as k_B

# e   = 1.602176634e-19 # A *     s
# h   = 6.62607015e-34  # A * V * s²
# k_B = 1.380649e-23    # J / K

h_e_Vs: float = h / e
G_0: float = 2 * e * e / h
k_B_eV: float = k_B / e

# h_e_Vs = 4.135667696923859e-15 # Vs
# G_0 = 7.748091729e-05          # A/V
# k_B_eV = 8.617333262145e-5     # V/K

h_e_pVs: float = h_e_Vs * 1e12
G_0_muS: float = G_0 * 1e6
k_B_meV: float = k_B_eV * 1e3

# h_e_pVs = 0.004135667696923859 # pVs
# G_0_muS = 77.48091729863648    # µS
# k_B_meV = 0.08617333262145178  # mV/K

# parameter tolerances
V_tol_mV: int = 6  # meV
tau_tol: int = 4
T_tol_K: int = 4  # K
Delta_tol_meV: int = 6  # meV
Gamma_tol_meV: int = 6  # meV
