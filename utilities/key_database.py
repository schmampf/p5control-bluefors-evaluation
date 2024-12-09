"""
Database for POSSIBLE_MEASUREMENT_KEYS and PLOT_KEYS.

"""

POSSIBLE_MEASUREMENT_KEYS = {
    "magnetic_fields": [7, -2, 1e-3, "no_field"],
    "temperatures": [7, -3, 1e-6, "no_heater"],
    "temperatures_up": [7, -2, 1e-6, "no_heater"],
    "gate_voltages": [5, -2, 1e-3, "no_gate"],
}

PLOT_KEYS = {}

### X - Achsen ###

# V_bias #
PLOT_KEYS["V_bias_up_µV"] = [
    "self.mapped['voltage_axis']*1e6",
    r"$V_\mathrm{Bias}^\rightarrow$ (µV)",
]
PLOT_KEYS["V_bias_up_V"] = [
    "self.mapped['voltage_axis']*1e0",
    r"$V_\mathrm{Bias}^\rightarrow$ (mV)",
]
PLOT_KEYS["V_bias_up_mV"] = [
    "self.mapped['voltage_axis']*1e3",
    r"$V_\mathrm{Bias}^\rightarrow$ (V)",
]
PLOT_KEYS["V_bias_down_V"] = [
    "self.mapped['voltage_axis']*1e0",
    r"$V_\mathrm{Bias}^\leftarrow$ (V)",
]
PLOT_KEYS["V_bias_down_mV"] = [
    "self.mapped['voltage_axis']*1e3",
    r"$V_\mathrm{Bias}^\leftarrow$ (mV)",
]
PLOT_KEYS["V_bias_down_µV"] = [
    "self.mapped['voltage_axis']*1e6",
    r"$V_\mathrm{Bias}^\leftarrow$ (µV)",
]


### Y - Achsen ###

# arbitrary y-axis #
PLOT_KEYS["y_axis"] = ["self.mapped['y_axis']", r"$y$ (arb. u.)"]

# heater power #
PLOT_KEYS["heater_power_µW"] = [
    "self.mapped['y_axis']*1e6",
    r"$P_\mathrm{Heater}$ (µW)",
]
PLOT_KEYS["heater_power_mW"] = [
    "self.mapped['y_axis']*1e3",
    r"$P_\mathrm{Heater}$ (mW)",
]

# magnetic field #
PLOT_KEYS["uH_up_mT"] = ["self.mapped['y_axis']*1e3", r"$\mu_0H^\rightarrow$ (mT)"]
PLOT_KEYS["uH_up_T"] = ["self.mapped['y_axis']", r"$\mu_0H^\rightarrow$ (T)"]
PLOT_KEYS["uH_down_mT"] = ["self.mapped['y_axis']*1e3", r"$\mu_0H^\leftarrow$ (mT)"]
PLOT_KEYS["uH_down_T"] = ["self.mapped['y_axis']", r"$\mu_0H^\leftarrow$ (T)"]
PLOT_KEYS["uH_mT"] = ["self.mapped['y_axis']*1e3", r"$\mu_0H$ (mT)"]
PLOT_KEYS["uH_T"] = ["self.mapped['y_axis']", r"$\mu_0H$ (T)"]

# gate voltage #
PLOT_KEYS["V_gate_up_V"] = [
    "self.mapped['y_axis']",
    r"$V_\mathrm{Gate}^\rightarrow$ (V)",
]
PLOT_KEYS["V_gate_down_V"] = [
    "self.mapped['y_axis']",
    r"$V_\mathrm{Gate}^\leftarrow$ (V)",
]
PLOT_KEYS["V_gate_V"] = ["self.mapped['y_axis']", r"$V_\mathrm{Gate}$ (V)"]
PLOT_KEYS["V_gate_up_mV"] = [
    "self.mapped['y_axis']*1e3",
    r"$V_\mathrm{Gate}^\rightarrow$ (mV)",
]
PLOT_KEYS["V_gate_down_mV"] = [
    "self.mapped['y_axis']*1e3",
    r"$V_\mathrm{Gate}^\leftarrow$ (mV)",
]
PLOT_KEYS["V_gate_mV"] = ["self.mapped['y_axis']*1e3", r"$V_\mathrm{Gate}$ (mV)"]

# microwave voltage #
PLOT_KEYS["V_ac_up_V"] = ["self.mapped['y_axis']", r"$V_\mathrm{AC}^\rightarrow$ (V)"]

# microwave frequency #
PLOT_KEYS["nu_up"] = [
    "self.mapped['y_axis']*1e-9",
    r"$\nu_\mathrm{AC}^\rightarrow$ (GHz)",
]


### Z - Achsen ###

# dIdV #
PLOT_KEYS["dIdV_up"] = [
    "self.mapped['differential_conductance_up']",
    r"d$I/$d$V$ ($G_0$)",
]
PLOT_KEYS["dIdV_down"] = [
    "self.mapped['differential_conductance_down']",
    r"d$I/$d$V$ ($G_0$)",
]

# current #
PLOT_KEYS["I_up_nA_abs"] = ["self.mapped['current_up']*1e9", r"$|I|$ (nA)"]

# temperatures #
PLOT_KEYS["T_all_up_K"] = [
    "self.mapped['temperature_all_up']",
    r"$T_\mathrm{Sample}$ (K)",
]

# derived #
PLOT_KEYS["dIdV_up_asym"] = [
    "self.mapped['differential_conductance_up']-np.flip(self.mapped['differential_conductance_up'], axis=1)",
    r"$($d$I/$d$V)^\rightarrow-($d$I/$d$V)^\leftarrow$ ($G_0$)",
]
PLOT_KEYS["T_all_norm_up"] = [
    "self.mapped['temperature_all_up']/self.mapped['temperature_mean_up'].reshape(-1,1)",
    r"$T_\mathrm{Sample}(V_\mathrm{Bias}, V_\mathrm{AC})/T_\mathrm{Sample}(V_\mathrm{AC})$",
]
PLOT_KEYS["T_all_norm_down"] = [
    "self.mapped['temperature_all_down']/self.mapped['temperature_mean_down'].reshape(-1,1)",
    r"$T_\mathrm{Sample}(V_\mathrm{Bias}, V_\mathrm{AC})/T_\mathrm{Sample}(V_\mathrm{AC})$",
]


### N - Achsen ###
PLOT_KEYS["time_up"] = ["self.mapped['time_up']", r"time"]
PLOT_KEYS["T_up_mK"] = [
    "self.mapped['temperature_mean_up']*1e3",
    r"$T_\mathrm{Sample}$ (mK)",
]
PLOT_KEYS["T_up_K"] = [
    "self.mapped['temperature_mean_up']*1e0",
    r"$T_\mathrm{Sample}$ (K)",
]


### mapped_over_temperature ###
PLOT_KEYS["dIdV_up_T"] = [
    "self.mapped_over_temperature['differential_conductance_up']",
    r"d$I/$d$V$ ($G_0$)",
]
PLOT_KEYS["dIdV_down_T"] = [
    "self.mapped_over_temperature['differential_conductance_down']",
    r"d$I/$d$V$ ($G_0$)",
]
PLOT_KEYS["I_up_T_nA_abs"] = [
    "self.mapped_over_temperature['current_up']*1e9",
    r"$|I|$ (nA)",
]
PLOT_KEYS["heater_power_µW_up"] = [
    "self.mapped_over_temperature['y_axis_up']*1e6",
    r"$P_\mathrm{Heater}$ (µW)",
]
PLOT_KEYS["heater_power_µW_down"] = [
    "self.mapped_over_temperature['y_axis_down']*1e6",
    r"$P_\mathrm{Heater}$ (µW)",
]
PLOT_KEYS["heater_power_mW_up"] = [
    "self.mapped_over_temperature['y_axis_up']*1e3",
    r"$P_\mathrm{Heater}$ (mW)",
]
PLOT_KEYS["heater_power_mW_down"] = [
    "self.mapped_over_temperature['y_axis_down']*1e3",
    r"$P_\mathrm{Heater}$ (mW)",
]
PLOT_KEYS["T_axis_up_K"] = [
    "self.mapped_over_temperature['temperature_axis']",
    r"$T_\mathrm{Sample}^\rightarrow$ (K)",
]
PLOT_KEYS["T_axis_down_K"] = [
    "self.mapped_over_temperature['temperature_axis']",
    r"$T_\mathrm{Sample}^\leftarrow$ (K)",
]


### Irradiation Keys ###
# x-axis
PLOT_KEYS["irradiation_V_bias_up_mV"] = [
    "self.irradiation['voltage_axis']*1e3",
    r"$V_\mathrm{Bias}^\rightarrow$ (V)",
]
PLOT_KEYS["irradiation_V_bias_down_mV"] = [
    "self.irradiation['voltage_axis']*1e3",
    r"$V_\mathrm{Bias}^\leftarrow$ (V)",
]

# y-axis
PLOT_KEYS["irradiation_nu"] = [
    "self.irradiation['nu'][:, 1]*1e-9",
    r"$\nu_\mathrm{AC}$ (GHz)",
]
PLOT_KEYS["irradiation_nu_up"] = [
    "self.irradiation['nu'][:, 1]*1e-9",
    r"$\nu_\mathrm{AC}^\rightarrow$ (GHz)",
]
PLOT_KEYS["irradiation_nu_down"] = [
    "self.irradiation['nu'][:, 1]*1e-9",
    r"$\nu_\mathrm{AC}^\leftarrow$ (GHz)",
]
PLOT_KEYS["irradiation_v_ac"] = [
    "self.irradiation['v_ac'][0, :]*1e0",
    r"$V_\mathrm{AC}$ (V)",
]
PLOT_KEYS["irradiation_v_ac_up"] = [
    "self.irradiation['v_ac'][0, :]*1e0",
    r"$V_\mathrm{AC}^\rightarrow$ (V)",
]
PLOT_KEYS["irradiation_v_ac_down"] = [
    "self.irradiation['v_ac'][0, :]*1e0",
    r"$V_\mathrm{AC}^\leftarrow$ (V)",
]

# z-axis
PLOT_KEYS["irradiation_dIdV_up"] = [
    "self.irradiation['differential_conductance_up']",
    r"d$I/$d$V$ ($G_0$)",
]
PLOT_KEYS["irradiation_dIdV_down"] = [
    "self.irradiation['differential_conductance_down']",
    r"d$I/$d$V$ ($G_0$)",
]
PLOT_KEYS["irradiation_I_up_nA_abs"] = [
    "self.irradiation['current_up']*1e9",
    r"$|I|$ (nA)",
]

# n-axis
PLOT_KEYS["irradiation_T_up_K"] = [
    "self.irradiation['temperature_mean_up']*1e0",
    r"$T_\mathrm{Sample}$ (K)",
]
PLOT_KEYS["irradiation_T_down_K"] = [
    "self.irradiation['temperature_mean_down']*1e0",
    r"$T_\mathrm{Sample}$ (K)",
]


### Decrapted ###
PLOT_KEYS["I_up_A"] = ["self.mapped['current_up']*1e0", r"$I$ (A)"]
PLOT_KEYS["I_down_A"] = ["self.mapped['current_down']*1e0", r"$I$ (A)"]
PLOT_KEYS["I_up_mA"] = ["self.mapped['current_up']*1e3", r"$I$ (mA)"]
PLOT_KEYS["I_down_mA"] = ["self.mapped['current_down']*1e3", r"$I$ (mA)"]
PLOT_KEYS["I_up_muA"] = ["self.mapped['current_up']*1e6", r"$I$ (µA)"]
PLOT_KEYS["I_down_muA"] = ["self.mapped['current_down']*1e6", r"$I$ (µA)"]
PLOT_KEYS["I_up_nA"] = ["self.mapped['current_up']*1e9", r"$I$ (nA)"]
PLOT_KEYS["I_down_nA"] = ["self.mapped['current_down']*1e9", r"$I$ (nA)"]
PLOT_KEYS["I_up_pA"] = ["self.mapped['current_up']*1e12", r"$I$ (pA)"]
PLOT_KEYS["I_down_pA"] = ["self.mapped['current_down']*1e12", r"$I$ (pA)"]

PLOT_KEYS["I_up_T_A"] = ["self.mapped_over_temperature['current_up']*1e0", r"$I$ (A)"]
PLOT_KEYS["I_down_T_A"] = [
    "self.mapped_over_temperature['current_down']*1e0",
    r"$I$ (A)",
]
PLOT_KEYS["I_up_T_mA"] = ["self.mapped_over_temperature['current_up']*1e3", r"$I$ (mA)"]
PLOT_KEYS["I_down_T_mA"] = [
    "self.mapped_over_temperature['current_down']*1e3",
    r"$I$ (mA)",
]
PLOT_KEYS["I_up_T_muA"] = [
    "self.mapped_over_temperature['current_up']*1e6",
    r"$I$ (µA)",
]
PLOT_KEYS["I_down_T_muA"] = [
    "self.mapped_over_temperature['current_down']*1e6",
    r"$I$ (µA)",
]
PLOT_KEYS["I_up_T_nA"] = ["self.mapped_over_temperature['current_up']*1e9", r"$I$ (nA)"]
PLOT_KEYS["I_down_T_nA"] = [
    "self.mapped_over_temperature['current_down']*1e9",
    r"$I$ (nA)",
]
PLOT_KEYS["I_up_T_pA"] = [
    "self.mapped_over_temperature['current_up']*1e12",
    r"$I$ (pA)",
]
PLOT_KEYS["I_down_T_pA"] = [
    "self.mapped_over_temperature['current_down']*1e12",
    r"$I$ (pA)",
]
