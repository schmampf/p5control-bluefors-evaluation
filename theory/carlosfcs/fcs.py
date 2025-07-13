import os
import io
import hashlib
import pickle
import subprocess
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

WORK_DIR = "/Users/oliver/Documents/p5control-bluefors-evaluation/theory/carlosfcs/"
CACHE_DIR = os.path.join(WORK_DIR, "./.cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def hash_params(*params):
    m = hashlib.sha256()
    for p in params:
        if isinstance(p, np.ndarray):
            m.update(p.tobytes())
        else:
            m.update(str(p).encode())
    return m.hexdigest()


def bin_y_over_x(
    x: np.ndarray,
    y: np.ndarray,
    x_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    # Extend bin edges for histogram: shift by half a bin width for center alignment
    x_nu = np.append(x_bins, 2 * x_bins[-1] - x_bins[-2])  # Add one final edge
    x_nu = x_nu - (x_nu[1] - x_nu[0]) / 2

    # Count how many x-values fall into each bin
    _count, _ = np.histogram(x, bins=x_nu, weights=None)
    _count = np.array(_count, dtype="float64")
    _count[_count == 0] = np.nan

    # Sum of y-values in each bin
    _sum, _ = np.histogram(x, bins=x_nu, weights=y)

    # Return mean y per bin and count
    return _sum / _count, _count


def run_fcs(
    voltage_V: float,
    temperature_K: float = 0.0,
    energy_gap_V: float = 2e-4,
    dynes_parameter_V: float = 0.0,
    transmission: float = 0.5,
) -> np.ndarray:
    """
    Run the Fortran I-V solver for a given set of physical parameters.

    Parameters
    ----------
    voltage_V : float
        voltage (in V)
    temperature_K : float, optional
        Temperature in Kelvin.
    energy_gap_V : float, optional
        Superconducting gap in Volts.
    dynes_parameter_V : float, optional
        Dynes parameter in Volts.
    transmission : float, optional
        Transmission coefficient [0, 1].

    Returns
    -------
    data : np.ndarray
        data (as returned by solver).
    """

    nmax: int = 10
    iw: int = 2003
    nchi: int = 66

    workdir = "/Users/oliver/Documents/p5control-bluefors-evaluation/theory/carlosfcs/"
    fcs_exe = os.path.join(workdir, "fcs")
    fcs_in = os.path.join(workdir, f"fcs.in")
    tmp_input_dir = os.path.join(workdir, ".tmp")
    os.makedirs(tmp_input_dir, exist_ok=True)

    tmp_in = os.path.join(
        tmp_input_dir,
        f"{voltage_V:.3e}.in",
    )

    if dynes_parameter_V <= 0:
        dynes_parameter_V = 1e-7

    with open(fcs_in, "r") as f:
        lines = f.readlines()

    lines[0] = f"{transmission:.5f} (transmission)\n"
    lines[1] = f"{temperature_K:.5f} (temp in K)\n"
    lines[2] = f"{energy_gap_V*1e3:.6f} {energy_gap_V*1e3:.6f} (gap1,gap2 in meV)\n"
    lines[3] = (
        f"{dynes_parameter_V*1e3:.6f} {dynes_parameter_V*1e3:.6f} (eta1,eta2 = broadening in meV)\n"
    )
    lines[4] = f"{voltage_V*1e3:.8f} {voltage_V*1e3:.8f} 1.0 (vi,vf,vstep in mV)\n"
    lines[5] = f"{nmax} {iw} {nchi} (nmax,iw,nchi)\n"

    with open(tmp_in, "w") as f:
        f.writelines(lines)

    try:
        proc = subprocess.run(
            [fcs_exe],
            stdin=open(tmp_in, "r"),
            capture_output=True,
            text=True,
            cwd=workdir,
        )
    finally:
        if os.path.isfile(tmp_in):
            os.remove(tmp_in)

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)

    data = np.genfromtxt(io.StringIO(proc.stdout), dtype="float64")
    if data.size == 0:
        raise RuntimeError(
            "Fortran code produced no output. Check input sweep range and step."
        )

    # voltage = data[:, 0] * 1e-3  # Convert voltage to V
    # currents = data[:, 1:] * 1e-9  # Convert currents to A

    return data


def get_current_fcs(
    voltage_V: np.ndarray,
    energy_gap_V: float = 2e-4,
    transmission: float = 0.5,
    temperature_K: float = 0.0,
    dynes_parameter_V: float = 0.0,
    n_worker: int = 16,
    chunks=4,
) -> np.ndarray:
    """
    Get the current for a given set of physical parameters using the FCS solver.

    Parameters
    ----------
    voltage_V : np.ndarray
        Array of voltages (in V) to sweep.
    temperature_K : float, optional
        Temperature in Kelvin.
    energy_gap_V : float, optional
        Superconducting gap in Volts.
    dynes_parameter_V : float, optional
        Dynes parameter in Volts.
    transmission : float, optional
        Transmission coefficient [0, 1].

    Returns
    -------
    currents : np.ndarray
        Currents in A (as returned by solver).
    """

    m_max: int = 10

    key = hash_params(
        voltage_V, transmission, energy_gap_V, temperature_K, dynes_parameter_V
    )
    cached_dir = os.path.join(CACHE_DIR, key)
    cached_dir_npz = os.path.join(CACHE_DIR, f"{key}.npz")

    if os.path.exists(cached_dir_npz):
        data = np.load(cached_dir_npz)
        return data["I"]
    else:
        stepsize_V = np.abs(np.nanmax(voltage_V) - np.nanmin(voltage_V)) / (
            len(voltage_V) - 1
        )
        final_value_V = np.nanmax(np.abs(voltage_V))
        input_voltage_V = np.arange(0, final_value_V + stepsize_V, stepsize_V)

        with ThreadPoolExecutor(max_workers=n_worker) as executor:
            futures = []
            for i, v in enumerate(input_voltage_V):
                futures.append(
                    executor.submit(
                        run_fcs,
                        voltage_V=v,
                        temperature_K=temperature_K,
                        energy_gap_V=energy_gap_V,
                        dynes_parameter_V=dynes_parameter_V,
                        transmission=transmission,
                    )
                )

            temp_voltage = []
            temp_currents = []

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="IV simulations",
                unit="sim",
            ):
                result = future.result()
                v = np.array(result[0]) * 1e-3  # Convert voltage to V
                print(v)
                i = np.array(result[1:]) * 1e-9  # Convert currents to A

                temp_voltage.append(v)
                temp_voltage.append(-v)
                temp_currents.append(i)
                temp_currents.append(-i)

            temp_voltage = np.array(temp_voltage)
            temp_currents = np.array(temp_currents)
            print(temp_voltage, temp_voltage.shape, temp_currents, temp_currents.shape)

            # remove NaN values from temp_voltages and temp_currents
            valid_indices = ~np.isnan(temp_voltage)
            temp_voltages = temp_voltage[valid_indices]
            temp_currents = temp_currents[valid_indices, :]

            # Initialize the output array for currents
            currents = np.full((voltage_V.shape[0], m_max + 1), np.nan)

            for i in range(temp_currents.shape[1]):
                currents[:, i] = bin_y_over_x(
                    x=temp_voltages,
                    y=temp_currents[:, i],
                    x_bins=voltage_V,
                )[0]

            np.savez(
                cached_dir,
                I=currents,
                V=voltage_V,
                Delta=energy_gap_V,
                tau=transmission,
                T=temperature_K,
                Gamma=dynes_parameter_V,
            )

            return currents
