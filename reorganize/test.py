import h5py
import numpy as np
import matplotlib.pyplot as plt

path = r"/home/dacap/Downloads/2023-11-04_G0_antenna.hdf5"

with h5py.File(path, "r") as f:
    data = f[
        "measurement/var=(vna_amplitudes,(-3.1e+01,0e+00),Bm) const=[(vna_frequency,1.5e+10,Hz)]/nu=-30.0dBm/sweep/thermo"
    ]

    print(np.nanmean(data["T"]))
