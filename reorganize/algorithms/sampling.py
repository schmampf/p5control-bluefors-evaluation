# region imports
# std


# third-party
import numpy as np
import torch

# local
import utilities.logging as Logger
import algorithms.binning as Binning


# endregion


def downsample(x: np.ndarray, y: np.ndarray, target_freq: float):
    data_freq = 1 / np.median(np.diff(x))

    assert target_freq > 0, "Target frequency must be greater than 0"
    assert target_freq < data_freq, "Target frequency must be less than data frequency"

    if data_freq > 3 * target_freq:
        Logger.print(
            Logger.WARNING,
            msg=f"Warning: Target frequency {target_freq:.2f} Hz is less than 1/3 of data frequency {data_freq:.2f} Hz",
        )

    bins = np.arange(np.min(x), np.max(x), 1 / target_freq)

    binned, counts = Binning.bin(x, y, bins)

    return binned, bins, counts, target_freq


def upsample(x: np.ndarray, y: np.ndarray, method: str, factor: float):
    k = np.full((2, len(x)), np.nan)
    k[0, :] = x
    k[1, :] = y

    # Apply PyTorch interpolation
    m = torch.nn.Upsample(mode=method, scale_factor=factor)
    big = m(torch.from_numpy(np.array([k])))
    x = np.array(big[0, 0, :])
    y = np.array(big[0, 1, :])
