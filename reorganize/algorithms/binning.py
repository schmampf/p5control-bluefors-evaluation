# region imports
# std


# third-party
import numpy as np

# local
import utilities.logging as Logger

# endregion


def bin(
    x: np.ndarray, y: np.ndarray, x_bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Extend bin edges for histogram: shift by half a bin width for center alignment
    x_nu = np.append(x_bins, 2 * x_bins[-1] - x_bins[-2])  # Add one final edge
    x_nu = x_nu - (x_nu[1] - x_nu[0]) / 2

    # Count how many x-values fall into each bin
    count, _ = np.histogram(x, bins=x_nu, weights=None)
    count = np.array(count, dtype="float64")
    count[count == 0] = np.nan

    # Sum of y-values in each bin
    sum, _ = np.histogram(x, bins=x_nu, weights=y)

    # Return mean y per bin and count
    return sum / count, count


def re_bin(
    y: np.ndarray, z: np.ndarray, y_binned: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    number_of_bins = len(y_binned)
    counter = np.zeros(number_of_bins, dtype=int)
    result = np.zeros((number_of_bins, z.shape[1]), dtype=float)

    # Assign y-values to bins
    dig = np.digitize(y, bins=y_binned) - 1
    valid_indices = (dig >= 0) & (dig < number_of_bins)

    # Count entries and sum z-values into bins
    np.add.at(counter, dig[valid_indices], 1)
    np.add.at(result, dig[valid_indices], z[valid_indices])

    # Average values where data exists
    with np.errstate(invalid="ignore"):  # Suppress divide-by-zero warnings
        result[counter > 0] /= counter[counter > 0, None]
        result[counter == 0] = np.nan  # Mark empty bins as NaN

    # Top-down fill: fill missing rows from previous valid row above
    last_valid = None
    for i in range(number_of_bins):
        if counter[i] > 0:
            last_valid = result[i].copy()
        elif last_valid is not None:
            result[i] = last_valid

    return result, counter
