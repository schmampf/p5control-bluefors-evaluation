import jax.numpy as jnp
from jax import jit, Array


@jit
def bin_y_over_x(
    x: Array,
    y: Array,
    x_bins: Array,
) -> Array:

    # Extend bin edges for histogram: shift by half a bin width for center alignment
    x_nu = jnp.append(x_bins, 2 * x_bins[-1] - x_bins[-2])  # Add one final edge
    x_nu = x_nu - (x_nu[1] - x_nu[0]) / 2

    # Count how many x-values fall into each bin
    _count, _ = jnp.histogram(x, bins=x_nu, weights=None)
    _count = jnp.where(_count == 0, jnp.nan, _count)

    # Sum of y-values in each bin
    _sum, _ = jnp.histogram(x, bins=x_nu, weights=y)

    # Return mean y per bin and count
    return _sum / _count
