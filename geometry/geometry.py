# imports and pre-definitions

import numpy as np
from numpy.typing import NDArray

from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry


def to_polygons(geom: BaseGeometry) -> list[Polygon]:
    """
    Convert any shapely geometry (BaseGeometry) into a list of Polygon(s).
    Non-polygon geometries are ignored.
    """
    if geom.is_empty:
        return []

    gtype = geom.geom_type

    if gtype == "Polygon":
        return [geom]

    elif gtype == "MultiPolygon":
        return list(geom.geoms)

    elif gtype == "GeometryCollection":
        polys = []
        for g in geom.geoms:
            polys.extend(to_polygons(g))  # recursive extraction
        return polys

    else:
        # You could buffer tiny non-polygons into polygons if desired:
        # e.g. for LineString: geom.buffer(ε)
        return []


def annular_wedge_polygon(
    r_inner: float,
    r_outer: float,
    start_angle_deg: float,
    end_angle_deg: float,
    n_points: int = 300,
    center: tuple[float, float] = (0.0, 0.0),
) -> Polygon:
    """Create a filled annular wedge (ring sector) polygon using NumPy."""
    cx, cy = center

    # convert to radians and handle wrap-around
    a0 = np.deg2rad(start_angle_deg % 360)
    a1 = np.deg2rad(end_angle_deg % 360)
    if a1 <= a0:
        a1 += 2 * np.pi

    # angles for outer and inner arcs
    outer_angles = np.linspace(a0, a1, n_points, dtype="float64")
    inner_angles = outer_angles[::-1]  # reversed

    # vectorized outer and inner coordinates
    outer_arc = np.column_stack(
        (
            cx + r_outer * np.cos(outer_angles),
            cy + r_outer * np.sin(outer_angles),
        )
    )
    inner_arc = np.column_stack(
        (
            cx + r_inner * np.cos(inner_angles),
            cy + r_inner * np.sin(inner_angles),
        )
    )

    # concatenate arcs and close the polygon
    points = np.vstack((outer_arc, inner_arc, outer_arc[0]))

    return Polygon(points)


def regular_polygon(
    n: int = 6,
    r: float = 10.0,
    center: tuple[float, float] = (0.0, 0.0),
    theta: float = 0.0,
) -> Polygon:
    """
    Returns a Regular Polygon.

    Parameters
    ----------
    n : int = 6
        Number of order. Should be at least 3.
    r : float = 10.0
        Base radius of the Regular Polygon
    center : tuple[float, float] = (0.0, 0.0)
        Point, where the polygon is centered.
    theta : float = 0.0
        Rotation of Polyon (deg).

    Returns
    ----------
    poly : Polygon
    """
    if n < 3:
        raise KeyError("n should be 3 or higher!")
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, n + 1)  # n sides + closing point
    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)
    poly = Polygon(np.column_stack((x, y)))
    return rotate(poly, theta, origin=center)


def circle(
    r: float = 10.0,
    center: tuple[float, float] = (0.0, 0.0),
    n_points: int = 360,
) -> Polygon:
    """
    Return vertices (x, y) of a circle.

    Parameters
    ----------
    r : float = 10.0
        Base radius where the polygon is centered
    center : tuple[float, float] = (0.0, 0.0)
        Amplitude of the Gaussian lobes
    n_points : int = 360
        Number of points on circumference. Should be at least 3.

    Returns
    ----------
    poly : Polygon
    """
    return regular_polygon(n=n_points, r=r, center=center)


def rectangle(
    a: float = 10.0,
    b: float = 10.0,
    center: tuple[float, float] = (0.0, 0.0),
    theta: float = 0.0,
) -> Polygon:
    cx, cy = center
    x = np.array([-a / 2, a / 2, a / 2, -a / 2], dtype="float64") + cx
    y = np.array([-b / 2, -b / 2, b / 2, b / 2], dtype="float64") + cy
    poly = Polygon(np.column_stack((x, y)))
    return rotate(poly, theta, origin=center)


def square(
    a: float = 10.0,
    center: tuple[float, float] = (0.0, 0.0),
    theta: float = 0.0,
) -> Polygon:
    return rectangle(a=a, b=a, center=center, theta=theta)


def oblique_lattice(
    a: float = 1.0,
    b: float = 1.0,
    gamma: float = 45.0,
    R: float | tuple[float, float] = 10.0,
) -> NDArray[np.float64]:

    if isinstance(R, tuple):
        R_x, R_y = R
    else:
        R_x = R_y = R

    gamma = np.deg2rad(gamma) % (2 * np.pi)
    u_x = np.abs(a)
    v_x = (np.abs(b) * np.cos(gamma)) % u_x
    v_y = np.abs(b) * np.sin(gamma)

    x_i = np.arange(-R_x / u_x - 1, R_x / u_x + 2, dtype="float64")
    y_i = np.arange(-R_y / v_y - 1, R_y / v_y + 2, dtype="float64")

    grid_x, grid_y = np.meshgrid(x_i * u_x, y_i * v_y)
    x_shift = np.meshgrid(x_i, (y_i * v_x) % u_x)[1]
    x = np.round(np.ravel(grid_x - x_shift), 9)
    y = np.round(np.ravel(grid_y), 9)

    i_0 = np.argmin(x**2 + y**2)
    x, y = x - x[i_0], y - y[i_0]

    mask_x = np.logical_and(x <= R_x, x >= -R_x)
    mask_y = np.logical_and(y <= R_y, y >= -R_y)
    mask = np.logical_and(mask_x, mask_y)

    return np.column_stack((x[mask], y[mask]))


def regular_lattice(
    crystal: str = "oblique",
    a: float = 1.0,
    b: float = 2.0,
    gamma: float = 20,
    R: float | tuple[float, float] = 10,
) -> NDArray[np.float64]:
    match crystal:
        case "oblique":
            lattice = oblique_lattice(a=a, b=b, gamma=gamma, R=R)
        case "rectangular":
            lattice = oblique_lattice(a=a, b=b, gamma=90, R=R)
        case "square":
            lattice = oblique_lattice(a=a, b=a, gamma=90, R=R)
        case "rhombic":
            gamma_rad = np.deg2rad(gamma) % (2 * np.pi)
            lattice = oblique_lattice(
                a=a, b=a / np.sqrt(2 + 2 * np.cos(2 * gamma_rad)), gamma=gamma, R=R
            )
        case "hexagonal":
            lattice = oblique_lattice(
                a=a * np.sqrt(3),
                b=a * np.sqrt(3),
                gamma=60,
                R=R,
            )
        case _:
            raise KeyError(
                "crystal must be: oblique, rectangular, square, rhombic or hexagonal."
            )
    return lattice


def platonic_tiles(
    n: int = 3,
    a: float = 1.0,
    w: float = 0.2,
    R: float | tuple[float, float] = 10,
    center: tuple[float, float] = (0.0, 0.0),
    theta: float = 0.0,
    inverse: bool = False,
) -> MultiPolygon:

    if isinstance(R, tuple):
        R_x = R[0]
        R_y = R[1]
    else:
        R_x = R_y = R
    R_eff = (R_x + 2 * a, R_y + 2 * a)

    cx, cy = center

    polys: list[Polygon] = []
    if n == 4:
        lattice = regular_lattice(crystal="square", a=a, R=R_eff)
        for x, y in lattice:
            poly = regular_polygon(n=4, r=(a - w) / np.sqrt(2), center=(x, y), theta=45)
            polys.append(poly)
    elif n == 6:
        lattice = regular_lattice(crystal="hexagonal", a=a, R=R_eff)
        for x, y in lattice:
            poly = regular_polygon(n=6, r=(a - w), center=(x, y), theta=30)
            polys.append(poly)
    elif n == 3:
        lattice = regular_lattice(crystal="hexagonal", a=a, R=R_eff)
        for x, y in lattice:
            poly1 = regular_polygon(n=3, r=a - w, theta=30, center=(x, y))
            poly2 = regular_polygon(
                n=3, r=a - w, theta=210, center=(x - a * np.sqrt(3) / 2, y - a / 2)
            )
            polys.append(poly1)
            polys.append(poly2)
    else:
        raise KeyError("n must be 3,4 or 6.")

    tiles = MultiPolygon(polys)

    if isinstance(R, tuple):
        poly = rectangle(a=R_x, b=R_y)
    else:
        poly = circle(r=R_x)

    if inverse:
        tiles = poly.difference(tiles)
    else:
        tiles = poly.intersection(tiles)

    tiles = rotate(tiles, angle=theta)
    tiles = translate(tiles, xoff=cx, yoff=cy)

    return MultiPolygon(to_polygons(tiles))


def radial_lobe(
    n: int = 6,
    r_base: float = 5,
    amplitude: float = 5,
    phase: float = 0.0,
    angle: float = 0,
    n_points: int = 1801,
) -> Polygon:
    """
    Generate a rotationally symmetric lobed shape defined by r(θ) = r_base + amplitude * cos(n * θ + phase)
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    r = r_base + amplitude * np.cos(n * theta + phase)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    poly = Polygon(np.column_stack([x, y])).buffer(0)
    return rotate(poly, angle, origin=(0, 0))


def gaussian_ring(
    n: int = 6,
    r0: float = 6,
    amplitude: float = 3,
    width: float = 3,
    sigma: float = 1,
    phase: float = 0.0,
    angle: float = 0,
    n_points: int = 1801,
):
    """
    Create a rotationally symmetric shape with Gaussian radial lobes.

    Parameters
    ----------
    n : int
        Number of symmetric repetitions (e.g. 6 for hexagonal symmetry)
    r0 : float
        Base radius where the ring is centered
    amplitude : float
        Amplitude of the Gaussian lobes
    width : float
        Radial thickness of the overall shape
    sigma : float
        Angular width of each Gaussian lobe (in radians)
    phase : float
        Phase rotation offset in radians
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    mod_theta = (
        np.mod(theta * n + phase + np.pi, 2 * np.pi) - np.pi
    )  # periodic angle per lobe

    r = r0 + amplitude * np.exp(-0.5 * (mod_theta / sigma) ** 2)
    r_inner = np.clip(r - width, 0, None)

    outer = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    inner = np.column_stack(
        [r_inner[::-1] * np.cos(theta[::-1]), r_inner[::-1] * np.sin(theta[::-1])]
    )

    poly = Polygon(np.vstack([outer, inner])).buffer(0)
    return rotate(poly, angle, origin=(0, 0))
