import numpy as np

from pysteps import io, motion, rcparams
from pysteps.utils import conversion, dimension
from pysteps.visualization import plot_precip_field
from scipy.ndimage import map_coordinates

import numba as nb

"""
Original code (slightly modified) from pysteps documentation:
https://pysteps.readthedocs.io/en/latest/auto_examples/advection_correction.html#sphx-glr-auto-examples-advection-correction-py

"""


def advection_correction(R, T=5, t=1):
    """
    R = np.array([qpe_previous, qpe_current])
    T = time between two observations (5 min)
    t = interpolation timestep (1 min)
    """

    # Evaluate advection
    oflow_method = motion.get_method("LK")
    fd_kwargs = {"buffer_mask": 10}  # avoid edge effects
    V = oflow_method(np.log(R), fd_kwargs=fd_kwargs)

    # Perform temporal interpolation
    Rd = np.zeros((R[0].shape))
    x, y = np.meshgrid(np.arange(R[0].shape[1], dtype=float), np.arange(R[0].shape[0], dtype=float))

    interpolated_frames = []

    for i in range(t, T + t, t):
        pos1 = (y - i / T * V[1], x - i / T * V[0])
        # order=3: cubic interpolation, order=1: linear interpolation
        R1 = map_coordinates(R[0], pos1, order=1)

        pos2 = (y + (T - i) / T * V[1], x + (T - i) / T * V[0])
        R2 = map_coordinates(R[1], pos2, order=1)

        interp = ((T - i) * R1 + i * R2) / T
        interpolated_frames.append(interp)

    return interpolated_frames


def advection_correction_with_motion(R, motion, T=5, t=1):
    """
    Perform advection correction with motion.

    Parameters:
    R (np.array): Array containing two frames of data [qpe_previous, qpe_current].
    motion (tuple): Tuple containing the motion vector (motion[0] for x-axis, motion[1] for y-axis).
    T (int): Time between two observations in minutes (default is 5 minutes).
    t (int): Interpolation timestep in minutes (default is 1 minute).

    Returns:
    list: List of interpolated frames.

    """
    # Perform temporal interpolation
    x, y = np.meshgrid(np.arange(R[0].shape[1], dtype=float), np.arange(R[0].shape[0], dtype=float))

    interpolated_frames = []

    for i in range(t, T + t, t):
        pos1 = (y - i / T * motion[1], x - i / T * motion[0])
        # order=3: cubic interpolation, order=1: linear interpolation
        R1 = map_coordinates(R[0], pos1, order=1)

        pos2 = (y + (T - i) / T * motion[1], x + (T - i) / T * motion[0])
        R2 = map_coordinates(R[1], pos2, order=1)

        interp = ((T - i) * R1 + i * R2) / T**2
        interpolated_frames.append(interp)

    return interpolated_frames


def interpolate_ensemble(arrs1, arrs2, motion, T=5, t=1):
    """
    Interpolates between two arrays based on a given motion vector.

    Parameters:
    - arrs1 (ndarray): First array to interpolate.
    - arrs2 (ndarray): Second array to interpolate.
    - motion (tuple): Motion vector (x, y) specifying the direction and magnitude of motion.
    - T (int): Total number of interpolation steps.
    - t (int): Time step between each interpolation.

    Returns:
    - R0 (ndarray): Interpolated array.

    """
    x, y = np.meshgrid(np.arange(arrs1[0].shape[1], dtype=float), np.arange(arrs1[0].shape[0], dtype=float))

    R0 = np.zeros_like(arrs1)

    factor = 1 / T**2

    for i in range(t, T + 1, t):
        pos1 = np.array((y - i / T * motion[1], x - i / T * motion[0]))
        pos1 = pos1.reshape(pos1.shape[0], -1)
        # NOTE: Expensive parts in this function is the map_coords call,
        # each call about 25% of time spent in this function
        R1 = map_coords(arrs1, pos1)
        R1 = R1.reshape(arrs1.shape)

        pos2 = np.array((y + (T - i) / T * motion[1], x + (T - i) / T * motion[0]))
        pos2 = pos2.reshape(pos2.shape[0], -1)
        R2 = map_coords(arrs2, pos2)
        R2 = R2.reshape(arrs2.shape)

        # NOTE: This is also a time-consuming line, ~45% of time spent in this function
        R0 += ((T - i) * R1 + i * R2) * factor

    return R0


@nb.njit(fastmath=True, parallel=False)
def map_coords(ars, coords):
    """
    Map input arrays to new coordinates.

    Args:
        ars (ndarray): Input arrays with shape (m, nx, ny).
        coords (ndarray): Coordinate array with shape (2, n).

    Returns:
        ndarray: Mapped values with shape (n, m).

    Modified to 2D from https://stackoverflow.com/a/62692531

    """
    # these have shape (n, 2)
    ij = coords.T.astype(np.int16)
    fij = (coords.T - ij).astype(np.float32)
    n = ij.shape[0]
    m = ars.shape[0]
    out = np.empty((n, m), dtype=np.float64)

    for l in nb.prange(n):
        i0 = ij[l, 0]
        j0 = ij[l, 1]
        # k0 = ij[l, 2]
        # Note: don't write i1, j1, k1 = ijk[l, :3]+1 -- much slower.
        i1, j1 = i0 + 1, j0 + 1
        fi1 = fij[l, 0]
        fj1 = fij[l, 1]
        # fk1 = fij[l, 2]

        fi0, fj0 = 1 - fi1, 1 - fj1
        for i in range(ars.shape[0]):
            out[l, i] = (
                fi0 * fj0 * ars[i, i0, j0]
                + fi0 * fj0 * ars[i, i0, j0]
                + fi0 * fj1 * ars[i, i0, j1]
                + fi0 * fj1 * ars[i, i0, j1]
                + fi1 * fj0 * ars[i, i1, j0]
                + fi1 * fj0 * ars[i, i1, j0]
                + fi1 * fj1 * ars[i, i1, j1]
                + fi1 * fj1 * ars[i, i1, j1]
            )
    return out.T
