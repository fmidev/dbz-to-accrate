import numpy as np

from pysteps import io, motion, rcparams
from pysteps.utils import conversion, dimension
from pysteps.visualization import plot_precip_field
from scipy.ndimage import map_coordinates

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
    x, y = np.meshgrid(
        np.arange(R[0].shape[1], dtype=float), np.arange(R[0].shape[0], dtype=float)
    )

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
