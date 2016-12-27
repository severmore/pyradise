import numpy as np


def dbm2w(value_dbm):
    return pow(10, value_dbm / 10 - 3)

# noinspection PyUnusedLocal
def isotropic_rp(**kwargs):
    return 1.0


# noinspection PyUnusedLocal
def dipole_rp(azimuth, **kwargs):
    return abs(np.cos(np.pi / 2 * np.sin(azimuth)) / np.cos(azimuth))
