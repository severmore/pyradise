import numpy as np


def dbm2w(value_dbm):
    return 10 ** (value_dbm / 10 - 3)


def w2dbm(value_watt):
    return 10 * np.log10(value_watt) + 30


def db2lin(value_db):
    return 10 ** (value_db / 10)


def lin2db(value_linear):
    return 10 * np.log10(value_linear)


# noinspection PyUnusedLocal
def isotropic_rp(**kwargs):
    """
    Returns constant (1.0)
    :return:
    """
    return 1.0


# noinspection PyUnusedLocal
def dipole_rp(*, azimuth, **kwargs):
    """
    Returns dipole directional gain
    :param azimuth:
    :return:
    """
    c = np.cos(azimuth)
    s = np.sin(azimuth)
    if np.abs(c) > 1e-9:
        return np.abs(np.cos(np.pi / 2 * s) / c)
    else:
        return 0.0


# TODO: check right formula, guess type of antenna element if considering not dipole
# noinspection PyUnusedLocal
def array_dipole_rp(*, azimuth, n, **kwargs):
    """
    Returns dipole array directional gain
    :param azimuth:
    :param n:
    :return:
    """
    c = np.cos(azimuth)
    s = np.sin(azimuth)
    if np.abs(s) < 1e-9:
        return 1.0
    elif np.abs(c) > 1e-9:
        return np.abs(np.sin(np.pi / 2 * n * s) / np.sin(np.pi/2 * s)) / n
    else:
        return 0.0


# TODO: check right formula
# noinspection PyUnusedLocal
def helix_rp(*, azimuth, n, **kwargs):
    """
    Returns helix antenna directional gain
    :param azimuth:
    :param n:
    :return:
    """
    c = np.cos(azimuth)
    s = np.sin(azimuth)
    if np.abs(c) > 1e-9:
        return np.abs(c * np.sin(np.pi / 2 * n * c) / np.sin(np.pi/2 * c))
    else:
        return 0.0


def _patch_rp_factor(azimuth, tilt, wavelen, width, length):
    s_a = np.sin(azimuth)
    c_a = np.cos(azimuth)
    s_t = np.sin(tilt)
    c_t = np.cos(tilt)
    kw = np.pi / wavelen * width
    kl = np.pi / wavelen * length
    if np.abs(s_a) * np.abs(s_t) < 1e-9:
        return 1.0
    elif np.abs(c_a) + np.abs(c_t) > 1e-9:
        return np.sin(kw * s_a * s_t) / (kw * s_a * s_t) * np.cos(kl * s_a * c_t)
    else:
        return 0.


def _patch_theta_rp(azimuth, tilt, wavelen, width, length):
    return _patch_rp_factor(azimuth, tilt, wavelen, width, length) * np.cos(tilt)


def _patch_phi_rp(azimuth, tilt, wavelen, width, length):
    return -1 * _patch_rp_factor(azimuth, tilt, wavelen, width, length) * np.sin(tilt) * np.cos(azimuth)


# noinspection PyUnusedLocal
def patch_rp(*, azimuth, tilt, wavelen, width, length, **kwargs):
    """
    Returns directional gain (in linear scale, 0..1)
    :param azimuth:
    :param tilt:
    :param wavelen:
    :param width:
    :param length:
    :return:
    """
    return (np.abs(_patch_rp_factor(azimuth, tilt, wavelen, width, length)) *
            (np.cos(tilt)**2 + np.cos(azimuth)**2 * np.sin(tilt)**2)**0.5)
