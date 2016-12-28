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
        return np.abs(np.sin(np.pi / 2 * n * s) / np.sin(np.pi / 2 * s)) / n
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
        return np.abs(c * np.sin(np.pi / 2 * n * c) / np.sin(np.pi / 2 * c))
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
            (np.cos(tilt) ** 2 + np.cos(azimuth) ** 2 * np.sin(tilt) ** 2) ** 0.5)


def _reflection_c_parallel(grazing_angle, permittivity, conductivity, wavelen):
    eta = permittivity - 60j * wavelen * conductivity
    c = np.cos(grazing_angle)
    return (eta - c ** 2) ** 0.5


def _reflection_c_perpendicular(grazing_angle, permittivity, conductivity, wavelen):
    eta = permittivity - 60j * wavelen * conductivity
    c = np.cos(grazing_angle)
    return (eta - c ** 2) ** 0.5 / eta


# noinspection PyUnusedLocal
def reflection_constant(*, grazing_angle, polarization, permittivity, conductivity, wavelen, **kwargs):
    return -1.0 + 0.j


# noinspection PyUnusedLocal
def reflection(*, grazing_angle, polarization, permittivity, conductivity, wavelen, **kwargs):
    """
    Computes reflection coefficient from conducting surface with defined grazing angle and supported relative
    permittivity and conductivity of the surface. In order to set type of wave polarization, polarization
    parameter is specified. For parallel to surface polarized wave polarization should set to 1, for perpendicular -
    to 0; for circular - to 0.5. For different type of elliptic polarization use other value in the range of 0..1

    :param grazing_angle: an angle between normal to surface and wave vector
    :param polarization: determine the type of polarization of the grazing wave
    :param permittivity: the relative_permittivity of two media divided by the surface
    :param conductivity: the conductivity of the surface
    :param wavelen: the wave length of the grazing wave
    :return: the reflection coefficient for specified parameters and the given grazing angle
    """
    s = np.sin(grazing_angle)

    if polarization < 0 or polarization > 1:
        return float('nan')

    if polarization != 0:
        c_parallel = _reflection_c_parallel(grazing_angle, permittivity, conductivity, wavelen)
        reflection_parallel = (s - c_parallel) / (s + c_parallel)
    else:
        reflection_parallel = 0.j

    if polarization != 1:
        c_perpendicular = _reflection_c_perpendicular(grazing_angle, permittivity, conductivity, wavelen)
        reflection_perpendicular = (s - c_perpendicular) / (s + c_perpendicular)
    else:
        reflection_perpendicular = 0.j

    return polarization * reflection_parallel + (1 - polarization) * reflection_perpendicular


def free_space_path_loss_2d(*, distance, tx_rp, rx_rp, tx_height, rx_height, wavelen, **kwargs):
    """
    Computes free space signal attenuation between the transmitter and the receiver in linear scale.

    :param distance: the distance between transmitter and receiver
    :param tx_rp: a radiation pattern of the transmitter
    :param rx_rp: a radiation pattern of the receiver
    :param tx_height: a mount height of the transmitter
    :param rx_height: a mount height of the receiver
    :param wavelen: a wavelen of signal carrier
    :return: free space path loss in linear scale
    """
    # Ray geometry computation
    delta_height = np.abs(tx_height - rx_height)
    d0 = (delta_height**2 + distance**2)**0.5
    alpha0 = np.arctan(distance / delta_height)

    # Attenuation caused by radiation pattern
    g0 = (tx_rp(azimuth=alpha0, tilt=0, wavelen=wavelen, **kwargs) *
          rx_rp(azimuth=alpha0, tilt=0, wavelen=wavelen, **kwargs))

    k = wavelen / (4*np.pi)
    return (k*g0/d0)**2


def two_ray_path_loss(*, distance, time, speed, ground_reflection, tx_rp, rx_rp, tx_height, rx_height, wavelen,
                      **kwargs):
    """
    Computes free space signal attenuation between the transmitter and the receiver in linear scale.
    :param distance: the distance between transmitter and receiver
    :param time: current time
    :param speed: relative speed of the receiver
    :param ground_reflection: a function to compute a complex-valued reflection coefficient
    :param tx_rp: a radiation pattern of the transmitter
    :param rx_rp: a radiation pattern of the receiver
    :param tx_height: a mount height of the transmitter
    :param rx_height: a mount height of the receiver
    :param wavelen: a wavelen of signal carrier
    :return: free space path loss in linear scale
    """
    # Ray geometry computation
    delta_height = np.abs(tx_height - rx_height)
    sigma_height = np.abs(tx_height + rx_height)
    d0 = (delta_height ** 2 + distance ** 2) ** 0.5
    d1 = (sigma_height ** 2 + distance ** 2) ** 0.5
    alpha0 = np.arctan(distance / delta_height)
    alpha1 = np.arctan(distance / sigma_height)

    # Attenuation caused by radiation pattern
    g0 = (tx_rp(azimuth=alpha0, tilt=0, wavelen=wavelen, **kwargs) *
          rx_rp(azimuth=alpha0, tilt=0, wavelen=wavelen, **kwargs))

    g1 = (tx_rp(azimuth=alpha1, tilt=0, wavelen=wavelen, **kwargs) *
          rx_rp(azimuth=alpha1, tilt=0, wavelen=wavelen, **kwargs))

    # Attenuation due to reflections (reflection coefficient) computation
    r1 = ground_reflection(grazing_angle=alpha1, wavelen=wavelen, **kwargs)

    k = 2*np.pi / wavelen
    return (0.5/k)**2 * np.absolute( g0/d0*np.exp(-1j*k*(d0 + speed*time*np.sin(alpha0))) +
                                  g1*r1/d1*np.exp(-1j*k*(d1 - speed*time*np.sin(alpha1))) )**2


def signal2noise(*, rx_power, noise_power, **kwargs):
    """
    Computes Signal-to-Noise ratio. Input parameters are in logarithmic scale.
    :param rx_power:
    :param noise_power:
    :param kwargs:
    :return:
    """
    return db2lin(rx_power - noise_power)


def sync_angle(*, snr, preamble_duration=9.3e-6, bandwidth=1.2e6, **kwargs):
    return (snr*preamble_duration*bandwidth)**-0.5


