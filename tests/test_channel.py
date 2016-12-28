import unittest
import pyradise.channel as ch
import numpy as np


class PowerUnitsConversionTest(unittest.TestCase):

    def test_dbm2w(self):
        self.assertEqual(ch.dbm2w(30), 1)
        self.assertEqual(ch.dbm2w(0), 1e-3)

    def test_w2dbm(self):
        self.assertEqual(ch.w2dbm(1), 30)
        self.assertEqual(ch.w2dbm(1e-3), 0)

    def test_db2lin(self):
        self.assertEqual(ch.db2lin(10), 10)
        self.assertEqual(ch.db2lin(0), 1)

    def test_lin2db(self):
        self.assertEqual(ch.lin2db(1), 0)
        self.assertEqual(ch.lin2db(10), 10)


class RadiationPatternsTest(unittest.TestCase):

    def test_isotropic_rp(self):
        for i in np.linspace(0, 2*np.pi):
            self.assertEqual(ch.isotropic_rp(azimuth=i), 1.0)

    def test_dipole_rp(self):
        self.assertEqual(ch.dipole_rp(azimuth=0, n=10), 1.0)
        self.assertEqual(ch.dipole_rp(azimuth=0), 1.0)
        self.assertEqual(ch.dipole_rp(azimuth=np.pi/6), (2/3)**0.5)
        self.assertEqual(ch.dipole_rp(azimuth=np.pi/2), 0)

    # TODO: complete test
    def test_array_dipole_rp(self):
        self.assertEqual(ch.array_dipole_rp(azimuth=0, n=1), 1.0)
        self.assertEqual(ch.array_dipole_rp(azimuth=np.pi/2, n=1), 0)

    # TODO: complete test
    def test_helix_rp(self):
        self.assertEqual(ch.helix_rp(azimuth=0, n=1), 1.0)
        self.assertEqual(ch.helix_rp(azimuth=np.pi/2, n=1), 0)

    def test_patch_rp(self):
        wavelen = 0.34
        width = wavelen / 2
        length = wavelen / 2
        self.assertEqual(ch.patch_rp(azimuth=0, tilt=0, wavelen=wavelen, width=width, length=length), 1.0)
        self.assertEqual(ch.patch_rp(azimuth=np.pi/2, tilt=0, wavelen=wavelen, width=width, length=length), 1)
        self.assertEqual(ch.patch_rp(azimuth=0, tilt=np.pi/2, wavelen=wavelen, width=width, length=length), 1)
        # self.assertEqual(ch.patch_rp(np.pi/2, np.pi/2, wavelen, width, length), 0)


class ReflectionTest(unittest.TestCase):

    def test_reflection(self):
        self.assertEqual(ch.reflection(grazing_angle=0, polarization=0, permittivity=15,
                                       conductivity=0.03, wavelen=0.34), -1 + 0j)
        self.assertEqual(ch.reflection(grazing_angle=0, polarization=1, permittivity=15,
                                       conductivity=0.03, wavelen=0.34), -1 + 0j)
        self.assertEqual(ch.reflection(grazing_angle=0, polarization=0.5, permittivity=15,
                                       conductivity=0.03, wavelen=0.34), -1 + 0j)

if __name__ == '__main__':
    unittest.main()
