import unittest
import numpy as np
from brian2 import have_same_dimensions, farad, meter, ohm, second, msecond, uF, cm, um, siemens, ms
from brian2.units.allunits import pampere

from iteration_19_tapered_dendrites.data import CableParameters

def cable_params():

    return CableParameters.from_SI(
        c_m=0.01,          # F/m²
        Rm=2000,           # ohm*m²
        ra=1,              # ohm*m
        L=500e-6,          # m
        N=200,
        r0=2e-6,
        I_e=1.5e-12
    )

def assert_close(a, b):
    assert np.isclose(
        float(a / b),
        1.0
    ), f"{a} != {b}"

def assert_quantity_equal(a, b):
    assert have_same_dimensions(a, b)
    assert np.isclose(float(a / b), 1.0)

class MyTestCase(unittest.TestCase):

    @staticmethod
    def test_override_equals_rebuild():
        params = CableParameters.from_SI(
            c_m=0.01,
            Rm=2000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )
        p1 = params.with_SI_properties(Rm=4000)

        p2 = CableParameters.from_SI(
            c_m=0.01,
            Rm=4000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )

        assert p1.tau == p2.tau
        assert p1.b == p2.b

    def test_from_SI_units(self):
        p = CableParameters.from_SI(
            c_m=0.01,
            Rm=2000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )

        assert have_same_dimensions(
            p.c_m,
            farad / meter ** 2
        )

        assert have_same_dimensions(
            p.Rm,
            ohm * meter ** 2
        )

        assert have_same_dimensions(
            p.tau,
            second
        )

    @staticmethod
    def test_gL():
        p = CableParameters.from_SI(
            c_m=0.01,
            Rm=2000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )

        assert_close(
            p.gL,
            1 / p.Rm
        )

    @staticmethod
    def test_tau():
        p = CableParameters.from_SI(
            c_m=0.01,
            Rm=2000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )

        assert_close(
            p.tau,
            p.Rm * p.c_m
        )

    @staticmethod
    def test_dx():
        p = CableParameters.from_SI(
            c_m=0.01,
            Rm=2000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )

        assert_close(
            p.dx,
            p.L / (p.N - 1)
        )

    @staticmethod
    def test_b():
        p = CableParameters.from_SI(
            c_m=0.01,
            Rm=2000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )

        expected = (
                p.r0 /
                (2 * p.c_m * p.ra)
        )

        assert_close(
            p.b,
            expected
        )

    @staticmethod
    def test_override():
        p = CableParameters.from_SI(
            c_m=0.01,
            Rm=2000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )

        q = p.with_SI_properties(
            Rm=4000
        )

        assert_close(
            q.Rm,
            4000 * ohm * meter ** 2
        )

        assert_close(
            q.tau,
            q.Rm * q.c_m
        )

    @staticmethod
    def test_override_does_not_modify_original():
        p = CableParameters.from_SI(
            c_m=0.01,
            Rm=2000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )

        old = p.Rm

        q = p.with_SI_properties(
            Rm=5000
        )

        assert_close(
            p.Rm,
            old
        )

        assert_close(
            q.Rm,
            5000 * ohm * meter ** 2
        )

    def test_from_units_to_isi(self):
        p = CableParameters(c_m=1 * uF / cm **2, Rm = 2 * 1E4 * ohm * cm**2)

        self.assertAlmostEqual(20, p.tau / msecond)
        self.assertAlmostEqual(0.02, p.tau / second)
        numerical = p.to_numerical()
        self.assertAlmostEqual(0.02, numerical.tau)

    def test_numbers_used_in_simulation(self):
        Rm = 2 * 1E4 * ohm * cm ** 2

        p = CableParameters(c_m=1 * uF / cm ** 2,
        Rm = Rm,
        gL = 1 / Rm,
        ra = 100 * ohm * cm,
        L = 500.0 * um,
        N = 101,
        r0 = 2 * um,
        I_e = 150 * pampere)

        p_si = p.to_numerical()

        self.assertAlmostEqual(0.01, p_si.c_m)  # F/m²
        self.assertAlmostEqual(2.0, p_si.Rm)  # Ω·m²
        self.assertAlmostEqual(1.0, p_si.ra)  # Ω·m
        self.assertAlmostEqual(500e-6, p_si.L)  # m
        self.assertAlmostEqual( 5e-6, p_si.dx)  # m
        self.assertAlmostEqual( 2e-6, p_si.r0)  # m
        self.assertAlmostEqual( 1.5e-12, p_si.I_e) # Ampere

        p_from_numerical = CableParameters.from_numerical(p_si)

        self.assertAlmostEqual(1, p_from_numerical.c_m / (uF / cm ** 2))
        self.assertAlmostEqual(2e4, p_from_numerical.Rm / (ohm * cm ** 2))
        self.assertAlmostEqual(5e-5, p_from_numerical.gL / (siemens / cm ** 2))
        self.assertAlmostEqual(100.0, p_from_numerical.ra / (ohm * cm))
        self.assertAlmostEqual(500.0, p_from_numerical.L / um)
        self.assertAlmostEqual(5.0, p_from_numerical.dx / um)
        self.assertAlmostEqual(20.0, p_from_numerical.tau / ms)
        self.assertAlmostEqual(2.0, p_from_numerical.r0 / um)
        self.assertAlmostEqual(1.0, p_from_numerical.b / (cm ** 2 / second))
        self.assertAlmostEqual(1E5, p_from_numerical.b / (um ** 2 / ms))
        self.assertAlmostEqual(150.0, p_from_numerical.I_e / pampere)

    @staticmethod
    def test_x_roundtrip():
        p = CableParameters(
            L=500 * um,
            N=101,
            x=np.linspace(0, 500, 101) * um
        )

        p_si = p.to_numerical()

        assert isinstance(p_si.x, np.ndarray)
        assert np.isclose(p_si.x[-1], 500e-6)

        p2 = CableParameters.from_numerical(p_si)

        assert have_same_dimensions(p2.x, meter)
        assert np.allclose(
            p.x / meter,
            p2.x / meter
        )

    def test_change_N_recomputes_spatial_quantities(self):
        Rm = 2 * 1E4 * ohm * cm ** 2

        # Initial coarse discretization
        p11 = CableParameters(
            c_m=1 * uF / cm ** 2,
            Rm=Rm,
            gL=1 / Rm,
            ra=100 * ohm * cm,
            L=500.0 * um,
            N=11,
            r0=2 * um,
            I_e=150 * pampere,
        )

        # Check derived quantities exist
        self.assertIsNotNone(p11.gL)
        self.assertIsNotNone(p11.tau)
        self.assertIsNotNone(p11.b)
        self.assertIsNotNone(p11.dx)

        # Create x manually if this is not done in __post_init__
        x11 = np.linspace(0, p11.L, p11.N)

        # Store physical quantities
        c_m_11 = p11.c_m
        Rm_11 = p11.Rm
        gL_11 = p11.gL
        tau_11 = p11.tau
        b_11 = p11.b
        L_11 = p11.L
        r0_11 = p11.r0

        # Refine discretization
        p101 = p11.with_property(N=101)

        # Recompute x after changing N
        x101 = np.linspace(0, p101.L, p101.N)

        p101 = p101.with_property(x=x101)

        # --- Physical parameters must be unchanged ---
        self.assertAlmostEqual(1, p101.c_m / c_m_11)
        self.assertAlmostEqual(1, p101.Rm / Rm_11)
        self.assertAlmostEqual(1, p101.gL / gL_11)
        self.assertAlmostEqual(1, p101.tau / tau_11)
        self.assertAlmostEqual(1, p101.b / b_11)
        self.assertAlmostEqual(1, p101.L / L_11)
        self.assertAlmostEqual(1, p101.r0 / r0_11)

        self.assertEqual(11, p11.N)
        self.assertEqual(101, p101.N)

        self.assertAlmostEqual(50, p11.dx / um)
        self.assertAlmostEqual(5, p101.dx / um)

        self.assertEqual(11, len(p11.x))
        self.assertEqual(101, len(p101.x))

        self.assertAlmostEqual(0, p11.x[0] / um)
        self.assertAlmostEqual(500, p11.x[-1] / um)
        self.assertAlmostEqual(0, p101.x[0] / um)
        self.assertAlmostEqual(500, p101.x[-1] / um)



if __name__ == '__main__':
    unittest.main()
