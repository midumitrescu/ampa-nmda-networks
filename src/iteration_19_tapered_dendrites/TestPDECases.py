import unittest

import numpy as np

from brian2 import ms, um, uamp, cm, have_same_dimensions, ufarad, ohm, second, mV, volt, Hz, meter, uF, coulomb
from brian2.units.allunits import mampere, pampere, ampere
from scipy.sparse import diags

from iteration_19_tapered_dendrites.CylindricalDendritesPDE import dirac_delta_unitless
from iteration_19_tapered_dendrites.TaperredDendritesPDE import dirac_delta
from numpy.testing import assert_allclose

from CylindricalDendritesPDE import dirac_delta as dirac_cylindrical
from iteration_19_tapered_dendrites.data import to_SI


class TestPDECases(unittest.TestCase):

    def setUp(self):
        self.dx = 1 * um
        self.dt = 1 * ms

        self.x = np.arange(10) * self.dx

        self.t0 = 5 * ms
        self.w = 1 * uamp / cm
        self.L = 500.0 * um
        self.N = 11
        self.r_0 = 2 * um
        self.r_L = 0.5 * um

        k = (1 - self.r_L / self.r_0) / self.L
        self.r_of_x = self.r_0 * (1 - k * self.x)

    def test_before_impulse(self):
        result = dirac_delta(
            t=4 * ms,
            t0=self.t0,
            dt=self.dt,
            x0=5 * um,
            x=self.x,
            r_of_x=self.r_of_x,
            dx=self.dx,
            w=self.w,
        )

        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
        self.assertEqual(result.shape, (10,))
        assert_allclose(result / (uamp / um), 0)

    def test_after_impulse(self):
        result = dirac_delta(
            t=6 * ms,
            t0=5.99999999 * ms,
            dt=self.dt,
            x0=5 * um,
            x=self.x,
            r_of_x=self.r_of_x,
            dx=self.dx,
            w=self.w,
        )

        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
        self.assertEqual(result.shape, (10,))
        assert_allclose(result / (uamp / um), 0)

    def test_exact_node(self):
        result = dirac_delta(
            t=self.t0,
            t0=self.t0,
            dt=self.dt,
            x0=4 * um,
            x=self.x,
            r_of_x=self.r_of_x,
            dx=self.dx,
            w=self.w,
        )

        expected = np.zeros(10)
        expected[4] = 1.0

        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
        assert_allclose(result / (self.w / self.dx), expected, atol=1e-6)

    def test_halfway_between_nodes(self):
        result = dirac_delta(
            t=self.t0,
            t0=self.t0,
            dt=self.dt,
            x0=4.5 * um,
            x=self.x,
            r_of_x=self.r_of_x,
            dx=self.dx,
            w=self.w,
        )

        expected = np.zeros(10)
        expected[4] = 0.5
        expected[5] = 0.5

        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
        assert_allclose(result / (self.w / self.dx), expected, atol=1e-6)

    def test_quarter_between_nodes(self):
        result = dirac_delta(
            t=self.t0,
            t0=self.t0,
            dt=self.dt,
            x0=4.25 * um,
            x=self.x,
            r_of_x=self.r_of_x,
            dx=self.dx,
            w=self.w,
        )

        expected = np.zeros(10)
        expected[4] = 0.75
        expected[5] = 0.25

        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))

        assert_allclose(result / (self.w / self.dx), expected)

    def test_left_boundary(self):
        result = dirac_delta(
            t=self.t0,
            t0=self.t0,
            dt=self.dt,
            x0=-0.00001 * um,
            x=self.x,
            r_of_x=self.r_of_x,
            dx=self.dx,
            w=self.w,
        )

        expected = np.zeros(10)
        expected[0] = 1.0
        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
        assert_allclose(result / (self.w / self.dx), expected)

    def test_right_boundary(self):
        result = dirac_delta(
            t=self.t0,
            t0=self.t0,
            dt=self.dt,
            x0=10 * um,
            x=self.x,
            r_of_x=self.r_of_x,
            dx=self.dx,
            w=self.w,
        )

        expected = np.zeros(10)
        expected[-1] = 1.0
        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
        assert_allclose(result / (self.w / self.dx), expected)

    def test_conservation(self):
        positions = np.linspace(0, 9, 41) * um

        for x0 in positions:
            result = dirac_delta(
                t=self.t0,
                t0=self.t0,
                dt=self.dt,
                x0=x0,
                x=self.x,
                r_of_x=self.r_of_x,
                dx=self.dx,
                w=self.w,
            )
            self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
            total = np.sum(result) * self.dx
            self.assertAlmostEqual(
                float(total / self.w),
                1.0,
                places=12,
            )

    def test_only_one_or_two_nonzero_entries(self):
        positions = [
            0.5,
            2.3,
            5.8,
            8.1,
        ]

        for p in positions:
            result = dirac_delta(
                t=self.t0,
                t0=self.t0,
                dt=self.dt,
                x0=p * um,
                x=self.x,
                r_of_x=self.r_of_x,
                dx=self.dx,
                w=self.w,
            )

            self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
            nnz = np.count_nonzero(result / (self.w / self.dx))
            self.assertEqual(nnz, 2)

    def test_exact_node_has_single_nonzero(self):
        for x0 in self.x:
            result = dirac_delta(
                t=self.t0,
                t0=self.t0,
                dt=self.dt,
                x0=x0,
                x=self.x,
                r_of_x=self.r_of_x,
                dx=self.dx,
                w=self.w,
            )

            self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
            nnz = np.count_nonzero(np.abs(result / (mampere / cm ** 2)) > 1e-12)
            self.assertEqual(nnz, 1, msg=f"Node {x0} has 2 nonzero entries: {result[np.where(result != 0)]}")

    def test_dirac_delta_units(self):
        c_m = 1 * uF / cm ** 2
        Rm = 2 * 1E4 * ohm * cm ** 2
        gL = 1 / Rm
        # 1. Parameters
        L = 500.0 * um
        N = 11
        x = np.linspace(0, L, N)
        dx = L / (N - 1)  # um
        dt = 0.01 * ms

        tau = c_m / gL  # ms

        assert have_same_dimensions(tau, 1 * second)

        print("tau=", tau)

        w = 100 * pampere
        r_0 = 2 * um

        result = dirac_cylindrical(
            x0=-5 * um,
            t0=6.00000009 * ms,
            t=6 * ms,
            dt=dt,
            tau_m=tau,
            x=x,
            r_of_x=r_0,
            dx=dx,
            I_e=w
        )

        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
        self.assertEqual(result[1:].shape, (10,))
        self.assertAlmostEqual(31830.98861837907, result[0] / (uamp / cm**2), places=8)
        self.assertAlmostEqual(318.3098861837907, result[0] / (ampere / meter**2), places=8)
        assert_allclose(result[1:] / (uamp / cm**2), 0)

        # ------------------------------------------------------------------
        # Charge conservation:
        # ∑ i_e · (2πr) · dx · dt = I_e τ_m
        # ------------------------------------------------------------------

        injected_charge = np.sum(result * (2 * np.pi * r_0) * dx * dt)

        self.assertTrue(have_same_dimensions(injected_charge, coulomb))
        self.assertAlmostEqual(
            injected_charge / coulomb,
            (w * tau) / coulomb,
            places=12
        )

        # Equivalent invariant:
        self.assertAlmostEqual(
            np.sum(result) / (ampere / meter ** 2),
            (w * tau / (2 * np.pi * r_0 * dx * dt)) / (ampere / meter ** 2),
            places=12
        )

    def test_dirac_delta_units_for_no_input(self):
        c_m = 1 * uF / cm ** 2
        Rm = 2 * 1E4 * ohm * cm ** 2
        gL = 1 / Rm
        # 1. Parameters
        L = 500.0 * um
        N = 11
        x = np.linspace(0, L, N)
        dx = L / (N - 1)  # um

        tau = c_m / gL  # ms

        assert have_same_dimensions(tau, 1 * second)

        print("tau=", tau)

        w = 100 * pampere
        r_0 = 2 * um

        result = dirac_cylindrical(
            x0=-5 * um,
            t0=5 * ms,
            t=6 * ms,
            dt=0.1 * ms,
            tau_m=tau,
            x=x,
            r_of_x=r_0,
            dx=dx,
            I_e=w
        )

        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
        self.assertEqual(result.shape, (11,))
        assert_allclose(result / (uamp / cm ** 2), 0)

    def test_dirac_unitless_returns_float_array(self):

        c_m = 1 * uF / cm ** 2
        Rm = 2 * 1E4 * ohm * cm ** 2
        gL = 1 / Rm
        # 1. Parameters
        L = 500.0 * um
        N = 11
        x = np.linspace(0, L, N)
        dx = L / (N - 1)  # um

        tau = c_m / gL  # ms

        w = 100 * pampere
        r_0 = 2 * um

        result = dirac_delta_unitless(
            x0=to_SI(-5 * um, meter),
            t0=to_SI(6.00000009 * ms, second),
            t=to_SI(6 * ms, second),
            dt=to_SI(0.1 * ms, second),
            tau_m=to_SI(tau, second),
            x=to_SI(x, meter),
            r_of_x=to_SI(r_0, meter),
            dx=to_SI(dx, meter),
            I_e=to_SI(w, ampere)
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)

        self.assertEqual(result[0], 31.83098861837907)
        assert_allclose(result[1:] / (uamp / cm ** 2), 0)



class TestLinearTaperCableOneStep(unittest.TestCase):

    def setUp(self):
        # Small grid for testing
        self.N = 11

        self.L = 500 * um
        self.x = np.linspace(0, self.L, self.N)
        self.dx = self.L / (self.N - 1)

        # Parameters
        self.cm = 1 * ufarad / cm ** 2
        Rm = 2e4 * ohm * cm ** 2
        self.gL = 1 / Rm
        self.ra = 100 * ohm * cm

        self.tau = self.cm / self.gL

        self.r0 = 2 * um
        self.rL = 0.5 * um

        self.k = (1 - self.rL / self.r0) / self.L
        self.r_of_x = self.r0 * (1 - self.k * self.x)


        self.a = (
                self.k * self.r0 /
                (self.cm * self.ra *
                 np.sqrt(1 + self.r0 ** 2 * self.k ** 2))
        )

        self.b = (
                self.r0 * (1 - self.k * self.x) /
                (2 * self.cm * self.ra *
                 np.sqrt(1 + self.r0 ** 2 * self.k ** 2))
        )

        self.dt = 0.01 * ms

        self.A = self.build_matrix()

    def build_matrix(self):
        lower = self.b[1:] / self.dx ** 2 + self.a / (2 * self.dx)
        main = -1 / self.tau - 2 * self.b / self.dx ** 2
        upper = self.b[:-1] / self.dx ** 2 - self.a / (2 * self.dx)

        # Sparse tridiagonal matrix
        A = diags(
            diagonals=[lower, main, upper],
            offsets=[-1, 0, 1],
            format="lil"
        )

        return A.tocsr().toarray() * (1 / second)

    def test_matrix_initialization(self):

        self.assertAlmostEqual(50, self.dx / um)

        A = self.A
        self.assertEqual(
            self.A.shape,
            (self.N, self.N)
        )

        self.assertTrue(have_same_dimensions(self.A[0, 0], ms ** -1))

        self.assertAlmostEqual(self.a / (cm/second), 29.999865000911253, msg="Manuscript say 30 cm / s but that is an approximation."
                                                                                       "we are ignoring in the manuscript 1/sqrt(1 + k**2r0**2)")

        self.assertAlmostEqual(self.a / (meter / second), 0.29999865000911253,
                               msg="Manuscript say 0.3 m / s but that is an approximation."
                                   "we are ignoring in the manuscript 1/sqrt(1 + k**2r0**2)")

        self.assertAlmostEqual(self.b[0] / (cm ** 2 / second), 0.9999955000303749, msg="Manuscript say 1 cm^2 / s but that is an approximation."
                                                                                       "we are ignoring in the manuscript 1/sqrt(1 + k**2r0**2)")
        self.assertAlmostEqual(self.b[-1] / (cm ** 2 / second), 0.24999888,
                               msg="Manuscript say 0.25 cm^2 / s but that is an approximation."
                                   "we are ignoring in the manuscript 1/sqrt(1 + k**2r0**2)")

        self.assertAlmostEqual(self.b[0] / (meter ** 2 / second), 0.00009999955000303749)
        self.assertAlmostEqual(self.b[-1] / (meter ** 2 / second), 0.000024999888)

        self.assertAlmostEqual((self.b[0] / self.dx**2) / Hz, 4E4,
                              msg="Manuscript says 3.96 x 10E6 BUT our dx there is 5 um while here it is 50! So, 10^2 difference", places=0)

        self.assertAlmostEqual((self.b[-1] / self.dx ** 2) / Hz, 1E4,
                               msg="Manuscript says 10E6 BUT our dx there is 5 um while here it is 50! So, 10^2 difference", places=0)

        self.assertAlmostEqual(50, 1/self.tau * second)



    def test_one_forward_euler_step_no_input(self):
        # Simple voltage profile
        V = np.linspace(
            -70,
            20,
            self.N
        ) * mV

        dVdt = self.A @ V

        V_next = V + self.dt * dVdt

        self.assertTrue(have_same_dimensions(dVdt[0], volt / second))
        self.assertTrue(have_same_dimensions(V_next[0], volt))
        # The step should change voltage
        self.assertFalse(np.allclose(V_next / mV, V / mV))

    def test_constant_voltage_decay(self):
        # If V is spatially constant, diffusion terms vanish
        V0 = -70 * mV

        V = np.ones(self.N) * V0

        dVdt = self.A @ V

        expected = - V0 / self.tau

        assert_allclose(
            dVdt / (mV / ms),
            np.ones(self.N) *
            (expected / (mV / ms)),
            rtol=1e-10,
            atol=1e-10
        )

    def test_synaptic_input_one_step(self):
        V = np.zeros(self.N) * mV

        I_syn = dirac_delta(
            x0=250 * um,
            t0=0 * ms,
            t=0 * ms,
            dx=self.dx,
            dt=self.dt,
            x=self.x,
            r_of_x=self.r_of_x,
            w=1 * uamp / cm
        )

        dVdt = self.A @ V + I_syn / self.cm

        V_next = V + self.dt * dVdt

        # Some voltage must appear
        self.assertGreater(
            np.max(np.abs(V_next)),
            0 * mV
        )

        # Check dimensions
        self.assertTrue(
            have_same_dimensions(
                dVdt[0],
                volt / second
            )
        )


if __name__ == '__main__':
    unittest.main()
