import unittest

import numpy as np
from brian2 import ms, um, uamp, cm, have_same_dimensions, ufarad, ohm, second, mV, volt, Hz, meter, uF, coulomb, uvolt
from brian2.units.allunits import mampere, pampere, ampere
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from scipy.sparse import diags, eye
from scipy.sparse.linalg import factorized

from CylindricalDendritesPDE import dirac_delta as dirac_cylindrical
from iteration_19_tapered_dendrites.CylindricalDendritesPDE import dirac_delta_unitless, \
    plot_tuckwell_solution_closed_cable, plot_tuckwell_solution_closed_cable_difference
from iteration_19_tapered_dendrites.TaperredDendritesPDE import dirac_delta
from iteration_19_tapered_dendrites.data import to_SI, CableParameters, NumericalCableParameters


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
        self.assertAlmostEqual(31830.98861837907, result[0] / (uamp / cm ** 2), places=8)
        self.assertAlmostEqual(318.3098861837907, result[0] / (ampere / meter ** 2), places=8)
        assert_allclose(result[1:] / (uamp / cm ** 2), 0)

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

    def test_diract_components_with_units(self):

        c_m = 1 * uF / cm ** 2
        Rm = 2 * 1E4 * ohm * cm ** 2
        gL = 1 / Rm
        # 1. Parameters
        L = 500.0 * um
        N = 11
        x = np.linspace(0, L, N)
        dx = L / (N - 1)  # um
        dt = 0.1 * ms

        tau = c_m / gL  # ms

        w = 100 * pampere
        r0 = 2 * um

        delta_xt = 1.0 / (dx * dt)

        i_e = w * tau / (2 * np.pi * r0) * delta_xt

        self.assertTrue(have_same_dimensions(i_e, ampere / meter ** 2))
        self.assertAlmostEqual(31.83098861837907, i_e /  pampere * um**2)
        self.assertAlmostEqual(31.83098861837907, to_SI(i_e))

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

        self.assertAlmostEqual(self.a / (cm / second), 29.999865000911253,
                               msg="Manuscript say 30 cm / s but that is an approximation."
                                   "we are ignoring in the manuscript 1/sqrt(1 + k**2r0**2)")

        self.assertAlmostEqual(self.a / (meter / second), 0.29999865000911253,
                               msg="Manuscript say 0.3 m / s but that is an approximation."
                                   "we are ignoring in the manuscript 1/sqrt(1 + k**2r0**2)")

        self.assertAlmostEqual(self.b[0] / (cm ** 2 / second), 0.9999955000303749,
                               msg="Manuscript say 1 cm^2 / s but that is an approximation."
                                   "we are ignoring in the manuscript 1/sqrt(1 + k**2r0**2)")
        self.assertAlmostEqual(self.b[-1] / (cm ** 2 / second), 0.24999888,
                               msg="Manuscript say 0.25 cm^2 / s but that is an approximation."
                                   "we are ignoring in the manuscript 1/sqrt(1 + k**2r0**2)")

        self.assertAlmostEqual(self.b[0] / (meter ** 2 / second), 0.00009999955000303749)
        self.assertAlmostEqual(self.b[-1] / (meter ** 2 / second), 0.000024999888)

        self.assertAlmostEqual((self.b[0] / self.dx ** 2) / Hz, 4E4,
                               msg="Manuscript says 3.96 x 10E6 BUT our dx there is 5 um while here it is 50! So, 10^2 difference",
                               places=0)

        self.assertAlmostEqual((self.b[-1] / self.dx ** 2) / Hz, 1E4,
                               msg="Manuscript says 10E6 BUT our dx there is 5 um while here it is 50! So, 10^2 difference",
                               places=0)

        self.assertAlmostEqual(50, 1 / self.tau * second)

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
            dVdt[1:-1] / (mV / ms),
            np.ones(self.N-2) *
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

Rm = 2 * 1E4 * ohm * cm ** 2
default_params = CableParameters(c_m=1 * uF / cm ** 2,
                                 rm=Rm,
                                 gL=1 / Rm,
                                 ra=100 * ohm * cm,
                                 L=500.0 * um,
                                 N=101,
                                 r0=2 * um,
                                 I_e=150 * pampere)


def create_difussion_matrix(p: NumericalCableParameters):

    x = p.x
    dx = p.dx

    tau = p.tau
    b = p.b

    difussion = np.ones(len(x) - 1) * b / dx ** 2
    difussion_decay = np.ones(len(x)) * (-1 / tau - 2 * b / dx ** 2)

    # Sparse tridiagonal matrix
    A = diags(
        diagonals=[difussion, difussion_decay, difussion],
        offsets=[-1, 0, 1],
        format="lil"
    )

    # ensure boundary conditions automatically in A matrix
    #A[0, 0] = -1 / tau - 2 * b / dx ** 2
    A[0, 1] = 2 * b / dx ** 2
    A[-1, -2] = 2 * b / dx ** 2
    #A[-1, -1] = -1 / tau - 2 * b / dx ** 2
    return A

class TestForwardEulerInCylinderOneStep(unittest.TestCase):

    def test_parameter_values_small_N(self):
        simulation_params = default_params.with_property(t=10 * ms, N=11)
        p = simulation_params.to_numerical()

        A = create_difussion_matrix(p)
        self.assertEqual(1E-4, p.b)
        self.assertEqual(1, simulation_params.b / cm ** 2 * second)

        self.assertAlmostEqual(np.sqrt(2)*1E-3, p.lambd())
        self.assertAlmostEqual(np.sqrt(2) * 1E-3, simulation_params.lambd() / meter)
        self.assertAlmostEqual(np.sqrt(2)*1E3, simulation_params.lambd() / um)
        self.assertAlmostEqual(np.sqrt(2) / 10, simulation_params.lambd() / cm)

        diag = -1/p.tau - 2 /25 * 1E6
        diag_cp = -1/0.02 - 2 /25 * 1E6
        assert_array_equal(np.ones(9) * diag, A.diagonal()[1:10])

        diff = 1 / 25 * 1E6

        assert_allclose(np.ones(9) * diff, A.diagonal(-1)[:-1])
        self.assertAlmostEqual(2 / 25 * 1E6, A.diagonal(-1)[-1])

        assert_allclose(np.ones(9) * diff, A.diagonal(1)[1:])
        self.assertAlmostEqual(2 / 25 * 1E6, A.diagonal(1)[0])




    def test_parameter_values_moderate_N(self):
        simulation_params = default_params.with_property(t=10 * ms, N=101)
        p = simulation_params.to_numerical()

        A = create_difussion_matrix(p)
        self.assertEqual(1E-4, p.b)
        self.assertEqual(1, simulation_params.b / cm ** 2 * second)
        self.assertAlmostEqual(np.sqrt(2) * 1E-3, simulation_params.lambd() / meter)
        self.assertAlmostEqual(np.sqrt(2) * 1E3, simulation_params.lambd() / um)
        self.assertAlmostEqual(np.sqrt(2) / 10, simulation_params.lambd() / cm)

        self.assertAlmostEqual(5E-6, p.dx)
        self.assertAlmostEqual(5, simulation_params.dx / um)

        diag = -1 / p.tau - 2 / 25 * 1E8
        diag_cp = -1 / 0.02 - 2 / 25 * 1E8
        assert_allclose(np.ones(101) * diag, A.diagonal())
        assert_allclose(np.ones(101) * diag_cp, A.diagonal())

        diff = 1 / 25 * 1E8

        assert_allclose(np.ones(99) * diff, A.diagonal(-1)[:-1])
        self.assertAlmostEqual(2 / 25 * 1E8, A.diagonal(-1)[-1])

        assert_allclose(np.ones(99) * diff, A.diagonal(1)[1:])
        self.assertAlmostEqual(2 / 25 * 1E8, A.diagonal(1)[0])

    def test_parameter_values_large_N(self):
        simulation_params = default_params.with_property(t=10 * ms, N=1001)
        p = simulation_params.to_numerical()

        A = create_difussion_matrix(p)
        self.assertEqual(1E-4, p.b)
        self.assertEqual(1, simulation_params.b / cm ** 2 * second)
        self.assertAlmostEqual(np.sqrt(2) * 1E-3, simulation_params.lambd() / meter)
        self.assertAlmostEqual(np.sqrt(2) * 1E3, simulation_params.lambd() / um)
        self.assertAlmostEqual(np.sqrt(2) / 10, simulation_params.lambd() / cm)

        self.assertEqual(5E-7, p.dx)
        self.assertEqual(0.5, simulation_params.dx / um)

        diag = -1 / p.tau - 2 / 25 * 1E10
        diag_cp = -1 / 0.02 - 2 / 25 * 1E10
        assert_allclose(np.ones(1001) * diag, A.diagonal())
        assert_allclose(np.ones(1001) * diag_cp, A.diagonal())

        diff = 1 / 25 * 1E10

        assert_allclose(np.ones(999) * diff, A.diagonal(-1)[:-1])
        self.assertAlmostEqual(2 / 25 * 1E10, A.diagonal(-1)[-1])

        assert_allclose(np.ones(999) * diff, A.diagonal(1)[1:])
        self.assertAlmostEqual(2 / 25 * 1E10, A.diagonal(1)[0])


    def test_one_dirac_step(self):
        simulation_params = default_params.with_property(t=10 * ms, N=6)
        p = simulation_params.to_numerical()

        # 1. Parameters
        dx = p.dx
        dt = to_SI(1E-8 * second)

        A = create_difussion_matrix(p)

        V = np.zeros(len(p.x))

        x0 = to_SI(200 * um)
        t0 = to_SI(1 * ms)
        I_e = to_SI(1.5 * pampere)

        dx = p.dx

        def synaptic_input_profile(t, x0, dt):
            return dirac_delta_unitless(x0=x0, t0=t0, x=p.x, t=t, dx=dx, dt=dt, I_e=I_e,
                                        tau_m=p.tau, r_of_x=p.r0)

        t = t0 - 0.1 * dt

        i_of_t = synaptic_input_profile(t, x0, dt)
        inputed_current = dx * dt * np.sum(i_of_t)

        self.assertEqual(I_e * p.tau / (2 * np.pi * p.r0), inputed_current)


        V_n_euler = np.copy(V)
        V_n_plus_1_euler = V_n_euler + dt * (A @ V_n_euler + i_of_t)

        print(f"{I_e * p.tau / (2 * np.pi * p.r0 * dx) : .6e}")
        print(f"{V_n_plus_1_euler[2] : .6e}")

        self.assertAlmostEqual(I_e * p.tau / (2 * np.pi * p.r0 * dx), V_n_plus_1_euler[2])
        self.assertAlmostEqual(I_e * p.tau / (2 * np.pi * p.r0), V_n_plus_1_euler[2] * dx)


    def test_current_injection_increased_x_discretization(self):
        simulation_params = default_params.with_property(t=10 * ms, N=101)
        p = simulation_params.to_numerical()

        # 1. Parameters
        dx = p.dx
        dt = to_SI(1E-8 * second)

        A = create_difussion_matrix(p)

        V = np.zeros(len(p.x))

        x0 = to_SI(250 * um)
        t0 = to_SI(1 * ms)
        I_e = to_SI(1.5 * pampere)

        dx = p.dx

        def synaptic_input_profile(t, x0, dt):
            return dirac_delta_unitless(x0=x0, t0=t0, x=p.x, t=t, dx=dx, dt=dt, I_e=I_e,
                                        tau_m=p.tau, r_of_x=p.r0)

        t = t0 - 0.1 * dt

        x0_index = 50

        i_of_t = synaptic_input_profile(t, x0, dt)
        inputed_current = dx * dt * np.sum(i_of_t)

        self.assertEqual(I_e * p.tau / (2 * np.pi * p.r0), inputed_current)

        V_n_euler = np.copy(V)
        V_n_plus_1_euler = V_n_euler + dt * (A @ V_n_euler + i_of_t)

        print(f"{I_e * p.tau / (2 * np.pi * p.r0 * dx) : .6e}")
        print(f"{V_n_plus_1_euler[2] : .6e}")

        self.assertAlmostEqual(I_e * p.tau / (2 * np.pi * p.r0 * dx), V_n_plus_1_euler[x0_index])
        self.assertAlmostEqual(I_e * p.tau / (2 * np.pi * p.r0), V_n_plus_1_euler[x0_index] * dx)

def load_simulation(filename):

    data = np.load(filename)

    times = data["times"]
    V_s = data["V_s"]

    simulation_params = default_params.with_SI_properties(
        t=times[-1],
        N=int(data["N"]),
        dt=data["dt"],
        L=data["L"],
        I_e=data["I_e"],
    )

    p = simulation_params.to_numerical()

    x0 = data["x0"] * meter
    t0 = 0.1 * ms

    return times, V_s, p, x0, t0

class TestCrankNicolsonOneStep(unittest.TestCase):

    def setUp(self):
        simulation_params = default_params.with_property(t=10*ms, N=1001)
        self.si_units = simulation_params.to_numerical()
        self.dt = to_SI(1E-8 * second)

        # 1. Parameters
        dx = self.si_units.dx
        tau = self.si_units.tau
        dt = self.dt

        # Spatial domain and initial condition
        x = self.si_units.x
        b = self.si_units.b
        difussion = np.ones(len(x) - 1) * b / dx ** 2
        difussion_decay = np.ones(len(x)) * (-1 / tau - 2 * b / dx ** 2)


        # Sparse tridiagonal matrix
        A = diags(
            diagonals=[difussion, difussion_decay, difussion],
            offsets=[-1, 0, 1],
            format="lil"
        )

        # ensure boundary conditions automatically in A matrix
        A[0, 0] = -1 / tau - 2 * b / dx ** 2
        A[0, 1] = 2 * b / dx ** 2
        A[-1, -2] = 2 * b / dx ** 2
        A[-1, -1] = -1 / tau - 2 * b / dx ** 2

        I = eye(A.shape[0], format="csc")

        self.A = A
        # Crank-Nicolson matrices
        self.L = (I - 0.5 * dt * A).tocsc()
        self.R = (I + 0.5 * dt * A).tocsc()

        # Factorize once
        self.solve = factorized(self.L)

    def test_one_step(self):
        V = np.zeros(len(self.si_units.x))
        dt = self.dt
        self.x0 = to_SI(250 * um)
        I_e = to_SI(1.5 * pampere)
        dx = self.si_units.dx

        def synaptic_input_profile(t, x0, dt):
            return dirac_delta_unitless(x0=x0, t0=to_SI(1 * ms), x=self.si_units.x, t=t, dx=dx, dt=dt, I_e=I_e,
                                        tau_m=self.si_units.tau, r_of_x=self.si_units.r0)

        t = to_SI(1 * ms) - 1E-10
        x0 = to_SI(250 * um)

        id_x0 = np.searchsorted(self.si_units.x, x0)
        # RHS
        syn_input_t = synaptic_input_profile(t=t, x0=x0, dt=dt / 2)
        syn_input_t_half = synaptic_input_profile(t=t + dt / 2, x0=x0, dt=dt / 2)
        input_t_and_t_half = 1 / 2 * dt * (syn_input_t + syn_input_t_half)

        inputed_current = dx * dt/2 * np.sum(syn_input_t)

        self.assertEqual(I_e * self.si_units.tau / (2 * np.pi * self.si_units.r0), inputed_current)

        self.assertEqual(0 , np.sum(syn_input_t_half))


        V_n_euler = np.copy(V)
        V_n_plus_1_euler = V_n_euler + dt * ( self.A @ V_n_euler + input_t_and_t_half)

        print(f"{I_e * self.si_units.tau / (2 * np.pi * self.si_units.r0) : .6e}")
        print(f"{V_n_plus_1_euler[500] * dt : .6e}")

        self.assertEqual(I_e * self.si_units.tau / (2 * np.pi * self.si_units.r0 * dx), V_n_plus_1_euler[500])

        rhs = self.R @ V + input_t_and_t_half
        # Solve:
        # (I - dt/2 A) V_new = rhs
        V_t_plus_1 = self.solve(rhs)

        print("Input ", input_t_and_t_half[id_x0-3:id_x0+3])
        print("R@V ", (self.R @ V)[id_x0-3:id_x0+3])
        print("R@V + input", (self.R @ V + input_t_and_t_half)[id_x0-3:id_x0+3])
        print("V t+1",V_t_plus_1[id_x0-4:id_x0+4] * volt / uvolt)

        V_t_plus_1_uvolt = V_t_plus_1[id_x0 - 4:id_x0 + 4] * volt / uvolt

        mu = self.si_units.b * self.dt / self.si_units.dx ** 2
        lambda_ = self.dt / self.si_units.tau

        print(mu, lambda_)

        injected_charge = (
                np.sum(input_t_and_t_half)
                *
                (2 * np.pi * self.si_units.r0 * self.si_units.dx)
        )

        expected_voltage = I_e * self.si_units.tau / (2 * np.pi * self.si_units.r0 * self.si_units.dx * self.si_units.c_m)
        print("Expected:", expected_voltage)
        print("Actual:", V_t_plus_1[id_x0])

        self.assertAlmostEqual(expected_voltage, V_t_plus_1[id_x0])

        expected_charge = I_e * self.si_units.tau

        self.assertAlmostEqual(
            injected_charge,
            expected_charge,
            delta=expected_charge * 1e-12
        )

    def test_prefactor(self):


        numerical_prefactor = self.si_units.I_e * self.si_units.tau / (2 * np.pi * self.si_units.r0)

        theory_prefactor = self.si_units.I_e * self.si_units.R_lambda()
        # r_lambda =  self.rm / (2 * np.pi * self.r0 * self.lambd())
        # lambd = math.sqrt(self.r0 * self.rm / (2 * self.ra))
        print(numerical_prefactor / theory_prefactor)

        print(self.si_units.tau * self.si_units.lambd() / self.si_units.rm)

    def test_tuckwell_theory_vs_simulation(self):

        times, V_s, p, x0, t0 = load_simulation("saved_simulations/cable_sim_N301_L500_dt10ns_x0250um.npz")
        desired_positions = [250]

        plot_tuckwell_solution_closed_cable(times=times, V_s = V_s, p=p, desired_positions= desired_positions, t0 = t0, x0 = x0, sim_type="Crank-Nicolson")
        for offset in np.arange(5, 100, step=50):
            plot_tuckwell_solution_closed_cable_difference(V_s=V_s, desired_positions=desired_positions, p=p, t0=t0,
                                                           times=times, x0=x0, t0_offset=offset)


if __name__ == '__main__':
    unittest.main()
