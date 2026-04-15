"""Sanity tests for μ–σ curve and fsolve: fast checks, no plots. Run with sanity test runner."""
import unittest
from brian2 import mV, Hz

from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import (
    SiegertGradients,
    newton_fsolve_find_mu_for_fixed_sigma,
)
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control


class LookForAllSolutionsSanityTests(unittest.TestCase):
    """Unit tests for fsolve and Siegert μ–σ: solution exists and rate matches."""

    def test_look_for_one_solution_using_fsolve(self):
        siegert_gradient = SiegertGradients.for_experiment(palmer_control)
        r_target = 0.3 * Hz
        sigma_v = 3 * mV
        solution = newton_fsolve_find_mu_for_fixed_sigma(
            siegert_gradient=siegert_gradient, sigma_v=sigma_v, r_target=r_target
        )
        self.assertAlmostEqual(-60.32809490445717, solution / mV)
        self.assertAlmostEqual(
            r_target / Hz,
            siegert_gradient.firing_rate(mu_v=solution, sigma_v=sigma_v) / Hz,
        )

    def test_fsolve_sigma_small_mu_close_to_threshold(self):
        """With small σ (1e-5 mV), fsolve finds μ close to threshold."""
        siegert_gradient = SiegertGradients.for_experiment(palmer_control)
        theta = palmer_control.neuron_params.theta
        r_target = 0.3 * Hz
        sigma_small = 1e-5 * mV
        solution = newton_fsolve_find_mu_for_fixed_sigma(
            siegert_gradient=siegert_gradient,
            sigma_v=sigma_small,
            r_target=r_target,
            mu_0=theta + 0.5 * mV,
        )
        self.assertAlmostEqual(
            float(theta / mV), float(solution / mV), delta=2.0,
            msg="μ for small σ should be close to threshold",
        )

    def test_fsolve_sigma_stable_rate_matches_target(self):
        """With σ=3 mV, fsolve finds μ such that firing rate at (μ, σ) equals 0.3 Hz."""
        siegert_gradient = SiegertGradients.for_experiment(palmer_control)
        r_target = 0.3 * Hz
        sigma_stable = 3 * mV
        solution = newton_fsolve_find_mu_for_fixed_sigma(
            siegert_gradient=siegert_gradient,
            sigma_v=sigma_stable,
            r_target=r_target,
        )
        rate_at_solution = siegert_gradient.firing_rate(
            mu_v=solution, sigma_v=sigma_stable
        )
        self.assertAlmostEqual(
            float(r_target / Hz), float(rate_at_solution / Hz), places=3,
            msg="Firing rate at solution should be 0.3 Hz",
        )
