import unittest

import numpy as np
from brian2 import mV, ms, Hz, have_same_dimensions, is_dimensionless

from iteration_12_transfer_function_of_lif_neurons.SiegerGradientDescentTestCases import \
    compute_LIF_curves_for_mus_sigmas
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import rate_LIF_whitenoise, \
    integration_limits, SiegertGradientDescent

tau_m = 20 * ms
mu = -55 * mV
v_reset = -65 * mV
theta = -50 * mV
sigma = 5 * mV
tau_ref = 2 * ms

r_target = 0.3 * Hz

class SiegertBaseComputationsCase(unittest.TestCase):
    def test_rate_LIF_white_noise_can_handle_zero_noise(self):
        mu = -35 * mV

        T = tau_m * np.log((mu - v_reset) / (mu - theta))
        rate_determ = 1. / (T + tau_ref)

        self.assertEqual(rate_determ / Hz,
                         rate_LIF_whitenoise(mu, tau_membrane=tau_m, sigma_v=0 * mV, theta=theta, tau_ref=tau_ref,
                                             V_reset=v_reset) / Hz)

    def test_sigma_zero_above_threshold_returns_deterministic_rate(self):
        """When σ≈0 and μ > θ: rate = 1/(T + τ_ref) with T = τ_m·ln((μ−V_r)/(μ−θ))."""
        mu = -35 * mV  # above theta=-50
        T = tau_m * np.log((mu - v_reset) / (mu - theta))
        expected = 1.0 / (T + tau_ref)
        actual = rate_LIF_whitenoise(mu, tau_membrane=tau_m, sigma_v=0 * mV, theta=theta, tau_ref=tau_ref, V_reset=v_reset)
        self.assertAlmostEqual(float(expected / Hz), float(actual / Hz), places=9)

    def test_sigma_zero_below_threshold_returns_zero(self):
        """When σ≈0 and μ ≤ θ: rate = 0."""
        mean = -50.0 * mV  # at or below theta=-40
        theta = -40 * mV
        actual = rate_LIF_whitenoise(mean, tau_membrane=20 * ms, sigma_v=0 * mV, theta=theta, tau_ref=2 * ms, V_reset=-55 * mV)
        self.assertAlmostEqual(0.0, float(actual / Hz), places=9)

    def test_sigma_extremely_small_below_threshold_returns_zero(self):
        """When σ≈0 and μ ≤ θ: rate = 0."""
        mean = -50.0 * mV  # at or below theta=-40
        theta = -40 * mV
        actual = rate_LIF_whitenoise(mean, tau_membrane=20 * ms, sigma_v=1E-10 * mV, theta=theta, tau_ref=2 * ms, V_reset=-55 * mV)
        self.assertAlmostEqual(0.0, float(actual / Hz), places=9)

    def test_limit_units(self):

        object_under_test = SiegertGradientDescent(tau_m=tau_m, theta = theta, v_reset=v_reset, tau_ref = tau_ref)

        self.assertTrue(have_same_dimensions(1*mV, mu))

        lower_limit, upper_limit = integration_limits(V_mean=mu, V_reset=v_reset, sigma_v=sigma, theta=theta)
        self.assertTrue(is_dimensionless(lower_limit))
        self.assertTrue(is_dimensionless(upper_limit))

        self.assertTrue(is_dimensionless(object_under_test.E(lower_limit)))
        self.assertTrue(is_dimensionless(object_under_test.E(upper_limit)))

    def test_lif_computation_for_multiple_sigmas_returns_correct_shape(self):
        mus = np.linspace(-65, -45, 1001) * mV
        sigmas = np.array([0.5, 1., 2., 4., 6.]) * mV

        object_under_test = compute_LIF_curves_for_mus_sigmas(mus, sigmas)
        self.assertEqual(len(object_under_test[0]), len(mus))
        self.assertEqual((len(sigmas), len(mus)), object_under_test.shape)


if __name__ == '__main__':
    unittest.main()
