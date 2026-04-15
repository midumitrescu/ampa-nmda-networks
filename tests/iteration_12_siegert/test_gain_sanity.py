"""Sanity tests for Siegert gain: find_sigma_by_f_solve and related. Run with sanity test runner."""
import unittest
from brian2 import mV, Hz

from iteration_12_siegert.gain_computations import (
    find_sigma_for_mu_producing_rate_gain,
    palmer_control,
)
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import (
    SiegertGradients,
)


class SiegertGainSanityTests(unittest.TestCase):
    """Unit tests for gain/sigma solving."""

    def test_find_sigma_by_f_solve(self):
        siegert_gradient = SiegertGradients.for_experiment(palmer_control)
        mu_fixed = palmer_control.neuron_params.theta - 10 * mV
        gain_target = 2.5 * Hz / mV
        sigma = find_sigma_for_mu_producing_rate_gain(
            palmer_control, mu=mu_fixed, gain=gain_target
        )
        self.assertAlmostEqual(4.676, sigma / mV, places=3)
        self.assertAlmostEqual(
            gain_target * mV / Hz,
            siegert_gradient.d_rate_d_mu(mu_v=mu_fixed, sigma_v=sigma) * mV / Hz,
            places=8,
        )
