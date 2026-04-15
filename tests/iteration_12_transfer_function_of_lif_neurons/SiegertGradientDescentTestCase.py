import unittest
import matplotlib.pyplot as plt
import numpy as np
from brian2 import mV, second, ms, Hz, have_same_dimensions, is_dimensionless

from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import rate_LIF_whitenoise, \
    integration_limits, SiegertGradientDescent, create_anneal_decay_schedule, plot_grad_descent
from Plotting import show_plots_non_blocking
from iteration_8_compute_mean_steady_state.models_and_configs import palmer_experiment_0_1_Hz_with_NMDA_block
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control

tau_m = 20 * ms
mu = -55 * mV
v_reset = -65 * mV
theta = -50 * mV
sigma = 5 * mV
tau_ref = 2 * ms

r_target = 0.3 * Hz

learning_rate=1e-3 * (mV * second)**2

class MyTestCase(unittest.TestCase):

    def test_units_of_gradient_Loss(self):
        object_under_test = SiegertGradientDescent(tau_m=tau_m, theta=theta, v_reset=v_reset, tau_ref=tau_ref)

        self.assertTrue(have_same_dimensions(1 * Hz,
                                             rate_LIF_whitenoise(mu=mu, tau_membrane=tau_m, sigma_v=sigma, theta=theta,
                                                                 V_reset=v_reset, tau_ref=tau_ref)))
        self.assertTrue(have_same_dimensions(1 * Hz,
                                             rate_LIF_whitenoise(mu=mu, tau_membrane=tau_m, sigma_v=sigma, theta=theta,
                                                                 V_reset=v_reset, tau_ref=tau_ref) - 0.05 * Hz))

        # mu_v, sigma_v, r_target
        grad_L_mu, grad_L_sigma = object_under_test.gradient_loss(mu_v=mu, sigma_v=sigma, r_target=r_target)
        self.assertTrue(have_same_dimensions(1 * Hz ** 2 / mV, grad_L_mu))
        self.assertTrue(have_same_dimensions(1 * Hz ** 2 / mV, grad_L_sigma))

        self.assertTrue(have_same_dimensions(1 * mV, learning_rate * grad_L_mu))
        self.assertTrue(have_same_dimensions(1 * mV, learning_rate * grad_L_sigma))


if __name__ == '__main__':
    unittest.main()
