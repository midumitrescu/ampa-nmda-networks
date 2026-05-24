import unittest

import numpy as np
from matplotlib import pyplot as plt

from brian2 import Hz, mV

from Plotting import show_plots_non_blocking
from iteration_12_siegert.SolutionByLine import fit_mu_and_sigma_two_rates_and_delta_mu, compute_data_and_fit, \
    plot_data_vs_line_fit
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients
from iteration_12_transfer_function_of_lif_neurons.config import default_diffusion_lif_config


class SolutionsByLineTestCase(unittest.TestCase):
    def test_something(self):
        lif_config = default_diffusion_lif_config
        siegert_gradient = SiegertGradients.for_lif_config(lif_config)

        target_rates = [0.05 * Hz, 0.18 * Hz]
        delta_mu = 0.7

        control_line, mk801_line, mu_v, sigma_v = fit_mu_and_sigma_two_rates_and_delta_mu(delta_mu, lif_config,
                                                                                          target_rates)

        self.assertAlmostEqual(-47.63479608034277, mu_v / mV, places=5)
        self.assertAlmostEqual(1.9099, sigma_v / mV, places=4)
        self.assertAlmostEqual(50E-3, siegert_gradient.firing_rate(mu_v=mu_v, sigma_v=sigma_v) / Hz,  delta=1e-4)
        self.assertAlmostEqual(180E-3, siegert_gradient.firing_rate(mu_v=mu_v + delta_mu * mV, sigma_v=sigma_v) / Hz, delta=1e-3)

        self.assertEqual( -0.27540724290884105, control_line.slope)
        self.assertEqual(-0.25015639560297126,  mk801_line.slope)
        self.assertEqual(-11.016289716353642,  control_line.intercept)
        self.assertEqual(-10.00625582411885,  mk801_line.intercept)


        print(f"delta mu {delta_mu}: m mk801= {mk801_line.slope}, m control = {control_line.slope}"
              f"b mk801 = {mk801_line.intercept}, b control = {control_line.intercept}")

    def test_refactoring(self):
        lif_config = default_diffusion_lif_config

        mus_to_sigmas_mk801, line_fit_data_mk801 = compute_data_and_fit(lif_config=lif_config.with_label("MK801"),
                                                                        r_target=0.05 * Hz)
        mus_to_sigmas_control, line_fit_data_control = compute_data_and_fit(lif_config=lif_config.with_label("Control"),
                                                                            r_target=0.18 * Hz)

        x = np.linspace(-65, -40, 1001)

        # Apply the line equation
        y = line_fit_data_control.slope * x + line_fit_data_control.intercept

        plt.plot(x, y, label=f"y = {line_fit_data_control.slope:.2f}x - {np.abs(line_fit_data_control.intercept):.2f}")
        plt.plot(mus_to_sigmas_control.mus, mus_to_sigmas_control.sigmas, label=f"Data")

        plt.legend()
        plt.xlabel("$\mu$")
        plt.ylabel("$\sigma$")
        plt.title("Line through two points")

        show_plots_non_blocking()

        plot_data_vs_line_fit([mus_to_sigmas_mk801, mus_to_sigmas_control],
                              [line_fit_data_mk801, line_fit_data_control], caller_test_case=self)


if __name__ == '__main__':
    unittest.main()
