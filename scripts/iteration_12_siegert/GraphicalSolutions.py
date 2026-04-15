import sys

from loguru import logger

from Plotting import show_plots_non_blocking
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients
from iteration_12_transfer_function_of_lif_neurons.config import default_diffusion_lif_config, DiffusionLIFConfig

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

import unittest

from brian2 import Hz, mvolt, mV
import matplotlib.pyplot as plt

from iteration_12_transfer_function_of_lif_neurons.LookForAllSolutionsMuSigma import (
    mu_to_sigma_for_constant_rate, mu_to_sigma_for_constant_gain,
    compute_sigma_necessary_for_given_rate_derivative_and_mean, )
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control



class SolveByGrapicalSolutionScripts(unittest.TestCase):

    def test_solve_graphical_for_two_rates(self):
        nmda_block_mu_to_sigma = mu_to_sigma_for_constant_rate(
            default_diffusion_lif_config.with_label("MK-801"), r_target=0.05 * Hz
        )
        control_mu_to_sigma = mu_to_sigma_for_constant_rate(
            default_diffusion_lif_config.with_label("Control"), r_target=0.3 * Hz
        )

        plt.plot(nmda_block_mu_to_sigma.mus, nmda_block_mu_to_sigma.sigmas, color="orange", label=f"{nmda_block_mu_to_sigma.exp_label}, r = {nmda_block_mu_to_sigma.r_target / Hz} Hz",
                 lw=2)
        plt.plot(control_mu_to_sigma.mus - 0.7, control_mu_to_sigma.sigmas, color="black",
                 label=f"{control_mu_to_sigma.exp_label}, r = {control_mu_to_sigma.r_target / Hz} Hz",
                 lw=2)
        plt.xlabel(r"$\mu_v$ [mV]")
        plt.ylabel(r"$\sigma_v$ [mV]")
        plt.title(r"$\mu$ vs $\sigma_v$ dependency for constant firing rate "
                  "predicted by first time passage formula")
        plt.tight_layout()
        plt.legend()
        show_plots_non_blocking(caller_test_case=self)

        from scipy.interpolate import interp1d

        f1 = interp1d(nmda_block_mu_to_sigma.mus, nmda_block_mu_to_sigma.sigmas, kind='cubic')
        f2 = interp1d(control_mu_to_sigma.mus - 0.7, control_mu_to_sigma.sigmas, kind='cubic')

        def f(x):
            return f1(x) - f2(x)

        from scipy.optimize import brentq

        x_intersect = brentq(f, -59, -41)
        y_intersect = f1(x_intersect)

        print("mu solution ", x_intersect)
        print("sigma solution ", x_intersect)
        sg = SiegertGradients.default()
        print(f"First rate: {sg.firing_rate(x_intersect, y_intersect)}. Second rate {sg.firing_rate(x_intersect + 0.7, y_intersect)}")

    def test_solve_graphical_for_rate_and_derivative(self):
        nmda_block_mu_to_sigma = mu_to_sigma_for_constant_gain(
            default_diffusion_lif_config.with_label("MK-801"), gain= (0.3 - 0.05) * Hz / ( 0.7 * mvolt )
        )

    def test_check_components_of_binary_search_1(self):
        lif_config_mk801 = default_diffusion_lif_config.with_label("MK-801")

        siegert_gradients = SiegertGradients.for_lif_config(lif_config_mk801)
        siegert_gradients.d_rate_d_mu(mu_v = -50 * mvolt, sigma_v = 0.1 * mvolt)
        print(siegert_gradients.d_rate_d_mu(mu_v = -50 * mvolt, sigma_v = 0.1 * mvolt))
        target_gain = (0.3 - 0.05) * Hz / (0.7 * mV)

        print(target_gain)
        compute_sigma_necessary_for_given_rate_derivative_and_mean(mu = -50 * mvolt,
                                                                   target_gain=target_gain,
                                                                   lif_config=lif_config_mk801)









