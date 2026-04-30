import unittest

from brian2 import mV, Hz
from numpy.testing import assert_allclose

from iteration_12_siegert.GraphicalSolutions import taylor_error
from iteration_12_transfer_function_of_lif_neurons.LookForAllSolutionsMuSigma import mu_to_sigma_for_constant_gain, \
    mu_to_sigma_for_constant_rate
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients
from iteration_12_transfer_function_of_lif_neurons.config import default_diffusion_lif_config


class GraphicSolutionsTestCase(unittest.TestCase):

    @staticmethod
    def test_search_for_rate_produces_reasonable_result():
        rate_mk801 = 0.05 * Hz
        object_under_test = mu_to_sigma_for_constant_rate(
            default_diffusion_lif_config.with_label("MK-801"), r_target=rate_mk801
        )

        assert_allclose(object_under_test.firing_rates(SiegertGradients.default()), rate_mk801)
        assert_allclose(object_under_test.firing_rates_no_units(SiegertGradients.default()), rate_mk801 / Hz)


    @staticmethod
    def test_search_for_gain_produces_reasonable_result():
        gain = 0.09699832465740052 * Hz / mV
        object_under_test = mu_to_sigma_for_constant_gain(
            default_diffusion_lif_config.with_label("MK-801"), gain=gain
        )

        sg = SiegertGradients.default()
        assert_allclose(object_under_test.d_rate_d_mus(sg), gain)
        assert_allclose(object_under_test.d_rate_d_mus_no_units(sg), gain / Hz * mV)

    def test_taylor_error_computation(self):
        mu_sol = -47.60912502853491
        sigma_sol = 1.9036551073068055
        delta_mu = 0.7
        for (mu, sigma, delta) in [(mu_sol, sigma_sol, delta_mu), (mu_sol* mV, sigma_sol, delta_mu), (mu_sol * mV, sigma_sol*mV, delta_mu), (mu_sol, sigma_sol, delta_mu*mV)]:
            object_under_test = taylor_error(mu_sol = mu, sigma_sol = sigma, delta_mu=delta, siegert_gradients=SiegertGradients.default())
            self.assertAlmostEqual(0.0622796, object_under_test/ Hz, places=6)

