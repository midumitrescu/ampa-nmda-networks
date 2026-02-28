import unittest

from brian2 import nsiemens, mV

from iteration_12_siegert.compare_vm_sigma_vm_theory_vs_simulation.computations import mean_and_sigma
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_8_compute_mean_steady_state.models_and_configs import wang_recurrent_config
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control


class SiegertTestCase(unittest.TestCase):
    def test_call_one_mean_sigma_with_no_nmda(self):
        up_state_base = {
            "N": 2000,
            "nu": 82,
            "N_nmda": 0,
            "nu_nmda": 0,
        }

        result = mean_and_sigma(n=1, up_state_base=up_state_base, base=palmer_control)

        self.assertEqual(result.mu_v_no_nmda, result.mu_v_with_nmda)
        self.assertEqual(result.sigma_v_no_nmda, result.sigma_v_with_nmda)
        self.assertEqual(result.firing_rate_no_nmda, result.firing_rate_with_nmda)

    def test_call_one_mean_sigma_with_nmda(self):
        up_state_base = {
            "N": 2000,
            "nu": 82,
            "N_nmda": 10,
            "nu_nmda": 10,
        }

        result =  mean_and_sigma(n=1, up_state_base=up_state_base, base=palmer_control)

        self.assertLess(result.mu_v_no_nmda, result.mu_v_with_nmda)
        self.assertLess(result.sigma_v_no_nmda, result.sigma_v_with_nmda)
        self.assertLessEqual(result.firing_rate_no_nmda, result.firing_rate_with_nmda)

    def test_computing_E_0_with_nmda(self):
        base = Experiment(wang_recurrent_config)

        self.assertEqual(0.2, base.effective_time_constant_up_state.mean_x_nmda())
        self.assertEqual(0.9, base.effective_time_constant_up_state.mean_s_nmda())
        self.assertAlmostEqual(0.02198201216, base.effective_time_constant_up_state.compute_mean_g_nmda() / nsiemens)
        self.assertAlmostEqual(-48.75363871, base.effective_time_constant_up_state.E_0_with_nmda() / mV)

    def test_compute_firing_rate_at_threshold(self):
        base = Experiment(wang_recurrent_config)
        object_under_test = SiegertGradients.for_experiment(base)

        self.assertEqual(-50 , object_under_test.theta / mV)

        self.assertEqual(0, object_under_test.firing_rate(-50.00000001 * mV, sigma_v = 0 * mV))


if __name__ == '__main__':
    unittest.main()
