import unittest

from brian2 import nsiemens, mV

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_8_compute_mean_steady_state.models_and_configs import wang_recurrent_config
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control
rom iteration_12_siegert.SiegertScripts import mean_and_sigma



class MyTestCase(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
