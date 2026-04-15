import unittest

from brian2 import Hz, mV

from iteration_12_siegert.df_utils import prepare_experiment_with_N_tot
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control, palmer_nmda_block


class CheckSimulationValues(unittest.TestCase):

    def test_values_for_control_simulation(self):

        experiment = palmer_control.with_properties({
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "g_e", "g_i", "g_nmda"],
            "t_range": [0, 60 * 1000],
            "in_testing": False
        })
        up_state_base = experiment.network_params.up_state.params

        for n_current in range(1, 100):
            object_under_test = prepare_experiment_with_N_tot(n_current, up_state_base=up_state_base, base=experiment)
            self.assertEqual(0.25, object_under_test.network_params.up_state.gamma)
            self.assertEqual(0.8, 1 / (1 + object_under_test.network_params.up_state.gamma))
            self.assertEqual(0.2, object_under_test.network_params.up_state.gamma / (1 + object_under_test.network_params.up_state.gamma))
            self.assertEqual(n_current, object_under_test.network_params.up_state.N)
            self.assertEqual(n_current, object_under_test.network_params.up_state.N_E + object_under_test.network_params.up_state.N_I)

            if n_current % 5 == 0:
                self.assertEqual(0.8, object_under_test.network_params.up_state.N_E / object_under_test.network_params.up_state.N)
                self.assertEqual(0.2, object_under_test.network_params.up_state.N_I / object_under_test.network_params.up_state.N)
            else:
                self.assertGreater(object_under_test.network_params.up_state.N_E /  object_under_test.network_params.up_state.N, 0.8)
                self.assertLess(object_under_test.network_params.up_state.N_I /  object_under_test.network_params.up_state.N, 0.2)

            self.assertEqual(82, object_under_test.network_params.up_state.nu / Hz)
            # due to control, we have N NMDA

            self.assertEqual(10, object_under_test.network_params.up_state.N_NMDA)
            self.assertEqual(10, object_under_test.network_params.up_state.nu_nmda / Hz)

            self.assertEqual(-50, object_under_test.neuron_params.theta / mV)

    def test_values_for_control_simulation_no_firing(self):
        palmer_control_no_firing = palmer_control.with_properties({
            "panel": "Control_no_firing",
            "theta": 100
        })

        experiment = palmer_control_no_firing.with_properties({
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "g_e", "g_i", "g_nmda"],
            # "t_range": [0, 1000],
            "t_range": [0, 60 * 1000],
            "in_testing": False
        })
        up_state_base = experiment.network_params.up_state.params

        for n_current in range(1, 100):
            object_under_test = prepare_experiment_with_N_tot(n_current, up_state_base=up_state_base, base=experiment)
            self.assertEqual(0.25, object_under_test.network_params.up_state.gamma)
            self.assertEqual(0.8, 1 / (1 + object_under_test.network_params.up_state.gamma))
            self.assertEqual(0.2, object_under_test.network_params.up_state.gamma / (1 + object_under_test.network_params.up_state.gamma))
            self.assertEqual(n_current, object_under_test.network_params.up_state.N)
            self.assertEqual(n_current, object_under_test.network_params.up_state.N_E + object_under_test.network_params.up_state.N_I)

            if n_current % 5 == 0:
                self.assertEqual(0.8, object_under_test.network_params.up_state.N_E / object_under_test.network_params.up_state.N)
                self.assertEqual(0.2, object_under_test.network_params.up_state.N_I / object_under_test.network_params.up_state.N)
            else:
                self.assertGreater(object_under_test.network_params.up_state.N_E /  object_under_test.network_params.up_state.N, 0.8)
                self.assertLess(object_under_test.network_params.up_state.N_I /  object_under_test.network_params.up_state.N, 0.2)

            self.assertEqual(82, object_under_test.network_params.up_state.nu / Hz)
            # due to control, we have N NMDA

            self.assertEqual(10, object_under_test.network_params.up_state.N_NMDA)
            self.assertEqual(10, object_under_test.network_params.up_state.nu_nmda / Hz)

            self.assertEqual(100, object_under_test.neuron_params.theta / mV)


    def test_values_for_NMDA_Block_simulation(self):

        experiment = palmer_nmda_block.with_properties({
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "g_e", "g_i", "g_nmda"],
            "t_range": [0, 60 * 1000],
            "in_testing": False
        })
        up_state_base = experiment.network_params.up_state.params

        for n_current in range(1, 100):
            object_under_test = prepare_experiment_with_N_tot(n_current, up_state_base=up_state_base, base=experiment)
            self.assertEqual(0.25, object_under_test.network_params.up_state.gamma)
            self.assertEqual(0.8, 1 / (1 + object_under_test.network_params.up_state.gamma))
            self.assertEqual(0.2, object_under_test.network_params.up_state.gamma / (1 + object_under_test.network_params.up_state.gamma))
            self.assertEqual(n_current, object_under_test.network_params.up_state.N)
            self.assertEqual(n_current, object_under_test.network_params.up_state.N_E + object_under_test.network_params.up_state.N_I)

            if n_current % 5 == 0:
                self.assertEqual(0.8, object_under_test.network_params.up_state.N_E / object_under_test.network_params.up_state.N)
                self.assertEqual(0.2, object_under_test.network_params.up_state.N_I / object_under_test.network_params.up_state.N)
            else:
                self.assertGreater(object_under_test.network_params.up_state.N_E /  object_under_test.network_params.up_state.N, 0.8)
                self.assertLess(object_under_test.network_params.up_state.N_I /  object_under_test.network_params.up_state.N, 0.2)

            self.assertEqual(82, object_under_test.network_params.up_state.nu / Hz)

            self.assertEqual(0, object_under_test.network_params.up_state.N_NMDA)
            self.assertEqual(0, object_under_test.network_params.up_state.nu_nmda / Hz)

            self.assertEqual(-50, object_under_test.neuron_params.theta / mV)



if __name__ == '__main__':
    unittest.main()
