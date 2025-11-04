import unittest

import numpy as np
from brian2 import meter, siemens, second

from Configuration import Experiment, NetworkParams
from iteration_6_step_external_input.network_with_separated_external_and_network_input import \
    wang_model_with_separated_external_vs_network_input, plot_simulation
from iteration_6_step_external_input.nnmda_input_canceled_after_1_s import simulate_with_1s_step_input
from iteration_6_step_external_input.step_input_example import generate_step_input


class NMDANetworkWithStepInputTestCase(unittest.TestCase):

    def test_N_E_can_still_be_passed_as_parameter(self):
        new_config = {
            "N_E": 100,
            "sim_time": 2_000,
            "record_N": 10,
        }

        object_under_test = Experiment(new_config)

        self.assertEqual(0.25, object_under_test.network_params.gamma)
        self.assertEqual(125, object_under_test.network_params.N)
        self.assertEqual(100, object_under_test.network_params.N_E)
        self.assertEqual(25, object_under_test.network_params.N_I)

    def test_N_can_be_passed_as_parameter(self):
        new_config = {
            "N": 125,
            "sim_time": 2_000,
            "record_N": 10,
        }

        object_under_test = Experiment(new_config)

        self.assertEqual(0.25, object_under_test.network_params.gamma)
        self.assertEqual(125, object_under_test.network_params.N)
        self.assertEqual(100, object_under_test.network_params.N_E)
        self.assertEqual(25, object_under_test.network_params.N_I)

    def test_N_can_be_passed_as_parameter(self):
        new_config = {
            "N": 125,
            "sim_time": 2_000,
            "record_N": 10,
        }
        object_under_test = Experiment(new_config)

        self.assertEqual(0.25, object_under_test.network_params.gamma)


    def test_step_experiment_example(self):
        np.random.seed(42)

        new_config = {
            "N": 100,
            "sim_time": 5_000,
            "record_N": 10,
        }

        object_under_test = Experiment(new_config)

        generate_step_input(object_under_test)

    def test_controll_network_turns_off_high_frequency_synapse_correctly(self):
        pass

    '''
    Simulate step for 1s then reduce input
    '''
    def test_simulate_with_step_input_and_separated_external_vs_network_input(self):
        np.random.seed(0)

        new_config = {
            "N": 1000,
            "sim_time": 5_000,
            "t_range": [[0, 1000], [0, 5_000]],

            NetworkParams.KEY_NU_E_OVER_NU_THR:5.45,

            NetworkParams.KEY_EPSILON: 0.3,
            "g": 4,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,

            "record_N": 10,
            "hidden_variables_to_record": ["sigmoid_v", "x", "g_nmda", "I_nmda", "one_minus_g_nmda"],
            "model": wang_model_with_separated_external_vs_network_input
        }

        object_under_test = Experiment(new_config)

        rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor = simulate_with_1s_step_input(object_under_test, in_testing=True, eq=None)
        plot_simulation(object_under_test, rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor)

        first_values_g_ampa_ext = g_monitor.g_ext_ampa[1][1000:2000] / siemens * (meter**2)
        last_values_g_ampa_ext = g_monitor.g_ext_ampa[1][-1000:] / siemens * (meter**2)
        np.all(first_values_g_ampa_ext != 0)
        np.all(last_values_g_ampa_ext == 0)

    '''
        Simulate step for 5s then reduce input
        '''

    def test_check_that_input_step_shuts_down_network_activity(self):
        np.random.seed(0)

        new_config = {
            "N": 1000,
            "sim_time": 10_000,
            "t_range": [[0, 10_000], [0, 5_000]],

            NetworkParams.KEY_NU_E_OVER_NU_THR: 5.6,

            NetworkParams.KEY_EPSILON: 0.3,
            "g": 4,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,

            "record_N": 10,
            "hidden_variables_to_record": ["sigmoid_v", "x", "g_nmda", "I_nmda", "one_minus_g_nmda"],
            "model": wang_model_with_separated_external_vs_network_input
        }

        object_under_test = Experiment(new_config)

        rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor = simulate_with_1s_step_input(
            object_under_test, in_testing=True, eq=None, step_length= 5 * second)
        plot_simulation(object_under_test, rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor)

    '''
        To a pulse input (5s on, 5s off) we can get:

        1: for g_ampa 2, the network is not firing at all
        2: for g_ampa 3, 4, the network is firing for first 5 s, then quickly quiescent after step input is shut down
        3: for g_ampa 5, 6, the network is firing for the first 5s, then "slowly" quiescent. Meaning, some activity is visible 
        for 1-2 s after step input is shut down
        4: for g_ampa 7, 8, 9, 10: the network is firing for 5 s, then continuing activity. 
        As expected, because of increased g_ampa, overall activity is higher.

        As a note, g_ampa 7 (and also g_nmda) dies off approx afer 10s-13s of input.
        '''

    def test_look_for_various_nmda_g_max(self):
        np.random.seed(0)

        for g_ampa in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            new_config = {
                "N": 1000,
                "sim_time": 10_000,
                "t_range": [[0, 10_000], [0, 5_000]],

                NetworkParams.KEY_NU_E_OVER_NU_THR: 5.6,

                NetworkParams.KEY_EPSILON: 0.3,
                "g": 4,
                "g_ampa": g_ampa * 1e-06,
                "g_gaba": g_ampa * 1e-06,

                "record_N": 10,
                "hidden_variables_to_record": ["sigmoid_v", "x", "g_nmda", "I_nmda", "one_minus_g_nmda"],
                "model": wang_model_with_separated_external_vs_network_input,
                "panel": f"Check if network can sustain activity without external input. G Ampa = {g_ampa}"
            }

            object_under_test = Experiment(new_config)

            rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor = simulate_with_1s_step_input(
                object_under_test, in_testing=True, eq=None, step_length=5 * second)
            plot_simulation(object_under_test, rate_monitor, spike_monitor, v_monitor, g_monitor,
                            internal_states_monitor)


if __name__ == '__main__':
    unittest.main()
