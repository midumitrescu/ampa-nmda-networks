import unittest

import numpy as np
from brian2 import meter, siemens, second
from numpy.testing import assert_almost_equal

from Configuration import Experiment, NetworkParams
from iteration_5_nmda.network_with_nmda import wang_model_with_extra_variables
from iteration_6_step_external_input import network_with_separated_external_and_network_input
from iteration_6_step_external_input.network_with_separated_external_and_network_input import \
    wang_model_with_separated_external_vs_network_input, plot_simulation
from iteration_6_step_external_input.network_with_step_inactivation_not_working import sim_and_plot
from iteration_6_step_external_input.nnmda_input_canceled_after_1_s import simulate_with_1s_step_input
from iteration_6_step_external_input.step_input_example import generate_step_input


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
class NMDANetworkWithStepInputTestCase(unittest.TestCase):

    '''
    Mode 1: Network is quiescent
    '''
    def test_show_mode_1(self):
        np.random.seed(0)

        new_config = {
            "N": 1000,
            "sim_time": 10_000,
            "t_range": [[0, 10_000]],

            NetworkParams.KEY_NU_E_OVER_NU_THR: 5.6,

            NetworkParams.KEY_EPSILON: 0.3,
            "g": 4,
            "g_ampa": 2 * 1e-06,
            "g_gaba": 2 * 1e-06,

            "record_N": 10,
            "hidden_variables_to_record": ["sigmoid_v", "x", "g_nmda", "I_nmda", "one_minus_g_nmda"],
            "model": wang_model_with_separated_external_vs_network_input,
            "panel": f"Show a network setting that does not start"
        }

        object_under_test = Experiment(new_config)

        rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor = simulate_with_1s_step_input(
            object_under_test, in_testing=True, eq=None, step_length= 5 * second)
        plot_simulation(object_under_test, rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor)

    '''
        Mode 2: Network is firing then quickly dying off.
    '''
    def test_show_mode_2(self):
        np.random.seed(0)

        for g_ampa in [5, 6]:
            new_config = {
                "N": 1000,
                "sim_time": 10_000,
                "t_range": [[0, 10_000]],

                NetworkParams.KEY_NU_E_OVER_NU_THR: 5.6,

                NetworkParams.KEY_EPSILON: 0.3,
                "g": 4,
                "g_ampa": g_ampa * 1e-06,
                "g_gaba": g_ampa * 1e-06,

                "record_N": 10,
                "hidden_variables_to_record": ["sigmoid_v", "x", "g_nmda", "I_nmda", "one_minus_g_nmda"],
                "model": wang_model_with_separated_external_vs_network_input,
                "panel": f"Show a network that starts but does not show sustained activity. G Ampa = {g_ampa}"
            }

            object_under_test = Experiment(new_config)

            rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor = simulate_with_1s_step_input(
                object_under_test, in_testing=True, eq=None, step_length= 5 * second)
            plot_simulation(object_under_test, rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor)

    '''
        Mode 3: Network is firing then activity is self-sustaining.
    '''
    def test_show_mode_3(self):
        np.random.seed(0)

        for g_ampa in [8 , 9, 10]:
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
                "panel": f"Show a network that starts and shows sustained activity after external input is shut down. G Ampa = {g_ampa}"
            }

            object_under_test = Experiment(new_config)

            rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor = simulate_with_1s_step_input(
                object_under_test, in_testing=True, eq=None, step_length= 5 * second)
            plot_simulation(object_under_test, rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor)

    def test_check_that_activity_does_not_die_off(self):
        np.random.seed(0)

        new_config = {
            "N": 1000,
            "sim_time": 40_000,
            "t_range": [[0, 40_000]],

            NetworkParams.KEY_NU_E_OVER_NU_THR: 5.6,
            NetworkParams.KEY_EPSILON: 0.3,
            "g": 4,
            "g_ampa": 8 * 1e-06,
            "g_gaba": 8 * 1e-06,

            "record_N": 10,
            "hidden_variables_to_record": ["sigmoid_v", "x", "g_nmda", "I_nmda", "one_minus_g_nmda"],
            "model": wang_model_with_separated_external_vs_network_input,
            "panel": f"Check that activity does not die off after some time"
        }

        object_under_test = Experiment(new_config)

        rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor = simulate_with_1s_step_input(
            object_under_test, in_testing=True, eq=None, step_length= 2 * second)
        plot_simulation(object_under_test, rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor)