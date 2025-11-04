import unittest

import numpy as np
from brian2 import meter, siemens
from numpy.testing import assert_almost_equal

from Configuration import Experiment, NetworkParams
from iteration_5_nmda.network_with_nmda import wang_model_with_extra_variables
from iteration_6_step_external_input import network_with_separated_external_and_network_input
from iteration_6_step_external_input.network_with_separated_external_and_network_input import \
    wang_model_with_separated_external_vs_netork_input
from iteration_6_step_external_input.network_with_step_inactivation_not_working import sim_and_plot
from iteration_6_step_external_input.step_input_example import generate_step_input


class MyTestCase(unittest.TestCase):

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
     Here I wanted to test a simple step current implementation.
     This resulted, however, in the issue mentioned in Step_Input.ipynb.
     
     I.E. while recording from neuron 1, that has input from external and also from network, after 1 second, 
     g_ampa still stays very much up.
     
     So, my idea was to split the inputs from the network and that was external!
     Thus, we can monitor what is network generated and what is inputted by the external input.
    '''
    def test_simulate_with_step_input(self):
        np.random.seed(0)

        new_config = {
            "N": 1000,
            "sim_time": 10_000,
            "t_range": [[0, 5000], [5000, 10000]],

            NetworkParams.KEY_NU_E_OVER_NU_THR:5.45,

            NetworkParams.KEY_EPSILON: 0.3,
            "g": 4,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,

            "record_N": 10,
            "hidden_variables_to_record": ["sigmoid_v", "x", "g_nmda", "I_nmda", "one_minus_g_nmda"],
            "model": wang_model_with_extra_variables
        }

        object_under_test = Experiment(new_config)

        sim_and_plot(object_under_test)

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
            "model": wang_model_with_separated_external_vs_netork_input
        }

        object_under_test = Experiment(new_config)

        _, _, _, g_monitor, _ = network_with_separated_external_and_network_input.sim_and_plot(object_under_test)
        last_g_ampa_ext = g_monitor.g_ext_ampa[1][-1000:] / siemens * (meter**2)
        np.all(last_g_ampa_ext == 0)


if __name__ == '__main__':
    unittest.main()
