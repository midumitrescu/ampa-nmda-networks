import unittest

import numpy as np

from Configuration import Experiment, NetworkParams
from iteration_5_nmda.network_with_nmda import wang_model_extended
from iteration_6_step_external_input.network_with_step_input import simulate_with_step_input, sim_and_plot
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

    def test_simulate_with_step_input(self):
        np.random.seed(0)

        new_config = {
            "N": 1000,
            "sim_time": 5_000,

            NetworkParams.KEY_NU_E_OVER_NU_THR:10,

            NetworkParams.KEY_EPSILON: 0.3,

            "record_N": 10,
            "hidden_variables_to_record": ["sigmoid_v", "x", "g_nmda", "I_nmda", "one_minus_g_nmda"],
            "model": wang_model_extended
        }

        object_under_test = Experiment(new_config)

        sim_and_plot(object_under_test)


if __name__ == '__main__':
    unittest.main()
