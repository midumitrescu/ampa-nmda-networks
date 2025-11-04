import unittest

import numpy as np
from brian2 import meter, siemens

from Configuration import Experiment, NetworkParams
from iteration_5_nmda.network_with_nmda import wang_model_with_extra_variables
from iteration_6_step_external_input import network_with_separated_external_and_network_input
from iteration_6_step_external_input.network_with_step_inactivation_not_working import sim_and_plot


class IncorrectNMDAInactivationTestCase(unittest.TestCase):
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

            NetworkParams.KEY_NU_E_OVER_NU_THR: 5.45,

            NetworkParams.KEY_EPSILON: 0.3,
            "g": 4,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,

            "record_N": 10,
            "hidden_variables_to_record": ["sigmoid_v", "x", "g_nmda", "I_nmda", "one_minus_g_nmda"],
            "model": wang_model_with_extra_variables
        }

        object_under_test = Experiment(new_config)

        _, _, _, g_monitor, _ = sim_and_plot(object_under_test)
        last_values_g_e = g_monitor.g_e[1][-1000:] / siemens * (meter ** 2)
        np.all(last_values_g_e != 0)
        print("Keep this as example of synapse that does not get deactivated after some time. "
              "Because this did not happened, the mechanism of deactivating synapses did not work.")

    def test_conditions_for_bifurcation_due_to_network_activity(self):
        np.random.seed(0)

        new_config = {
            "N": 1000,
            "sim_time": 10_000,
            "t_range": [[0, 5000], [5000, 10000], [9_000, 9_500]],

            NetworkParams.KEY_NU_E_OVER_NU_THR: 5.5,

            NetworkParams.KEY_EPSILON: 0.3,
            "g": 4,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,

            "record_N": 10,
            "hidden_variables_to_record": ["sigmoid_v", "x", "g_nmda", "I_nmda", "one_minus_g_nmda"],
            "model": wang_model_with_extra_variables,
            "panel": "Network bifurcates towards asynchonous irregular"
        }

        object_under_test = Experiment(new_config)

        sim_and_plot(object_under_test)

    def test_long_simulation_with_potential_for_bifurcation(self):
        np.random.seed(0)

        new_config = {
            "N": 1000,
            "sim_time": 20_000,
            "t_range": [[5000, 20000], [15_750, 16_250]],

            NetworkParams.KEY_NU_E_OVER_NU_THR: 5.45,

            NetworkParams.KEY_EPSILON: 0.3,
            "g": 4,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,

            "record_N": 10,
            "hidden_variables_to_record": ["sigmoid_v", "x", "g_nmda", "I_nmda", "one_minus_g_nmda"],
            "model": wang_model_with_extra_variables
        }

        object_under_test = Experiment(new_config)
        _, _, _, g_monitor, _ = sim_and_plot(object_under_test)
