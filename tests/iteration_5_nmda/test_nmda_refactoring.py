import unittest
import numpy as np
import time
import matplotlib.pyplot as plt
from brian2 import ms

from numpy.testing import assert_array_equal

from Configuration import Experiment, NetworkParams, PlotParams
from iteration_5_nmda_refactored.network_with_nmda import sim, sim_and_plot, create_connectivity_matrix

plt.rcParams['text.usetex'] = True

'''
We have 3 big issue that we are having an issue with:

1. the model is incorrect! we need a g_nmda_bar for the max gnmda available. However, we want to use same notation
as Wang. Otherwise this will get cumbersome and not understandable.
2. We want to break all components of the model to make it very easy to understand what is what
3. We want, if we introduce some extra variables in another model, for the simulate function to still work!
'''

extended_model = """
dv/dt = 1/C * (- I_L - I_syn - I_Exc - I_Inh - I_nmda): volt (unless refractory)
        
I_L = g_L * (v-E_leak): amp / meter ** 2
I_syn = g_e_syn * (v-E_ampa): amp / meter ** 2
I_Exc = g_e * (v-E_ampa): amp / meter ** 2
I_Inh = g_i * (v-E_gaba): amp / meter ** 2
I_nmda = g_nmda * (v - E_ampa): amp / meter** 2

dg_e_syn/dt = -g_e_syn / tau_ampa  : siemens / meter**2
dg_e/dt = -g_e / tau_ampa : siemens / meter**2
dg_i/dt = -g_i / tau_gaba  : siemens / meter**2

g_nmda = g_nmda_max * sigmoid_v * s_nmda: siemens / meter**2
g_nmda_max: siemens / meter**2

ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha * x_nmda * (1 - s_nmda) : 1
x_nmda_not_cliped : 1
dx_nmda/dt = - x_nmda / tau_nmda_rise : 1

sigmoid_v = 1/(1 + exp(-0.062 * v/mvolt) * (MG_C/mmole / 3.57)) : 1
one_minus_s_nmda = 1 - s_nmda : 1
alpha_x_t = alpha * x_nmda: Hz
s_drive = alpha * x_nmda * (1 - s_nmda) : Hz
"""

def sorted(array: np.ndarray) -> np.ndarray:
    stacked = array.T  # Transpose to make each row a data point

    # Sort first by the 0th column (sources) and then by the 1st column (targets)
    sorted_stacked = stacked[np.lexsort((stacked[:, 1], stacked[:, 0]))]
    return sorted_stacked.T

class RefactoredNMDAInput(unittest.TestCase):
    def test_nmda_configuration_is_parsed(self):
        new_config = {
            "sim_time": 10,
            "record_N": 1,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["sigmoid_v", "x_nmda", "g_nmda", "I_nmda", "one_minus_s_nmda"],
            Experiment.KEY_SELECTED_MODEL: extended_model
        }

        object_under_test = Experiment(new_config)
        sim(experiment=object_under_test)

    def test_sim_and_plot(self):
        new_config = {
            "sim_time": 100,
            "record_N": 2,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["sigmoid_v", "x_nmda", "g_nmda", "I_nmda", "one_minus_s_nmda"],
            Experiment.KEY_SELECTED_MODEL: extended_model
        }

        object_under_test = Experiment(new_config)
        sim_and_plot(object_under_test)

    def test_sim_and_plot(self):
        new_config = {
            "sim_time": 100,
            "N": 10,
            "record_N": 2,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["sigmoid_v", "x_nmda", "g_nmda", "I_nmda", "one_minus_s_nmda"],
            Experiment.KEY_SELECTED_MODEL: extended_model
        }

        object_under_test = Experiment(new_config)
        sim_and_plot(object_under_test)

    def test_sim_and_plot_with_external_input(self):
        interesting_nus = [1.8, 1.9, 2., 2.1]
        interesting_nus = [2]
        for nu_ext_over_nu_thr in interesting_nus:
            config = {
                NetworkParams.KEY_NU_E_OVER_NU_THR: nu_ext_over_nu_thr,
                NetworkParams.KEY_EPSILON: 0.1,
                "g": 4,
                "g_ampa": 2.4e-06,
                "g_gaba": 2.4e-06,
                "g_nmda": 1.6e-06,

                Experiment.KEY_SELECTED_MODEL: extended_model,
                #Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["sigmoid_v",  "g_nmda", "I_nmda", "one_minus_s_nmda", "s_drive"],
                Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda"],
                "record_N": 10,

                #Experiment.KEY_SIMULATION_CLOCK: 0.005 * ms,

                "t_range": [[0, 2000]],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE, PlotParams.AvailablePlots.HIDDEN_VARIABLES]
            }
            sim_and_plot(Experiment(config))

    def test_creation_of_paired_inputs(self):
        N = 10
        epsilon = 0.1
        config = {
            "N": N,
            NetworkParams.KEY_EPSILON: epsilon,
            "t_range": [[120, 170]],
            "in_testing": True
        }

        experiment = Experiment(config)
        connectivity_matrix, pairs = create_connectivity_matrix(experiment)

        sources, targets = connectivity_matrix.nonzero()
        assert_array_equal(pairs, np.vstack((sources, targets)))

    def test_creation_of_connectivity_matrix(self):
        N = 1000
        epsilon = 0.1
        for i in range(100):
            config = {
                "N": N,
                NetworkParams.KEY_EPSILON: epsilon,
                "t_range": [[120, 170]],
                "in_testing": False
            }

            experiment = Experiment(config)
            connectivity_matrix, pairs = create_connectivity_matrix(experiment)

            self.assertEqual(connectivity_matrix.shape, (N, N))
            assert_array_equal(np.zeros(N), connectivity_matrix.diagonal())
            assert_array_equal(np.ones(N) * int(N * epsilon), connectivity_matrix.sum(axis=1))

            sources, targets = connectivity_matrix.nonzero()
            should_be_same_source = np.vstack((pairs[0],sources))
            should_be_same_target = np.vstack((pairs[1],targets))
            assert_array_equal(sorted(pairs), sorted(np.vstack((sources, targets))))

    def test_creation_time_(self):
        config = {
            "N": 10000,
            NetworkParams.KEY_EPSILON: 0.1,
            "t_range": [[120, 170]],
            "in_testing": True
        }
        experiment = Experiment(config)

        start = time.perf_counter()
        connectivity_matrix = create_connectivity_matrix(experiment)

        first = time.perf_counter()
        i, j = connectivity_matrix.nonzero()

        second = time.perf_counter()

        print(f"Matrix generation time: {first - start}")
        print(f"Connectivity computation: {second - first}")

    def test_sorting_2_d_array(self):
        arr_1 = np.array([[1, 2, 3, 4],
                  [1, 2, 3, 4]])

        arr_2 = np.array([[1, 3, 2, 4],
                          [1, 3, 2, 4]])

        arr_3 = np.array([[1, 1, 1, 1],
                          [1, 3, 2, 4]])

        arr_4 = np.array([[1, 2, 3, 4, 0, 2, 0, 0, 1, 1, 3, 3, 4, 4, 4, 2, 1, 3, 2, 0],
                          [3, 4, 1, 0, 2, 1, 4, 3, 0, 2, 4, 0, 2, 3, 1, 0, 4, 2, 0, 1]])

        assert_array_equal(sorted(arr_1), arr_1)
        assert_array_equal(sorted(arr_2), arr_1)
        assert_array_equal(sorted(arr_3), [[1, 1, 1, 1], [1, 2, 3, 4]])
        assert_array_equal(sorted(arr_4), [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
                                                            [1, 2, 3, 4, 0, 2, 3, 4, 0, 0, 1, 4, 0, 1, 2, 4, 0, 1, 2, 3]])



if __name__ == '__main__':
    unittest.main()
