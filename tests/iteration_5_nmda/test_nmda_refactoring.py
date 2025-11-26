import unittest
import numpy as np
import time
import matplotlib.pyplot as plt
from brian2 import ms

from numpy.testing import assert_array_equal

from Configuration import Experiment, NetworkParams, PlotParams
from iteration_4_conductance_based_model.grid_computations_nmda_model import compare_g_nmda_vs_nu_ext_over_nu_thr
from iteration_5_nmda_refactored.network_with_nmda import sim, sim_and_plot, create_connectivity_matrix, \
    sim_with_full_nmda_connections_and_plot
from meeting_prof_schwalger_19_11 import network_under_nmda_automatically_bifurcating, network_without_nmda, \
    network_showing_synchronous_regular
from models import NMDAModels

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
            "sim_time": 100,
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

    def test_sim_and_plot_with_nmda_connection_to_inh_works(self):
        experiment = Experiment({
            NetworkParams.KEY_NU_E_OVER_NU_THR: 12,
            NetworkParams.KEY_EPSILON: 0.1,
            "g": 4,
            "g_ampa": 0.5e-06,
            "g_gaba": 0.5e-06,
            "g_nmda": 0.5e-5,
            Experiment.KEY_SELECTED_MODEL: NMDAModels.model_with_detailed_hidden_variables.eq,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda", "sigmoid_v",
                                                        "g_nmda", "I_nmda"],
            "record_N": 10,
            "t_range": [[0, 100]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.HIDDEN_VARIABLES]
        })
        sim_with_full_nmda_connections_and_plot(experiment, in_testing=True)

    def test_plot_and_grid_with_nmda_connection_to_inh_works(self):
        experiment = Experiment({
            NetworkParams.KEY_NU_E_OVER_NU_THR: 0,
            NetworkParams.KEY_EPSILON: 0.1,
            "g": 4,
            "g_ampa": 0.5e-06,
            "g_gaba": 0.5e-06,
            "g_nmda": 0,
            Experiment.KEY_SELECTED_MODEL: NMDAModels.model_with_detailed_hidden_variables.eq,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda", "sigmoid_v",
                                                        "g_nmda", "I_nmda"],
            "record_N": 10,
            "t_range": [[0, 50]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.HIDDEN_VARIABLES]
        })
        compare_g_nmda_vs_nu_ext_over_nu_thr(experiment, g_nmdas=[0.5e-5],
                                             nu_ext_over_nu_thrs=[11], nmda_to_inh_neurons = True)


    def test_bla(self):
        network_showing_synchronous_regular()

    def test_sim_and_plot_with_external_input_and_lower_clock(self):
        config = {
            NetworkParams.KEY_NU_E_OVER_NU_THR: 1.9,
            NetworkParams.KEY_EPSILON: 0.1,
            "g": 4,
            "g_ampa": 2.5e-06,
            "g_gaba": 2.5e-06,
            "g_nmda": 5e-07,

            Experiment.KEY_SELECTED_MODEL: extended_model,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda", "sigmoid_v",
                                                        "g_nmda", "I_nmda"],
            "record_N": 10,

            Experiment.KEY_SIMULATION_CLOCK: 0.005,

            "t_range": [[0, 2000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.HIDDEN_VARIABLES]
        }
        sim_and_plot(Experiment(config))


    @unittest.skip(reason="Switched generating paired input to using .nonzero(). Try to delete ASAP")
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
        connectivity_matrix = create_connectivity_matrix(experiment)

        sources, targets = connectivity_matrix.nonzero()
        assert_array_equal(connectivity_matrix, np.vstack((sources, targets)))

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
            connectivity_matrix = create_connectivity_matrix(experiment)

            self.assertEqual(connectivity_matrix.shape, (N, N))
            assert_array_equal(np.zeros(N), connectivity_matrix.diagonal())
            assert_array_equal(np.ones(N) * int(N * epsilon), connectivity_matrix.sum(axis=1))


            sources, targets = connectivity_matrix.nonzero()
            self.assertEqual(N* int(N * epsilon), len(sources))
            self.assertEqual(N * int(N * epsilon), len(targets))

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

class Scripts(unittest.TestCase):
    def test_sim_and_plot_with_external_input(self):
        #interesting_nus = [1.75, 1.8, 1.85, 1.9]
        #interesting_nus = [5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6]
        interesting_nus = [5.3, 5.4]
        for nu_ext_over_nu_thr in interesting_nus:
            config = {
                NetworkParams.KEY_NU_E_OVER_NU_THR: nu_ext_over_nu_thr,
                NetworkParams.KEY_EPSILON: 0.1,
                "g": 4,
                "g_ampa": 1e-06,
                "g_gaba": 1e-06,
                "g_nmda": 5e-06,

                Experiment.KEY_SELECTED_MODEL: extended_model,
                Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda", "sigmoid_v",  "g_nmda", "I_nmda"],
                "record_N": 10,

                #Experiment.KEY_SIMULATION_CLOCK: 0.005 * ms,

                "t_range": [[0, 500], [100, 200]],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE, PlotParams.AvailablePlots.HIDDEN_VARIABLES]
            }
            sim_and_plot(Experiment(config))

    def test_sim_and_plot_with_external_input_detailed(self):
        interesting_nus = [1.872, 1.8723]
        for nu_ext_over_nu_thr in interesting_nus:
            config = {
                NetworkParams.KEY_NU_E_OVER_NU_THR: nu_ext_over_nu_thr,
                NetworkParams.KEY_EPSILON: 0.1,
                "g": 4,
                "g_ampa": 2.5e-06,
                "g_gaba": 2.5e-06,
                "g_nmda": 5e-07,

                Experiment.KEY_SELECTED_MODEL: extended_model,
                Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda", "sigmoid_v",  "g_nmda", "I_nmda"],
                "record_N": 10,

                Experiment.KEY_SIMULATION_CLOCK: 0.005,

                "t_range": [[0, 2000], [1500, 1520]],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE, PlotParams.AvailablePlots.HIDDEN_VARIABLES]
            }
            sim_and_plot(Experiment(config))

    def test_grid_results(self):
        experiment = Experiment({
            NetworkParams.KEY_NU_E_OVER_NU_THR: 0,
            NetworkParams.KEY_EPSILON: 0.1,
            "g": 4,
            "g_ampa": 0.5e-06,
            "g_gaba": 0.5e-06,
            "g_nmda": 0,
            Experiment.KEY_SELECTED_MODEL: NMDAModels.model_with_detailed_hidden_variables.eq,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda", "sigmoid_v",
                                                        "g_nmda", "I_nmda"],
            "record_N": 10,

            Experiment.KEY_SIMULATION_CLOCK: 0.005,
            "t_range": [[0, 2000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE],

            "panel": "Corrected Network of AMPA, GABA and (a lot of) NMDA input. NMDA only between Exc"
        })
        #compare_g_nmda_vs_nu_ext_over_nu_thr(experiment, g_nmdas=[0, 0.2e-5, 0.3e-5, 0.4e-5, 0.5e-5], nu_ext_over_nu_thrs=[10.6, 10.8, 10.9])
        #compare_g_nmda_vs_nu_ext_over_nu_thr(experiment.with_property("panel", "Corrected Network of AMPA, GABA and (a lot of) NMDA input. NMDA between both Exc and Ihn"),
        #                                     g_nmdas=[0, 0.2e-5, 0.3e-5, 0.4e-5, 0.5e-5], nu_ext_over_nu_thrs=[10.6, 10.8, 10.9], nmda_to_inh_neurons=True)

        compare_g_nmda_vs_nu_ext_over_nu_thr(experiment, g_nmdas=[0, 0.4e-5, 0.5e-5],
                                             nu_ext_over_nu_thrs=[10.6, 10.8, 10.9])
        compare_g_nmda_vs_nu_ext_over_nu_thr(experiment.with_property("panel",
                                                                      "Corrected Network of AMPA, GABA and (a lot of) NMDA input. NMDA between both Exc and Ihn"),
                                             g_nmdas=[0, 0.4e-5, 0.5e-5],
                                             nu_ext_over_nu_thrs=[10.6, 10.8, 10.9], nmda_to_inh_neurons=True)


    def test_grid_results_system_bifurcates_but_does_it_bifurcate_also_for_various_nmda(self):
        experiment = Experiment({
            NetworkParams.KEY_NU_E_OVER_NU_THR: 0,
            NetworkParams.KEY_EPSILON: 0.1,
            "g": 4,
            "g_ampa": 2.5e-06,
            "g_gaba": 2.5e-06,
            "g_nmda": 5e-07,

            Experiment.KEY_SELECTED_MODEL: extended_model,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda",
                                                        "sigmoid_v", "g_nmda", "I_nmda"],
            "record_N": 10,

            #Experiment.KEY_SIMULATION_CLOCK: 0.005,
            "t_range": [[0, 2000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.HIDDEN_VARIABLES]
        })
        compare_g_nmda_vs_nu_ext_over_nu_thr(experiment, g_nmdas=[0, 0.5e-07, 1e-07, 1.5e-07, 2e-07, 5e-07],
                                             nu_ext_over_nu_thrs=[1.871, 1.872, 1.8723])

    def test_grid_results_try_to_avoid_bifurcation(self):
        experiment = Experiment({
            NetworkParams.KEY_NU_E_OVER_NU_THR: 0,
            NetworkParams.KEY_EPSILON: 0.1,
            "g": 4,
            "g_ampa": 0.5e-06,
            "g_gaba": 0.5e-06,
            "g_nmda": 0,
            Experiment.KEY_SELECTED_MODEL: NMDAModels.model_with_detailed_hidden_variables.eq,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda", "sigmoid_v",
                                                        "g_nmda", "I_nmda"],
            "record_N": 10,

            #Experiment.KEY_SIMULATION_CLOCK: 0.005,
            "t_range": [[300, 500], [0, 2000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.HIDDEN_VARIABLES]
        })
        compare_g_nmda_vs_nu_ext_over_nu_thr(experiment, g_nmdas=[0, 0.4e-5, 0.45e-5, 0.5e-5], nu_ext_over_nu_thrs=[10.6, 10.8, 10.9])
        compare_g_nmda_vs_nu_ext_over_nu_thr(experiment, g_nmdas=[0.43e-5, 0.4325e-5, 0.435e-5, 0.4375e-5, 0.44e-5], nu_ext_over_nu_thrs=[10.6, 10.8, 10.9], nmda_to_inh_neurons=True)

    def test_sim_and_plot_with_nmda_connection_to_inh_works_scan_param_space(self):
        experiment = Experiment({
            NetworkParams.KEY_NU_E_OVER_NU_THR: 12,
            NetworkParams.KEY_EPSILON: 0.1,
            "g": 4,
            "g_ampa": 0.5e-06,
            "g_gaba": 0.5e-06,
            "g_nmda": 0.5e-5,
            Experiment.KEY_SELECTED_MODEL: NMDAModels.model_with_detailed_hidden_variables.eq,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda", "sigmoid_v",
                                                        "g_nmda", "I_nmda"],
            "record_N": 10,
            "t_range": [[0, 2000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.HIDDEN_VARIABLES]
        })
        sim_with_full_nmda_connections_and_plot(experiment, in_testing=True)

    def test_one_element_from_grid(self):
        experiment = Experiment({
            NetworkParams.KEY_NU_E_OVER_NU_THR: 10.9,
            NetworkParams.KEY_EPSILON: 0.1,
            "g": 4,
            "g_ampa": 0.5e-06,
            "g_gaba": 0.5e-06,
            "g_nmda": 0.4e-5,
            Experiment.KEY_SELECTED_MODEL: NMDAModels.model_with_detailed_hidden_variables.eq,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda", "sigmoid_v",
                                                        "g_nmda", "I_nmda"],
            "record_N": 10,
            "t_range": [[0, 2000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE]
        })
        sim_with_full_nmda_connections_and_plot(experiment, in_testing=True)




if __name__ == '__main__':
    unittest.main()
