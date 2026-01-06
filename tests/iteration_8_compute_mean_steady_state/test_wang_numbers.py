import unittest

import numpy as np

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams
from iteration_8_compute_mean_steady_state.grid_computations import \
    sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_and_plot_up_down
from scripts.iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import wang_recurrent_config

wang_recurrent_config = Experiment(wang_recurrent_config).with_property("t_range",  [[0, 100]]).params

class WangNumberTestCases(unittest.TestCase):

    def test_wang_configuration_is_correctly_applied(self):
        object_under_test = Experiment(wang_recurrent_config)
        self.assertEqual(2000, object_under_test.network_params.up_state.N)
        self.assertEqual(1600, object_under_test.network_params.up_state.N_E)
        self.assertEqual(400, object_under_test.network_params.up_state.N_I)

    def test_up_down_with_wang_numbers(self):
        sim_and_plot_up_down(Experiment(wang_recurrent_config))

    '''
    Model starts firing between 80Hz input -> 0 Hz output, 90 Hz -> 20-40 Hz
    '''
    def test_only_up_with_wang_numbers(self):
        various_nu_ext = np.arange(0, 100, step=50)
        for nu_ext in various_nu_ext:
            same_state = {
                "N": 2000,
                "nu": nu_ext,
                "N_nmda": 10,
                "nu_nmda": 10,
            }
            experiment = Experiment(wang_recurrent_config).with_property("up_state", same_state).with_property(
                "down_state", same_state)
            sim_and_plot_up_down(experiment)

    def test_grid_increasing_N_E_vs_increasing_g_nmda(self):
        increasing_nmda = np.array([0.5e-9, 1e-9, 1.5e-9, 3e-9])
        increasing_N_E = [1300]
        for n_e in increasing_N_E:
            current_experiment = (Experiment(wang_recurrent_config).with_property(PlotParams.KEY_WHAT_PLOTS_TO_SHOW,
                                                                                  [PlotParams.AvailablePlots.RASTER_AND_RATE])
                                  .with_property("up_state", {
                "N_E": n_e,
                "N_I": 1000,
                "nu": 100,

                "N_nmda": 10,
                "nu_nmda": 10,
            }))
            sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state(current_experiment,
                                                                                     "Look for Palmer firing rates",
                                                                                     increasing_nmda)


if __name__ == '__main__':
    unittest.main()
