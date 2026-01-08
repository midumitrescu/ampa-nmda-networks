import unittest

from brian2 import siemens

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_8_compute_mean_steady_state.grid_computations import convert_to_experiment_matrix, \
    convert_to_experiment_list, sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state
from scripts.iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import wang_recurrent_config

wang_recurrent_config = Experiment(wang_recurrent_config).with_property("t_range", [[0, 100]]).params

class GridComputationTestCases(unittest.TestCase):

    def test_experiment_generation_from_list(self):
        object_under_test = convert_to_experiment_list(Experiment(wang_recurrent_config), [0, 1, 2])
        self.assertEqual(0, object_under_test[0].synaptic_params.g_nmda / siemens)
        self.assertEqual(1, object_under_test[1].synaptic_params.g_nmda / siemens)
        self.assertEqual(2, object_under_test[2].synaptic_params.g_nmda / siemens)

    def test_experiment_generation_from_matrix_0(self):
        object_under_test = convert_to_experiment_matrix(Experiment(wang_recurrent_config), [0, 1, 2])
        self.assertEqual(0, object_under_test[0][0].synaptic_params.g_nmda / siemens)
        self.assertEqual(1, object_under_test[0][1].synaptic_params.g_nmda / siemens)
        self.assertEqual(2, object_under_test[0][2].synaptic_params.g_nmda / siemens)

    def test_experiment_generation_from_matrix_1(self):
        object_under_test = convert_to_experiment_matrix(Experiment(wang_recurrent_config), [[0, 1, 2]])
        self.assertEqual(0, object_under_test[0][0].synaptic_params.g_nmda / siemens)
        self.assertEqual(1, object_under_test[0][1].synaptic_params.g_nmda / siemens)
        self.assertEqual(2, object_under_test[0][2].synaptic_params.g_nmda / siemens)

    def test_experiment_grid_generation_for_list_input(self):
        short_wang_config = Experiment(wang_recurrent_config).with_property("t_range", [[0, 100]])
        sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state(short_wang_config,
                                                                                 "Testing Palmer",
                                                                                 [1E-10, 1E-9, 1.35E-9])

    def test_experiment_grid_generation_for_1_row_matrix_input(self):
        short_wang_config = Experiment(wang_recurrent_config).with_property("t_range", [[0, 100]])
        sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state(short_wang_config,
                                                                                 "Testing Palmer",
                                                                                 [[1E-10, 1E-9, 1.35E-9]])

    def test_experiment_grid_generation_for_2_rows_matrix_input(self):
        short_wang_config = Experiment(wang_recurrent_config).with_property("t_range", [[0, 100]])
        sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state(
            short_wang_config,
            "Testing Palmer",
            [
                [1E-10, 1E-9, 1.35E-9],
                [1.5E-9, 2E-9, 3E-9]
            ], show_individual_plots=False)
