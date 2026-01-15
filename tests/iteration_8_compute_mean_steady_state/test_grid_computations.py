import unittest

from brian2 import siemens

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams
from iteration_8_compute_mean_steady_state.grid_computations import convert_to_experiment_matrix, \
    convert_to_experiment_list, sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state, \
    run_simulate_with_steady_state, parallelize_simulate_with_up_state_and_nmda
from scripts.iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import wang_recurrent_config

wang_experiment_short = Experiment(wang_recurrent_config).with_property("t_range", [[0, 100]])
wang_recurrent_config = wang_experiment_short.params

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
        sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state(wang_experiment_short,
                                                                                 "Testing Palmer",
                                                                                 [1E-10, 1E-9, 1.35E-9])

    def test_experiment_grid_generation_for_1_row_matrix_input(self):
        short_wang_config = Experiment(wang_recurrent_config).with_property("t_range", [[0, 100]])
        sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state(short_wang_config,
                                                                                 "Testing Palmer",
                                                                                 [[1E-10, 1E-9, 1.35E-9]])

    def test_experiment_grid_generation_for_2_rows_matrix_input(self):
        short_wang_config = Experiment(wang_recurrent_config).with_property("t_range", [[0, 100]])
        results = sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state(
            short_wang_config,
            "Testing Palmer",
            [
                [1E-10, 1E-9, 1.35E-9],
                [1.5E-9, 2E-9, 3E-9]
            ], show_individual_plots=False)

        self.assertEqual(1E-10, results[0, 0].experiment.synaptic_params.g_nmda / siemens)
        self.assertEqual(1E-9, results[0, 1].experiment.synaptic_params.g_nmda / siemens)
        self.assertEqual(1.35E-9, results[0, 2].experiment.synaptic_params.g_nmda / siemens)
        self.assertEqual(1.5E-9, results[1, 0].experiment.synaptic_params.g_nmda / siemens)
        self.assertEqual(2E-9, results[1, 1].experiment.synaptic_params.g_nmda / siemens)
        self.assertEqual(3E-9, results[1, 2].experiment.synaptic_params.g_nmda / siemens)

    def test_matrix_computation_works_with_all_plots(self):
        experiment_with_all_plots = (Experiment(wang_recurrent_config)
         .with_property("t_range", [[0, 100]])
         .with_property(PlotParams.KEY_WHAT_PLOTS_TO_SHOW, [PlotParams.AvailablePlots.RASTER_AND_RATE, PlotParams.AvailablePlots.CURRENTS, PlotParams.AvailablePlots.HIDDEN_VARIABLES]))

        sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state(
            experiment_with_all_plots,
            "Testing Palmer",
            [
                [1E-10, 1E-9, 1.35E-9],
                [1.5E-9, 2E-9, 3E-9]
            ], show_individual_plots=False)

    def test_paralelization_of_upstate_and_nmda_works(self):
        experiment = Experiment(wang_recurrent_config)
        t_range = [0, 1_000]
        results = parallelize_simulate_with_up_state_and_nmda(
            [experiment.with_property("t_range", t_range), experiment.with_property("t_range", t_range)])

        self.assertEqual(52, results[0].spikes.num_spikes)
        self.assertEqual(52, results[0].spikes.mean_rate)