import unittest

from brian2 import *

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams
from iteration_7_one_compartment_step_input.grid_computations import \
    sim_and_plot_experiment_grid_with_nmda_cut_off_in_down_state, \
    sim_and_plot_experiment_grid_with_increasing_nmda_input
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    single_compartment_with_nmda_and_logged_variables, sim_and_plot
from iteration_7_one_compartment_step_input.second_scripts import \
    show_up_down_states_with_different_nmda_rates_up_vs_down_example_1

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True


class OneCompartmentUpDownStates(unittest.TestCase):

    def test_up_down_state_can_be_simulated_and_plotted(self):
        config = {

            Experiment.KEY_IN_TESTING: True,
            Experiment.KEY_SIMULATION_METHOD: "euler",
            "panel": "Exemplifying up and down states without NMDA input",

            Experiment.KEY_SIMULATION_CLOCK: 0.5,

            "g": 1.25,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,
            "g_nmda": 0,

            "up_state": {
                "N_E": 1000,
                "gamma": 1,
                "nu": 100,

                "N_NMDA": 10,
                "nu_nmda": 10
            },
            "down_state": {
                "N_E": 100,
                "gamma": 3,
                "nu": 10,

                "N_NMDA": 10,
                "nu_nmda": 10
            },

            Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_and_logged_variables,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],
            Experiment.KEY_CURRENTS_TO_RECORD: ["I_L", "I_ampa", "I_gaba", "I_nmda"],

            "t_range": [[0, 20], [0, 25]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.CURRENTS,
                                                PlotParams.AvailablePlots.HIDDEN_VARIABLES]
        }
        sim_and_plot(Experiment(config))

    def test_up_down_state_with_nmda(self):
        config = {

            Experiment.KEY_IN_TESTING: True,
            Experiment.KEY_SIMULATION_METHOD: "euler",
            "panel": "Exemplifying up and down states without NMDA input",

            Experiment.KEY_SIMULATION_CLOCK: 0.5,

            "g": 1,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,
            "g_nmda": 2e-05,

            "up_state": {
                "N_E": 1000,
                "gamma": 1.2,
                "nu": 100,
            },
            "down_state": {
                "N_E": 100,
                "gamma": 3,
                "nu": 10,
            },

            Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_and_logged_variables,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],

            "t_range": [[0, 2000], [0, 250]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE]
        }
        sim_and_plot(Experiment(config))

    def test_grid_up_down_state_without_nmda(self):
        config = {

            Experiment.KEY_IN_TESTING: True,
            Experiment.KEY_SIMULATION_METHOD: "euler",
            "panel": "Exemplifying up and down states without NMDA input",

            Experiment.KEY_SIMULATION_CLOCK: 0.5,

            "g": 1,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,
            "g_nmda": 0,

            "up_state": {
                "N_E": 1000,
                "gamma": 1.2,
                "nu": 100,
            },
            "down_state": {
                "N_E": 100,
                "gamma": 4,
                "nu": 10,
            },

            PlotParams.KEY_PLOT_SMOOTH_WIDTH: 10,
            Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_and_logged_variables,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],

            Experiment.KEY_CURRENTS_TO_RECORD: ["I_L", "I_nmda", "I_fast"],

            "t_range": [[0, 250]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.CURRENTS]
        }

        experiment_0 = Experiment(config)
        experiment_1 = experiment_0.with_property("g_nmda", 2e-05)
        experiment_2 = experiment_1.with_property("g_nmda", 4e-5)
        experiment_3 = experiment_1.with_property("g_nmda", 5e-5)

        sim_and_plot_experiment_grid_with_nmda_cut_off_in_down_state(
            [experiment_0, experiment_1, experiment_2, experiment_3], title=self._testMethodName)

    def test_grid_up_down_state_with_constant_nmda_input_and_self_relaxing_down_state(self):

        for rate_nmda in [4, 6, 8, 10]:
            config = {

                Experiment.KEY_IN_TESTING: True,
                Experiment.KEY_SIMULATION_METHOD: "euler",
                "panel": "NMDA input with constant NMDA rate",

                Experiment.KEY_SIMULATION_CLOCK: 0.05,

                "g": 1,
                "g_ampa": 2.4e-06,
                "g_gaba": 2.4e-06,
                "g_nmda": 2e-05,

                "up_state": {
                    "N_E": 1000,
                    "gamma": 1.2,
                    "nu": 100,

                    "N_nmda": 10,
                    "nu_nmda": rate_nmda,
                },
                "down_state": {
                    "N_E": 0,
                    "gamma": 4,
                    "nu": 0,

                    "N_nmda": 10,
                    "nu_nmda": rate_nmda,
                },

                PlotParams.KEY_PLOT_SMOOTH_WIDTH: 10,
                Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_and_logged_variables,
                Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],

                Experiment.KEY_CURRENTS_TO_RECORD: ["I_L", "I_nmda", "I_fast"],

                "t_range": [[0, 3000]],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                    PlotParams.AvailablePlots.CURRENTS]
            }

            sim_and_plot_experiment_grid_with_increasing_nmda_input(Experiment(config),
                                                                    title="Up Down states with contant NDMA input. Can the Mg2+ block shut off NMDA input?",
                                                                    nmda_schedule=[0, 2e-05, 3e-5, 4e-5])

    def test_grid_up_down_state_with_constant_nmda_input(self):
        config = {

            Experiment.KEY_IN_TESTING: True,
            Experiment.KEY_SIMULATION_METHOD: "euler",
            "panel": "NMDA input with constant NMDA rate",

            Experiment.KEY_SIMULATION_CLOCK: 0.05,

            "g": 1,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,
            "g_nmda": 2e-05,

            "up_state": {
                "N_E": 1000,
                "gamma": 1.2,
                "nu": 100,

                "N_nmda": 10,
                "nu_nmda": 10,
            },
            "down_state": {
                "N_E": 100,
                "gamma": 4,
                "nu": 10,

                "N_nmda": 10,
                "nu_nmda": 10,
            },

            PlotParams.KEY_PLOT_SMOOTH_WIDTH: 10,
            Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_and_logged_variables,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],

            Experiment.KEY_CURRENTS_TO_RECORD: ["I_L", "I_nmda", "I_fast"],

            "t_range": [[0, 3000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.CURRENTS]
        }

        sim_and_plot_experiment_grid_with_increasing_nmda_input(Experiment(config),
                                                                title="Up Down states with contant NDMA input. Can the Mg2+ block shut off NMDA input?",
                                                                nmda_schedule=[0, 2e-05, 4e-5, 5e-5])

    def test_grid_up_down_state_with_nmda_N_depending_on_balanced_rate(self):
        config = {

            Experiment.KEY_IN_TESTING: True,
            Experiment.KEY_SIMULATION_METHOD: "euler",
            "panel": "NMDA input with NMDA rate depending on Up vs Down state",

            Experiment.KEY_SIMULATION_CLOCK: 0.5,

            "g": 1,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,
            "g_nmda": 2e-05,

            "up_state": {
                "N_E": 1000,
                "gamma": 1.2,
                "nu": 100,

                "N_nmda": 10,
                "nu_nmda": 10,
            },
            "down_state": {
                "N_E": 100,
                "gamma": 4,
                "nu": 10,

                "N_nmda": 2,
                "nu_nmda": 10,
            },

            PlotParams.KEY_PLOT_SMOOTH_WIDTH: 10,
            Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_and_logged_variables,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],

            Experiment.KEY_CURRENTS_TO_RECORD: ["I_L", "I_nmda", "I_fast"],

            "t_range": [[0, 3000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.CURRENTS]
        }

        sim_and_plot_experiment_grid_with_increasing_nmda_input(Experiment(config),
                                                                title="Up Down states with different N NMDA in Up and Down states",
                                                                nmda_schedule=[0, 2e-05, 4e-5, 5e-5])

    def test_grid_up_down_state_with_nmda_rate_depending_on_balanced_rate(self):
        config = {

            Experiment.KEY_IN_TESTING: True,
            Experiment.KEY_SIMULATION_METHOD: "euler",
            "panel": "NMDA input with NMDA rate depending on Up vs Down state",

            Experiment.KEY_SIMULATION_CLOCK: 0.5,

            "g": 1,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,
            "g_nmda": 2e-05,

            "up_state": {
                "N_E": 1000,
                "gamma": 1.2,
                "nu": 100,

                "N_nmda": 10,
                "nu_nmda": 10,
            },
            "down_state": {
                "N_E": 100,
                "gamma": 4,
                "nu": 10,

                "N_nmda": 10,
                "nu_nmda": 2,
            },

            PlotParams.KEY_PLOT_SMOOTH_WIDTH: 10,
            Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_and_logged_variables,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],

            Experiment.KEY_CURRENTS_TO_RECORD: ["I_L", "I_nmda", "I_fast"],

            "t_range": [[0, 3000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.CURRENTS]
        }

        sim_and_plot_experiment_grid_with_increasing_nmda_input(Experiment(config),
                                                                title="Up Down states with different NMDA rates in Up and Down states",
                                                                nmda_schedule=[0, 2e-05, 4e-5, 5e-5])

    def test_script_2(self):
        show_up_down_states_with_different_nmda_rates_up_vs_down_example_1()


if __name__ == '__main__':
    unittest.main()
