import unittest

from brian2 import *

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams
from iteration_7_one_compartment_step_input.grid_computations import sim_and_plot_experiment_grid
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    single_compartment_with_nmda_and_logged_variables, sim_and_plot

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True


class OneCompartmentUpDownStates(unittest.TestCase):

    def test_up_down_state(self):

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
            },
            "down_state": {
                "N_E": 100,
                "gamma": 3,
                "nu": 10,
            },

            Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_and_logged_variables,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],

            "t_range": [[0, 20], [0, 25]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE]
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

    def test_two_up_down_state_with_nmda(self):

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

            "t_range": [[0, 200], [0, 250]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE]
        }

        experiment_1 = Experiment(config)
        experiment_2 = experiment_1.with_property("g_nmda", 2e-4)
        experiment_3 = experiment_1.with_property("g_nmda", 2e-3)

        sim_and_plot_experiment_grid([experiment_1, experiment_2, experiment_3])




if __name__ == '__main__':
    unittest.main()
