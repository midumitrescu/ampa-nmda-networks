import unittest

from brian2 import *

from Configuration import Experiment, NetworkParams, PlotParams
from iteration_7_one_compartment_step_input.one_compartment_under_step_input import single_compartment_with_nmda, sim, \
    single_compartment_with_nmda_and_logged_variables, sim_and_plot

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True


class OneCompartmentWithStepInputTestCases(unittest.TestCase):

    def test_nmda_models_runs(self):
        for model in [single_compartment_with_nmda, single_compartment_with_nmda_and_logged_variables]:
            config = {

                NetworkParams.KEY_N_E: 1,
                NetworkParams.KEY_C_EXT: 100,
                NetworkParams.KEY_NU_E_OVER_NU_THR: 5 * 1e-3,
                NetworkParams.KEY_EPSILON: 0.,
                "g_nmda": 5e-07,

                Experiment.KEY_SELECTED_MODEL: model,
                Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda"],
                "record_N": 10,

                "t_range": [[0, 20]],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                    PlotParams.AvailablePlots.HIDDEN_VARIABLES]
            }

            sim(Experiment(config))

    def test_one_compartment_model_with_step_input_can_be_plotted(self):

        for N_ext in [0, 10, 100, 1000, 10_000]:
            config = {

                NetworkParams.KEY_N_E: 1,
                NetworkParams.KEY_C_EXT: N_ext,
                NetworkParams.KEY_NU_E_OVER_NU_THR: 5 * 1e-2,

                "g": 4,
                "g_ampa": 2.4e-05,
                "g_gaba": 2.4e-05,
                "g_nmda": 5e-06,

                Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda,
                Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda"],

                "t_range": [[0, 2_000], [1250, 1260]],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE]
            }

            sim_and_plot(Experiment(config))

    def test_show_euler_instability(self):

        config = {

            Experiment.KEY_IN_TESTING: True,
            Experiment.KEY_SIMULATION_CLOCK: 0.01,  # this is the default of Brian2

            NetworkParams.KEY_N_E: 1,
            NetworkParams.KEY_C_EXT: 10_000,
            NetworkParams.KEY_NU_E_OVER_NU_THR: 5 * 1e-2,

            "g": 4,
            "g_ampa": 2.4e-05,
            "g_gaba": 2.4e-05,
            "g_nmda": 5e-06,
            "method": "euler",

            Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda"],

            "t_range": [[0, 250], [245, 246]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE]
        }
        sim_and_plot(Experiment(config))

    def test_increase_N(self):

        for N_ext in np.linspace(1_000, 10_000, num=10):
            config = {

                Experiment.KEY_IN_TESTING: True,
                Experiment.KEY_SIMULATION_METHOD: "rk4",

                NetworkParams.KEY_N_E: 1,
                NetworkParams.KEY_C_EXT: N_ext,
                NetworkParams.KEY_NU_E_OVER_NU_THR: 5 * 1e-2,

                "g": 1.5,
                "g_ampa": 2.4e-05,
                "g_gaba": 2.4e-05,
                "g_nmda": 0,

                Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda,
                Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda"],

                "t_range": [[0, 2000]],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE]

            }

            sim_and_plot(Experiment(config))

    def test_understand_numerical_instability_in_v(self):

        for g in np.linspace(1.5, 1, 11):
            config = {

                Experiment.KEY_IN_TESTING: True,
                Experiment.KEY_SIMULATION_METHOD: "euler",

                Experiment.KEY_SIMULATION_CLOCK: 0.005,

                NetworkParams.KEY_N_E: 1,
                NetworkParams.KEY_C_EXT: 1000,
                NetworkParams.KEY_NU_E_OVER_NU_THR: 5 * 1e-2,

                "g": g,
                "g_ampa": 2.4e-06,
                "g_gaba": 2.4e-06,
                "g_nmda": 0,

                Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_and_logged_variables,
                Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],

                "t_range": [[0, 2000], [0, 250]],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE]
            }
            sim_and_plot(Experiment(config))


if __name__ == '__main__':
    unittest.main()
