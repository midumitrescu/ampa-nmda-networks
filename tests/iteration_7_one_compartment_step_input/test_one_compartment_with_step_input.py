import unittest

import brian2.devices.device
from brian2 import *
from matplotlib import gridspec

from Configuration import Experiment, NetworkParams, PlotParams
from Plotting import plot_non_blocking
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

    def test_understand_instabilities(self):

        for sim_clock in [0.01, 0.005, 0.001]:
            for N_ext in [10_000]:

                config = {

                    Experiment.KEY_SIMULATION_CLOCK: sim_clock,
                    Experiment.KEY_IN_TESTING: True,

                    NetworkParams.KEY_N_E: 1,
                    NetworkParams.KEY_C_EXT: N_ext,
                    NetworkParams.KEY_NU_E_OVER_NU_THR: 5 * 1e-2,

                    "g": 4,
                    "g_ampa": 2.4e-05,
                    "g_gaba": 2.4e-05,
                    "g_nmda": 5e-06,

                    Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda,
                    Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda"],

                    "t_range": [[0, 250], [245, 246]],
                    PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE]

                }

                sim_and_plot(Experiment(config))

                # 34s (haun) vs  1m 35s (euler)


if __name__ == '__main__':
    unittest.main()
