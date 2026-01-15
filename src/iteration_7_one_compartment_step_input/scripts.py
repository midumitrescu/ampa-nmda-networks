from brian2 import *

from Configuration import Experiment, NetworkParams, PlotParams
from iteration_7_one_compartment_step_input.one_compartment_under_step_input import \
    single_compartment_with_nmda_and_logged_variables, sim_and_plot

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True


def show_initial_up_state_implementation():
    for g in np.linspace(1.5, 1, 11):
        config = {

            Experiment.KEY_IN_TESTING: False,
            Experiment.KEY_SIMULATION_METHOD: "euler",

            Experiment.KEY_SIMULATION_CLOCK: 0.001,

            NetworkParams.KEY_N_E: 1,
            NetworkParams.KEY_C_EXT: 1000,
            NetworkParams.KEY_NU_E_OVER_NU_THR: 5 * 1e-2,

            "g": g,
            "g_ampa": 2.4e-05,
            "g_gaba": 2.4e-05,
            "g_nmda": 0,

            Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_and_logged_variables,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],

            "t_range": [[0, 3000], [0, 250]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE]
        }
        sim_and_plot(Experiment(config))


'''
A small modelling issue with the NMDA input with a nonlinearity that depends on the voltage (the known sigma(v) ... )

It is not effective in "blocking" "access" to NMDA channels for the rest voltage.

This experiment/graph should show exactly this issue
'''