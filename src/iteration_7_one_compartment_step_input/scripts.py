from brian2 import *

from Configuration import Experiment, NetworkParams, PlotParams
from iteration_7_one_compartment_step_input import Configuration_with_Up_Down_States, one_compartment_with_up_down
from iteration_7_one_compartment_step_input.grid_computations import sim_and_plot_experiment_grid_with_nmda_cut_off_in_down_state
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
def show_nmda_sigma_voltage_not_effective():
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
            "gamma": 4,
            "nu": 10,
        },

        PlotParams.KEY_PLOT_SMOOTH_WIDTH: 10,
        Experiment.KEY_SELECTED_MODEL: one_compartment_with_up_down.single_compartment_with_nmda_and_logged_variables,
        Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],

        Experiment.KEY_CURRENTS_TO_RECORD: ["I_L", "I_nmda", "I_fast"],
        "t_range": [[0, 3000]],
        PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                            PlotParams.AvailablePlots.CURRENTS]
    }

    experiment_1 = Configuration_with_Up_Down_States.Experiment(config)
    experiment_2 = experiment_1.with_property("g_nmda", 4e-5)
    experiment_3 = experiment_1.with_property("g_nmda", 5e-5)

    sim_and_plot_experiment_grid_with_nmda_cut_off_in_down_state([experiment_1, experiment_2, experiment_3], title="NMDA Magnesium Block not effective")

if __name__ == '__main__':
    #show_initial_up_state_implementation()
    show_nmda_sigma_voltage_not_effective()