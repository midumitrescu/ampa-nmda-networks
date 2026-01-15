
from brian2 import *

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams
from iteration_7_one_compartment_step_input.grid_computations import \
    sim_and_plot_experiment_grid_with_nmda_cut_off_in_down_state, \
    sim_and_plot_experiment_grid_with_increasing_nmda_input
from iteration_7_one_compartment_step_input.models_and_configs import single_compartment_with_nmda_and_logged_variables
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import sim_and_plot

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True


def show_up_down_states_with_different_nmda_rates_up_vs_down_example_1():
    return __show_up_down_states_with_different_nmda_rates_up_vs_down(up_state_nmda_rate=10, down_state_nmda_rate=2)

def __show_up_down_states_with_different_nmda_rates_up_vs_down(up_state_nmda_rate, down_state_nmda_rate):
    config = {

        Experiment.KEY_IN_TESTING: True,
        Experiment.KEY_SIMULATION_METHOD: "euler",
        "panel": "NMDA input with NMDA rate depending on Up vs Down state",

        Experiment.KEY_SIMULATION_CLOCK: 0.005,

        "g": 1,
        "g_ampa": 2.4e-06,
        "g_gaba": 2.4e-06,
        "g_nmda": 2e-05,

        "up_state": {
            "N_E": 1000,
            "gamma": 1.2,
            "nu": 100,

            "N_nmda": 10,
            "nu_nmda": up_state_nmda_rate,
        },
        "down_state": {
            "N_E": 100,
            "gamma": 4,
            "nu": 10,

            "N_nmda": 10,
            "nu_nmda": down_state_nmda_rate,
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