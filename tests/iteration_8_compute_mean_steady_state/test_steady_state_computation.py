import unittest

import matplotlib.pyplot as plt
from brian2 import ms, second, run, Equations, StateMonitor, defaultclock, NeuronGroup, mmole, kHz, mvolt, cm, msiemens, \
    mV, siemens

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams
from iteration_7_one_compartment_step_input.grid_computations import \
    sim_and_plot_experiment_grid_with_increasing_nmda_input
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    single_compartment_with_nmda_and_logged_variables
from iteration_8_compute_mean_steady_state.grid_computations import \
    sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state

from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_and_plot_up_down, \
    SteadyStateResults, sim_and_plot_down_up

config = {

    Experiment.KEY_IN_TESTING: True,
    Experiment.KEY_SIMULATION_METHOD: "euler",
    "panel": "Exemplifying up and down states without NMDA input",

    Experiment.KEY_SIMULATION_CLOCK: 0.5,
    Experiment.KEY_SIM_TIME: 200,

    "g": 1.25,
    "g_ampa": 2.4e-06,
    "g_gaba": 2.4e-06,
    "g_nmda": 2e-05,

    "up_state": {
        "N_E": 1000,
        "gamma": 1,
        "nu": 100,

        "N_nmda": 10,
        "nu_nmda": 10
    },
    "down_state": {
        "N_E": 100,
        "gamma": 3,
        "nu": 10,

        "N_nmda": 10,
        "nu_nmda": 10
    }
}


steady_model = """
dv/dt = 1/C * (- I_L - I_ampa - I_gaba - I_nmda): volt
I_L = g_L * (v-E_leak): amp / meter ** 2
I_ampa = g_e * (v - E_ampa): amp / meter ** 2
I_gaba = g_i * (v - E_gaba): amp / meter ** 2
I_nmda = g_nmda * (v - E_nmda): amp / meter** 2

dg_e/dt = -g_e / tau_ampa + g_ampa * N_E * r_e : siemens / meter**2
dg_i/dt = -g_i / tau_gaba + g_gaba * N_I * r_i : siemens / meter**2

g_nmda = g_nmda_max * sigmoid_v * s_nmda: siemens / meter**2
ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha * x_nmda * (1 - s_nmda) : 1
dx_nmda/dt = - x_nmda / tau_nmda_rise + 1 * N_N * r_nmda: 1

sigmoid_v = 1/(1 + exp(-0.062 * (v/mvolt)) * (MG_C/mmole / 3.57)): 1
"""


class MyTestCase(unittest.TestCase):
    def test_something(self):
        experiment = Experiment(config)

        defaultclock.dt = 0.01 * ms

        C = experiment.neuron_params.C

        theta = experiment.neuron_params.theta
        g_L = experiment.neuron_params.g_L
        E_leak = experiment.neuron_params.E_leak
        V_r = experiment.neuron_params.V_r

        g_ampa = experiment.synaptic_params.g_ampa
        g_gaba = experiment.synaptic_params.g_gaba
        g_nmda_max = experiment.synaptic_params.g_nmda

        E_ampa = experiment.synaptic_params.e_ampa
        E_gaba = experiment.synaptic_params.e_gaba
        E_nmda = experiment.synaptic_params.e_ampa

        MG_C = 1 * mmole  # extracellular magnesium concentration

        tau_ampa = experiment.synaptic_params.tau_ampa
        tau_gaba = experiment.synaptic_params.tau_gaba
        tau_nmda_decay = 100 * ms
        tau_nmda_rise = 2 * ms

        r_e = experiment.network_params.up_state.nu
        r_i = experiment.network_params.up_state.nu
        N_E = experiment.network_params.up_state.N_E
        N_I = experiment.network_params.up_state.N_I
        N_N = experiment.network_params.up_state.N_NMDA
        r_nmda = experiment.network_params.up_state.nu_nmda

        alpha = 0.5 * kHz  # saturation of NMDA channels at high presynaptic firing rates

        eqs = Equations(steady_model)

        neurons = NeuronGroup(1,
                              model=eqs,
                              method="euler")
        neurons.v[:] = -65 * mV

        v_monitor = StateMonitor(source=neurons,
                                 variables=["v", "g_e", "g_i", "g_nmda", "x_nmda", "s_nmda", "I_nmda"], record=True, dt=0.01 * ms)
        run(100 * ms, report="text", report_period=1 * second)

        fig, ax = plt.subplots(5, 1, sharex=True)

        ax[0].plot(v_monitor.t / ms, v_monitor[0].v / mV)
        ax[1].plot(v_monitor.t / ms, v_monitor[0].g_e / siemens * cm**2)
        ax[1].plot(v_monitor.t / ms, v_monitor[0].g_i  / siemens * cm**2)
        ax[1].plot(v_monitor.t / ms, v_monitor[0].g_nmda  / siemens * cm**2)
        ax[1].legend()

        ax[2].plot(v_monitor.t / ms, v_monitor[0].x_nmda)
        ax[3].plot(v_monitor.t / ms, v_monitor[0].s_nmda)
        ax[4].plot(v_monitor.t / ms, v_monitor[0].I_nmda)


        ax[0].set_title("Voltages")
        ax[0].set_ylabel("$V_m$ [mV]")
        ax[1].set_title("Conductances")
        ax[2].set_title("X")
        ax[3].set_title("S")
        ax[4].set_title("I NMDA")

        plt.show()
        results = SteadyStateResults(v_monitor)
        print(f"Steady value: {results}")

    def test_run_and_plot_one_nmda_simulation_with_steady_state_constant_nmda_input(self):
        config = {

            Experiment.KEY_IN_TESTING: True,
            Experiment.KEY_SIMULATION_METHOD: "euler",
            "panel": "NMDA input with constant NMDA rate",

            Experiment.KEY_SIMULATION_CLOCK: 0.05,

            "g": 1,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,
            "g_nmda": 4e-5,

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
            Experiment.KEY_STEADY_MODEL: steady_model,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],

            Experiment.KEY_CURRENTS_TO_RECORD: ["I_L", "I_nmda", "I_fast"],

            "t_range": [[0, 4000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.CURRENTS]
        }

        sim_and_plot_up_down(Experiment(config))
        sim_and_plot_down_up(Experiment(config))

    def test_run_and_plot_one_nmda_simulation_with_different_nmda_rates_up_down(self):
        config = {

            Experiment.KEY_IN_TESTING: True,
            Experiment.KEY_SIMULATION_METHOD: "euler",
            "panel": "NMDA input with constant NMDA rate",

            Experiment.KEY_SIMULATION_CLOCK: 0.05,

            "g": 1,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,
            "g_nmda": 4e-5,

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
            Experiment.KEY_STEADY_MODEL: steady_model,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],

            Experiment.KEY_CURRENTS_TO_RECORD: ["I_L", "I_nmda", "I_fast"],

            "t_range": [[0, 4000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.CURRENTS]
        }
        config_without_nmda_in_down_state = {

            Experiment.KEY_IN_TESTING: True,
            Experiment.KEY_SIMULATION_METHOD: "euler",
            "panel": "Down state without NMDA input",

            Experiment.KEY_SIMULATION_CLOCK: 0.05,

            "g": 1,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,
            "g_nmda": 4e-5,

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
                "nu_nmda": 0,
            },

            PlotParams.KEY_PLOT_SMOOTH_WIDTH: 10,
            Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_and_logged_variables,
            Experiment.KEY_STEADY_MODEL: steady_model,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],

            Experiment.KEY_CURRENTS_TO_RECORD: ["I_L", "I_nmda", "I_fast"],

            "t_range": [[0, 4000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE]
        }

        sim_and_plot_up_down(Experiment(config))
        sim_and_plot_down_up(Experiment(config))


        sim_and_plot_up_down(Experiment(config_without_nmda_in_down_state))
        sim_and_plot_down_up(Experiment(config_without_nmda_in_down_state))


    def test_grid_up_down_state_with_constant_nmda_input_and_steady(self):
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
            Experiment.KEY_STEADY_MODEL: steady_model,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],

            Experiment.KEY_CURRENTS_TO_RECORD: ["I_L", "I_nmda", "I_fast"],

            "t_range": [[0, 3000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.CURRENTS]
        }

        sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state(Experiment(config),
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

        sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state(Experiment(config),
                                                                title="Up Down states with different N NMDA in Up and Down states",
                                                                nmda_schedule=[0, 2e-05, 4e-5, 5e-5])


if __name__ == '__main__':
    unittest.main()
