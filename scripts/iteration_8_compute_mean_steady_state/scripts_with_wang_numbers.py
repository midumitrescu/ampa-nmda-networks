import unittest

import numpy as np
from brian2 import mV, mmole, second, ms, Hz

from BinarySeach import binary_search_for_target_value
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams, \
    NeuronModelParams, SynapticParams
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    single_compartment_with_nmda_and_logged_variables
from iteration_7_one_compartment_step_input.one_compartment_with_up_only import simulate_with_up_state_and_nmda, \
    sim_and_plot_up_with_state_and_nmda
from iteration_8_compute_mean_steady_state.grid_computations import \
    sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_and_plot_up_down

steady_model = """
dv/dt = 1/C * (- I_L - I_ampa - I_gaba - I_nmda): volt
I_L = g_L * (v-E_leak): amp
I_ampa = g_e * (v - E_ampa): amp
I_gaba = g_i * (v - E_gaba): amp
I_nmda = g_nmda * (v - E_nmda): amp

dg_e/dt = -g_e / tau_ampa + g_ampa * N_E * r_e : siemens
dg_i/dt = -g_i / tau_gaba + g_gaba * N_I * r_i : siemens

dx_nmda/dt = - x_nmda / tau_nmda_rise + g_x * N_N * r_nmda: 1
ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha * x_nmda * (1 - s_nmda) : 1
sigmoid_v = 1/(1 + (MG_C/mmole)/3.57 * exp(-0.062*(v/mvolt))): 1

g_nmda = g_nmda_max * sigmoid_v * s_nmda: siemens
"""
wang_recurrent_config = {

    Experiment.KEY_IN_TESTING: True,
    Experiment.KEY_SIMULATION_METHOD: "euler",
    "panel": "NMDA Network with UP/Down and Wang numbers from recurrent connections (network generated input)",

    Experiment.KEY_SIMULATION_CLOCK: 0.05,

    NeuronModelParams.KEY_NEURON_C: 0.5e-3,
    # Wang: Cm = 0.5 nF for pyramidal cells. We are using microFahrad / cm^2 => we need the extra e-3.
    # check tau membrane to be 20 ms!
    # ␶tau m = Cm/gL = 20 ms for excitatory cells

    # VL = -70 mV, the firing threshold Vth = - 50 mV, a reset potential Vreset = -55 mV
    NeuronModelParams.KEY_NEURON_E_L: -70,
    NeuronModelParams.KEY_NEURON_THRESHOLD: -50,
    NeuronModelParams.KEY_NEURON_V_R: -55,
    NeuronModelParams.KEY_NEURON_G_L: 25e-9,  # gL = 25 nS for pyramidal

    # Wang: I used the following values for the recurrent synaptic conductances (in nS)
    # I: we use overall in the configuration siemens / cm ** 2 => respect the scaling from Wang
    # for pyramidal cells: gext,AMPA = 2.1, g recurrent,AMPA = 0.05, grecurrent, NMDA = 0.165, and g recurrent, GABA = 1.3
    # NOTE: weirdly, this is how Wang compensates for the need of more inhibition than excitation in the balance
    SynapticParams.KEY_G_AMPA: 0.05e-9,
    SynapticParams.KEY_G_GABA: 0.04e-9,
    SynapticParams.KEY_G_NMDA: 0.165e-9,

    # ␶where the decay time constant of GABA currents is taken to be tau GABA = 5 ms
    SynapticParams.KEY_TAU_AMPA: 2,
    SynapticParams.KEY_TAU_GABA: 5,

    SynapticParams.KEY_TAU_NMDA_RISE: 2,
    SynapticParams.KEY_TAU_NMDA_DECAY: 100,

    "up_state": {
        "N": 2000,
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
    PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE, PlotParams.AvailablePlots.CURRENTS]
}

palmer_config_0_1_Hz_with_NMDA_block = Experiment(wang_recurrent_config).with_properties(
    {
        SynapticParams.KEY_G_NMDA: 0,
        "up_state": {
            "N": 2000,
            "nu": 82,
            "N_nmda": 0,
            "nu_nmda": 5,
        }})


def sigmoid_v(experiment, v):
    MG_C = experiment.synaptic_params.MG_C
    return 1 / (1 + (MG_C / mmole) / 3.57 * np.exp(-0.062 * (v / mV)))


def find_firing_rate_without_NMDA_with_N(experiment, N, sim_time=10 * second):
    up_state = experiment.params["up_state"]
    up_state["N"] = N

    experiment_with_nmda = experiment.with_properties({
        Experiment.KEY_SIM_TIME: sim_time / ms,
        "up_state": up_state})
    results_with_nmda = simulate_with_up_state_and_nmda(experiment_with_nmda)

    rate = results_with_nmda.total_spike_counts() / experiment_with_nmda.sim_time

    return rate / Hz


def find_firing_rate_without_NMDA_with_nu(experiment, nu, sim_time=10 * second):
    up_state = experiment.params["up_state"]
    up_state["nu"] = nu

    experiment_with_nmda = experiment.with_properties({
        Experiment.KEY_SIM_TIME: sim_time / ms,
        "up_state": up_state})
    results_with_nmda = simulate_with_up_state_and_nmda(experiment_with_nmda)

    rate = results_with_nmda.total_spike_counts() / experiment_with_nmda.sim_time

    return rate / Hz


def run_with_NMDA_and_obtain_firing_rate(experiment, g_nmda_max, sim_time=10 * second):
    experiment_with_nmda = experiment.with_properties({
        Experiment.KEY_SIM_TIME: sim_time / ms,
        SynapticParams.KEY_G_NMDA: g_nmda_max})
    results_with_nmda = simulate_with_up_state_and_nmda(experiment_with_nmda)

    rate = results_with_nmda.total_spike_counts() / experiment_with_nmda.sim_time

    return rate / Hz


class ScriptsNMDAWithWangNumbers(unittest.TestCase):

    def test_up_down_with_wang_numbers(self):
        sim_and_plot_up_down(Experiment(wang_recurrent_config))

    '''
    Model starts firing between 80Hz input -> 0 Hz output, 90 Hz -> 20-40 Hz
    '''

    def test_only_up_with_wang_numbers(self):
        various_nu_ext = np.arange(0, 100, step=10)
        for nu_ext in various_nu_ext:
            same_state = {
                "N": 2000,
                "nu": nu_ext,
                "N_nmda": 10,
                "nu_nmda": 10,
            }
            experiment = Experiment(wang_recurrent_config).with_property("up_state", same_state).with_property(
                "down_state", same_state)
            sim_and_plot_up_down(experiment)

    def test_grid_increasing_N_E_vs_increasing_g_nmda(self):
        increasing_nmda = np.array([0, 0.165e-9, 0.3e-9, 0.5e-9])
        increasing_nmda = np.array([0.5e-9, 1e-9, 1.5e-9, 3e-9])
        # increasing_N_E = [500, 1000, 1500, 2000] -> here, 1000 no firing, 1500 firing a lot
        # increasing_N_E = [1100, 1200, 1300, 1400] -> 1100, 1200 no firing. 1300 -> 0.05 Hz. 1400 Too much => 1300 is the right number of excitatory N
        increasing_N_E = [1300]
        for n_e in increasing_N_E:
            current_experiment = (Experiment(wang_recurrent_config).with_property(PlotParams.KEY_WHAT_PLOTS_TO_SHOW,
                                                                                  [PlotParams.AvailablePlots.RASTER_AND_RATE])
                                  .with_property("up_state", {
                "N_E": n_e,
                "N_I": 1000,
                "nu": 100,

                "N_nmda": 10,
                "nu_nmda": 10,
            }))
            sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state(current_experiment,
                                                                                     "Look for Palmer firing rates",
                                                                                     increasing_nmda)

    '''
    Looking for No NMDA -> 0.05 Hz and NMDA -> 0.3 Hz
    Search for these values using binary search
    '''

    def test_run_palmer_rates(self):
        palmer_experiment = (Experiment(wang_recurrent_config).with_property(PlotParams.KEY_WHAT_PLOTS_TO_SHOW,
                                                                             [PlotParams.AvailablePlots.RASTER_AND_RATE])
                             .with_property("up_state",
                                            {
                                                "N": 2000,
                                                "nu": 82,
                                                "N_nmda": 0,
                                                "nu_nmda": 10,
                                            })).with_property("t_range", [[0, 1_000]])
        # sim_and_plot_up_down(palmer_experiment)
        results_with_nmda = sim_and_plot_up_down(palmer_experiment.with_property("up_state",
                                                                                 {
                                                                                     "N": 2000,
                                                                                     "nu": 82,
                                                                                     "N_nmda": 10,
                                                                                     "nu_nmda": 5,
                                                                                 }))

        print(results_with_nmda)
        number_of_spikes = len(results_with_nmda.spikes['all_values']['t'][0]) / palmer_experiment.sim_time

    def test_search_for_nmda_palmer_rates(self):
        palmer_experiment = (Experiment(wang_recurrent_config).with_property("up_state",
                                                                             {
                                                                                 "N": 2000,
                                                                                 "nu": 82,
                                                                                 "N_nmda": 0,
                                                                                 "nu_nmda": 10,
                                                                             })).with_property("t_range", [[0, 1_000]])

        execute_palmer = lambda nmda_strength: run_with_NMDA_and_obtain_firing_rate(palmer_experiment, nmda_strength)
        res = binary_search_for_target_value(lower_value=0, upper_value=5E-8, func=execute_palmer, target_result=0.3)
        print(f"XXXXXXXXXXXXX {res}")
        # (1.5625e-10, 3.125e-10)


class ScriptsPalmerResultsWithoutNMDA(unittest.TestCase):

    # produces rate 0.05 Hz with up/down
    def test_example_1(self):
        palmer_experiment = (Experiment(wang_recurrent_config)
        .with_properties({
            "up_state":
                {
                    "N": 2000,
                    "nu": 82,
                    "N_nmda": 0,
                },
            "t_range": [[0, 10_000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW:
                [PlotParams.AvailablePlots.RASTER_AND_RATE]
        }))

        sim_and_plot_up_with_state_and_nmda(palmer_experiment)
        sim_and_plot_up_down(palmer_experiment)

    # produces rate 0.05 Hz with up/down
    def test_example_2(self):
        palmer_experiment = (Experiment(wang_recurrent_config).with_properties({
            "up_state":
                {
                    "N": 2000,
                    "nu": 81.54187093603468,
                    "N_nmda": 0,
                },
            "t_range": [[0, 10_000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW:
                [PlotParams.AvailablePlots.RASTER_AND_RATE]
        }))

        sim_and_plot_up_with_state_and_nmda(palmer_experiment)
        sim_and_plot_up_down(palmer_experiment)

    def test_search_for_nmda_rates_without_NMDA(self):
        palmer_experiment = (Experiment(wang_recurrent_config).with_property("up_state",
                                                                             {
                                                                                 "N": 2000,
                                                                                 "nu": 82,
                                                                                 "N_nmda": 0,
                                                                                 "nu_nmda": 0,
                                                                             })).with_property("t_range",
                                                                                               [[0, 1_000]])

        execute_palmer = lambda nu: find_firing_rate_without_NMDA_with_nu(palmer_experiment, nu)
        res = binary_search_for_target_value(lower_value=80, upper_value=100, func=execute_palmer,
                                             target_result=0.05)
        self.assertEqual(81.5418709360165, res[0])
        self.assertEqual(81.54187093603468, res[1])


class ScriptsPalmerResultsWithNMDA(unittest.TestCase):

    def test_example_2_produ(self):
        palmer_experiment = (Experiment(wang_recurrent_config).with_properties({
            SynapticParams.KEY_G_NMDA: 1.35E-9,
            "up_state":
                {
                    "N": 2000,
                    "nu": 82,
                    "N_nmda": 10,
                    "nu_nmda": 5,
                },
            "t_range": [[0, 10_000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW:
                [PlotParams.AvailablePlots.RASTER_AND_RATE]
        }))

        sim_and_plot_up_with_state_and_nmda(palmer_experiment)
        sim_and_plot_up_down(palmer_experiment)

    # (2.5e-10, 5e-10)
    def test_search_for_nmda_palmer_rates(self):
        palmer_experiment = (Experiment(wang_recurrent_config).with_property("up_state",
                                                                             {
                                                                                 "N": 2000,
                                                                                 "nu": 82,
                                                                                 "N_nmda": 10,
                                                                                 "nu_nmda": 5,
                                                                             }))

        execute_palmer = lambda nmda_strength: run_with_NMDA_and_obtain_firing_rate(palmer_experiment, nmda_strength)
        res = binary_search_for_target_value(lower_value=0, upper_value=1E-9, func=execute_palmer, target_result=0.3)
        print(f"XXXXXXXXXXXXX {res}")

    def test_run_grid(self):
        interesting_nmdas = [1E-10, 1E-9, 1.35E-9]
        sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state(palmer_config_0_1_Hz_with_NMDA_block,
                                                                                 "Palmer", interesting_nmdas)

    if __name__ == '__main__':
        unittest.main()
