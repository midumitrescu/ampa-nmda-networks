import unittest

from brian2 import msecond, siemens, cm

from iteration_9_simulation_via_noise_processes.one_compartment_with_python_native_simulation import simulate_native, \
    plot_simulation
import numpy as np

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams, \
    NeuronModelParams, SynapticParams
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    single_compartment_with_nmda_and_logged_variables
from iteration_8_compute_mean_steady_state.grid_computations import \
    sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_and_plot_with_weak_meanfield

wang_recurrent_config = {

    Experiment.KEY_IN_TESTING: True,
    Experiment.KEY_SIMULATION_METHOD: "euler",
    "panel": "NMDA Network with UP/Down and Wang numbers from recurrent connections (network generated input)",

    Experiment.KEY_SIMULATION_CLOCK: 0.05,

    NeuronModelParams.KEY_NEURON_C: 0.5e-3,
    # Wang: Cm = 0.5 nF for pyramidal cells. We are using microFahrad / cm^2 => we need the extra e-3.
    # check tau membrane to be 20 ms!
    "g": 1,
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
    SynapticParams.KEY_G_GABA: 1.3e-9,
    SynapticParams.KEY_G_NMDA: 0.165e-9,

    # ␶where the decay time constant of GABA currents is taken to be tau GABA = 5 ms
    SynapticParams.KEY_TAU_AMPA: 2,
    SynapticParams.KEY_TAU_GABA: 5,

    SynapticParams.KEY_TAU_NMDA_RISE: 2,
    SynapticParams.KEY_TAU_NMDA_DECAY: 100,

    "up_state": {
        "N_E": 1200,
        "N_I": 1000,
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
    Experiment.KEY_STEADY_MODEL: "native",
    Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],

    Experiment.KEY_CURRENTS_TO_RECORD: ["I_L", "I_nmda", "I_fast"],

    "t_range": [[0, 4000]],
    PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                        PlotParams.AvailablePlots.CURRENTS]
}


class MyTestCase(unittest.TestCase):


    def test_tau_ampa_relaxation_works_as_expected(self):
        experiment_config = Experiment(wang_recurrent_config).with_property(Experiment.KEY_SIMULATION_CLOCK, 0.005)
        self.assertEqual(20, experiment_config.neuron_params.tau / msecond)
        results = simulate_native(experiment_config)

        # tests for tau relaxation
        print("Look for g ampa at some very specific times w.r.t to tau ampa")
        dt = experiment_config.sim_clock / msecond
        tau_ampa = experiment_config.synaptic_params.tau_ampa / msecond

        mean_ampa = experiment_config.effective_time_constant_up_state.mean_excitatory_conductance() / (
                    siemens / cm ** 2)
        g_ampa_1_tau_ampa = results.g_e[int(tau_ampa / dt)]
        g_ampa_3_tau_ampa = results.g_e[int(3 * tau_ampa / dt)]
        g_ampa_4_tau_ampa = results.g_e[int(4 * tau_ampa / dt)]
        g_ampa_end = results.g_e[-1]
        print(
            f"At 1 tau ampa, we have: {g_ampa_1_tau_ampa : .6f} i.e. {g_ampa_1_tau_ampa / mean_ampa : .8f} from {mean_ampa : .8f} ")
        print(
            f"At 3 tau ampa, we have: {g_ampa_3_tau_ampa : .6f} i.e. {g_ampa_3_tau_ampa / mean_ampa : .8f} from {g_ampa_3_tau_ampa : .8f} ")
        print(
            f"At 4 tau ampa, we have: {g_ampa_4_tau_ampa : .6f} i.e. {g_ampa_4_tau_ampa / mean_ampa : .8f} from {mean_ampa : .8f} ")
        print(f"At end, we have: {g_ampa_end : .6f} i.e. {g_ampa_end / mean_ampa : .8f} from {mean_ampa : .8f} ")

        self.assertAlmostEqual(0.63258089, g_ampa_1_tau_ampa / mean_ampa)
        self.assertAlmostEqual(0.95039959, g_ampa_3_tau_ampa / mean_ampa)
        self.assertAlmostEqual(0.98177586, g_ampa_4_tau_ampa / mean_ampa)
        self.assertAlmostEqual(1, g_ampa_end / mean_ampa)

    def test_native_simulation(self):
        experiment_config = Experiment(wang_recurrent_config)

        self.assertEqual(20, experiment_config.neuron_params.tau / msecond)

        results = simulate_native(experiment_config)
        plot_simulation(results)

        # tests for tau relaxation
        print("Look for g ampa at some very specific times w.r.t to tau ampa")
        dt = experiment_config.sim_clock / msecond
        tau_ampa = experiment_config.synaptic_params.tau_ampa / msecond
        tau_gaba = experiment_config.synaptic_params.tau_gaba / msecond

        mean_ampa = experiment_config.effective_time_constant_up_state.mean_excitatory_conductance() / (siemens / cm ** 2)
        g_ampa_1_tau_ampa = results.g_e[int(tau_ampa / dt)]
        g_ampa_3_tau_ampa = results.g_e[int(3 * tau_ampa / dt)]
        g_ampa_4_tau_ampa = results.g_e[int(4 * tau_ampa / dt)]
        g_ampa_end = results.g_e[-1]
        print(
            f"At 1 tau ampa, we have: {g_ampa_1_tau_ampa : .6f} i.e. {g_ampa_1_tau_ampa / mean_ampa : .8f} from {mean_ampa : .8f} ")
        print(
            f"At 3 tau ampa, we have: {g_ampa_3_tau_ampa : .6f} i.e. {g_ampa_3_tau_ampa / mean_ampa : .8f} from {g_ampa_3_tau_ampa : .8f} ")
        print(
            f"At 4 tau ampa, we have: {g_ampa_4_tau_ampa : .6f} i.e. {g_ampa_4_tau_ampa / mean_ampa : .8f} from {mean_ampa : .8f} ")
        print(f"At end, we have: {g_ampa_end : .6f} i.e. {g_ampa_end / mean_ampa : .8f} from {mean_ampa : .8f} ")

        self.assertAlmostEqual(0.63258089, g_ampa_1_tau_ampa / mean_ampa)
        self.assertAlmostEqual(0.95039959, g_ampa_3_tau_ampa / mean_ampa)
        self.assertAlmostEqual(0.98177586, g_ampa_4_tau_ampa / mean_ampa)
        self.assertAlmostEqual(1, g_ampa_end / mean_ampa)


if __name__ == '__main__':
    unittest.main()
