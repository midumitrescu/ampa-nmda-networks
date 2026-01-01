import unittest

import numpy as np

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams, \
    NeuronModelParams, SynapticParams
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    single_compartment_with_nmda_and_logged_variables
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

g_nmda = g_nmda_max * sigmoid_v * s_nmda: siemens
ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha * x_nmda * (1 - s_nmda) : 1
dx_nmda/dt = - x_nmda / tau_nmda_rise + 1 * N_N * r_nmda: 1

sigmoid_v = 1/(1 + exp(-0.062 * (v/mvolt)) * (MG_C/mmole / 3.57)): 1
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
    PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                        PlotParams.AvailablePlots.CURRENTS]
}


class MyTestCase(unittest.TestCase):

    def test_wang_configuration_is_correctly_applied(self):
        object_under_test = Experiment(wang_recurrent_config)
        self.assertEqual(2000, object_under_test.network_params.up_state.N)
        self.assertEqual(1600, object_under_test.network_params.up_state.N_E)
        self.assertEqual(400, object_under_test.network_params.up_state.N_I)

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


if __name__ == '__main__':
    unittest.main()
