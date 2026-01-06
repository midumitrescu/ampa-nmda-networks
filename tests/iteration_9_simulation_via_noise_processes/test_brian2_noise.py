import unittest

import numpy as np

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams, \
    NeuronModelParams, SynapticParams
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    single_compartment_with_nmda_and_logged_variables
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import steady_model
from iteration_9_simulation_via_noise_processes.grid_computations import \
    sim_and_plot_experiment_grid_with_increasing_nmda_input_and_diffusion_process, \
    sim_and_plot_experiment_grid_with_increasing_nmda_noise
from iteration_9_simulation_via_noise_processes.one_compartment_with_difusion_process import \
    sim_and_plot_diffusion_process

diffusion_model = """
dv/dt = 1/C * (- I_L - I_ampa - I_gaba - I_nmda): volt
I_L = g_L * (v-E_leak): amp
I_ampa = g_e * (v - E_ampa): amp
I_gaba = g_i * (v - E_gaba): amp
I_nmda = g_nmda * (v - E_nmda): amp

dg_e/dt = (-(g_e - g_e_0) + sigma_e * xi ) / tau_ampa: siemens
dg_i/dt = (-(g_i - g_i_0) + sigma_i * xi) / tau_gaba: siemens

g_nmda = g_nmda_max * sigmoid_v * s_nmda: siemens
dx_nmda/dt = (-(x_nmda - x_0))/ tau_nmda_rise: 1
ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha * x_nmda * (1 - s_nmda) : 1

sigmoid_v = 1/(1 + exp(-0.062 * (v/mvolt)) * (MG_C/mmole / 3.57)): 1
"""

diffusion_model_with_up_down = """
dv/dt = 1/C * (- I_L - I_ampa - I_gaba - I_nmda): volt
I_L = g_L * (v-E_leak): amp
I_ampa = g_e * (v - E_ampa): amp
I_gaba = g_i * (v - E_gaba): amp
I_nmda = g_nmda * (v - E_nmda): amp

dg_e/dt = (-(g_e - up*g_e_0_up - down*g_e_0_down) + up * sigma_e_up * xi_0 + down * sigma_e_down * xi_0) / tau_ampa: siemens
dg_i/dt = (-(g_i - up*g_i_0_up - down*g_i_0_down) + up * sigma_i_up * xi_1 + down * sigma_i_down * xi_1) / tau_gaba: siemens

g_nmda = g_nmda_max * sigmoid_v * s_nmda: siemens
dx_nmda/dt = (-(x_nmda - up*x_0_up - down*x_0_down) + up * sigma_x_up * xi_2 + down * sigma_x_down * xi_2)/ tau_nmda_rise: 1
ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha * x_nmda * (1 - s_nmda) : 1

sigmoid_v = 1/(1 + exp(-0.062 * (v/mvolt)) * (MG_C/mmole / 3.57)): 1
up: 1
down: 1
"""

difussion_experiment_with_wang_numbers = {

    Experiment.KEY_IN_TESTING: True,
    Experiment.KEY_SIMULATION_METHOD: "heun",

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
    SynapticParams.KEY_G_NMDA: 0.165e-7,

    # ␶where the decay time constant of GABA currents is taken to be tau GABA = 5 ms
    SynapticParams.KEY_TAU_AMPA: 2,
    SynapticParams.KEY_TAU_GABA: 5,

    SynapticParams.KEY_TAU_NMDA_RISE: 2,
    SynapticParams.KEY_TAU_NMDA_DECAY: 100,

    "up_state": {
        "N_E": 1600,
        "N_I": 400,
        "nu": 100,

        "N_nmda": 10,
        "nu_nmda": 10,
    },
    "down_state": {
        "N_E": 100,
        "N_I": 200,
        "nu": 10,

        "N_nmda": 10,
        "nu_nmda": 2,
    },

    PlotParams.KEY_PLOT_SMOOTH_WIDTH: 10,
    Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_and_logged_variables,
    Experiment.KEY_STEADY_MODEL: steady_model,
    Experiment.KEY_DIFFUSION_MODEL: diffusion_model_with_up_down,
    Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "sigmoid_v", "g_nmda", "I_nmda"],

    Experiment.KEY_CURRENTS_TO_RECORD: ["I_L", "I_nmda", "I_ampa", "I_gaba"],

    "t_range": [[0, 4000]],
    PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                        PlotParams.AvailablePlots.CURRENTS,
                                        PlotParams.AvailablePlots.HIDDEN_VARIABLES]
}


class DiffusionProcessTestCases(unittest.TestCase):

    def test_diffusion_simulation_works(self):
        sim_and_plot_diffusion_process(Experiment(difussion_experiment_with_wang_numbers))
        sim_and_plot_diffusion_process(
            Experiment(difussion_experiment_with_wang_numbers).with_property(NeuronModelParams.KEY_NEURON_THRESHOLD,
                                                                             -40))

    def test_grid_diffusion_process_works(self):
        increasing_nmda = np.array([0.5e-9, 1e-9, 1.5e-9, 3e-9])
        sim_and_plot_experiment_grid_with_increasing_nmda_input_and_diffusion_process(
            Experiment(difussion_experiment_with_wang_numbers),
            "Diffusion Process", nmda_schedule=increasing_nmda)

    def test_wang_diffusion_bellow_threshold(self):
        sim_and_plot_diffusion_process(Experiment(difussion_experiment_with_wang_numbers)
                                       .with_property("up_state",
                                                      {
                                                          "N_E": 1600,
                                                          "N_I": 400,
                                                          "nu": 75,

                                                          "N_nmda": 10,
                                                          "nu_nmda": 10,
                                                      }))

    def test_difussion_with_one_state(self):
        sim_and_plot_diffusion_process(Experiment(difussion_experiment_with_wang_numbers).with_property("panel", "Poisson Process Simulation"))

    def test_grid_plot_inceasing_nmda_noise(self):
        sim_and_plot_experiment_grid_with_increasing_nmda_noise(Experiment(difussion_experiment_with_wang_numbers),
        "Increasing Noise Intensity in Diffusion Process", nmda_noise_schedule=[0, 0.5, 1, 1.5, 2])

    def test_compare_poisson_process_vs_diffusion_process(self):
        sim_and_plot_diffusion_process(Experiment(difussion_experiment_with_wang_numbers).with_property("panel", "Diffusion Process Simulation").with_property("theta", -40))
        #sim_and_plot_with_weak_meanfield(Experiment(difussion_experiment_with_wang_numbers).with_property("panel", "Poisson Process Simulation").with_property("theta", -40))

    def test_single_state(self):
        sim_and_plot_diffusion_process(Experiment(difussion_experiment_with_wang_numbers).with_property("up_state", {
        "N_E": 100,
        "N_I": 200,
        "nu": 10,

        "N_nmda": 10,
        "nu_nmda": 2,
    }))


if __name__ == '__main__':
    unittest.main()
