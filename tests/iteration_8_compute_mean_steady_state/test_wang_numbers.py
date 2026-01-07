import unittest

import numpy as np
from brian2 import kHz, mV, nS

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams, \
    SynapticParams, State
from iteration_8_compute_mean_steady_state.grid_computations import \
    sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_and_plot_up_down, \
    sim_steady_state
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import sigmoid_v
from scripts.iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import wang_recurrent_config

wang_recurrent_config = Experiment(wang_recurrent_config).with_property("t_range",  [[0, 100]]).params

class WangNumberTestCases(unittest.TestCase):

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
        various_nu_ext = np.arange(0, 100, step=50)
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
        increasing_nmda = np.array([0.5e-9, 1e-9, 1.5e-9, 3e-9])
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

    def test_compare_wang_numbers_to_theoretical_steady_state(self):
        experiment = Experiment(wang_recurrent_config)
        g_x = experiment.synaptic_params.g_x_nmda
        tau_nmda_rise = experiment.synaptic_params.tau_nmda_rise
        tau_nmda_decay = experiment.synaptic_params.tau_nmda_decay

        alpha = 0.5 * kHz  # saturation of NMDA channels at high presynaptic firing rates
        N_NMDA = experiment.network_params.up_state.N_NMDA
        nu_NMDA = experiment.network_params.up_state.nu_nmda

        x_0 = alpha * tau_nmda_decay * tau_nmda_rise * g_x * N_NMDA * nu_NMDA
        s_0 = x_0 / (1 + x_0)
        self.assertAlmostEqual(0.2000000, tau_nmda_rise * g_x * N_NMDA * nu_NMDA)
        self.assertAlmostEqual(0.9090909090909091, s_0)
        self.assertAlmostEqual(16, experiment.effective_time_constant_up_state.mean_excitatory_conductance() / nS)
        self.assertAlmostEqual(7.999999999999998, experiment.effective_time_constant_up_state.mean_inhibitory_conductance() / nS)

        steady_up_state_results = sim_steady_state(experiment, state=experiment.network_params.up_state)

        self.assertAlmostEqual(0.2000000, steady_up_state_results.x_nmda_steady)
        self.assertAlmostEqual(0.9090909090909091, steady_up_state_results.s_nmda_steady)
        self.assertAlmostEqual(16, steady_up_state_results.g_e_steady)
        self.assertAlmostEqual(7.999999999999998, steady_up_state_results.g_i_steady)
        self.assertAlmostEqual(-48.75341763069946, steady_up_state_results.v_steady)

        print(steady_up_state_results.g_nmda_steady)
        print(experiment.synaptic_params.g_nmda * sigmoid_v(experiment, steady_up_state_results.v_steady * mV) * steady_up_state_results.s_nmda_steady / nS)

    def test_palmer_experiment_with_NMDA_blocked_can_be_created(self):
        palmer_experiment = (Experiment(wang_recurrent_config)
        .with_properties({
            SynapticParams.KEY_G_NMDA: 1E-8,
            "up_state":
                {
                    "N": 2000,
                    "nu": 82,
                    State.KEY_OMEGA: 0,
                    #"N_nmda": 0,
                    "nu_nmda": 10,
                },
            "t_range": [[0, 10_000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW:
                [PlotParams.AvailablePlots.RASTER_AND_RATE]
        }))

        self.assertEqual(0, palmer_experiment.network_params.up_state.N_NMDA)

if __name__ == '__main__':
    unittest.main()
