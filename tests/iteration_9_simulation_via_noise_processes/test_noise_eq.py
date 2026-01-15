import unittest

from brian2 import msecond, siemens, cm

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_8_compute_mean_steady_state.models_and_configs import wang_recurrent_config
from iteration_9_simulation_via_noise_processes.one_compartment_with_python_native_simulation import simulate_native


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
