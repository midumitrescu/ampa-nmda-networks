import unittest

from iteration_14_check_units.run_with_one_input import simulate_with_one_presynaptic_spike, plot_dv_simulation
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_8_compute_mean_steady_state.models_and_configs import wang_recurrent_config


class ComputeDVs(unittest.TestCase):

    def test_dv_ampa_simulation(self):
        wang_experiment = Experiment(wang_recurrent_config)
        ampa_sim = simulate_with_one_presynaptic_spike(wang_experiment, synapse_active=(True, False, False))
        self.assertEqual(0.010873642128487404, ampa_sim.dv)

    def test_dv_gaba_simulation(self):
        wang_experiment = Experiment(wang_recurrent_config)
        gaba_sim = simulate_with_one_presynaptic_spike(wang_experiment, synapse_active=(False, True, False))
        self.assertEqual(0.002525296207380734, gaba_sim.dv)

    def test_dv_nmda_simulation(self):
        wang_experiment = Experiment(wang_recurrent_config)
        nmda_sim = simulate_with_one_presynaptic_spike(wang_experiment, synapse_active=(False, False, True))
        self.assertEqual(0.008799300139358479, nmda_sim.dv)

    def test_run_dv_simulation(self):
        palmer_experiment = (Experiment(wang_recurrent_config)
        .with_properties({
            "up_state":
                {
                    "N": 2000,
                    "nu": 82,
                    "N_nmda": 0,
                },
            # "down_state": {
            #    "N_E": 100,
            #    "gamma": 4,
            #    "nu": 10,

            #    "N_nmda": 0,
            #    "nu_nmda": 2,
            # },
            "t_range": [[0, 10_000]]
        }))
        ampa_sim = simulate_with_one_presynaptic_spike(palmer_experiment, synapse_active=(True, False, False))
        gaba_sim = simulate_with_one_presynaptic_spike(palmer_experiment, synapse_active=(False, True, False))
        nmda_sim = simulate_with_one_presynaptic_spike(palmer_experiment, synapse_active=(False, False, True))
        plot_dv_simulation(experiment=ampa_sim.experiment, simulation_results=[ampa_sim, gaba_sim, nmda_sim])


if __name__ == '__main__':
    unittest.main()
