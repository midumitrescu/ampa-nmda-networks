import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import second, msecond
from joblib import Parallel, delayed

from iteration_13_compte.compte_utils_deserialized import compute_binned_firing_rate
from iteration_13_compte.configs import AnExampleExperiment
import _8_compare_queues_at_different_position_for_subcritical_activity as it_8


class MyTestCase(unittest.TestCase):

    def test_ensure_external_input_induces_firing(self):
        res = it_8.execute_compte_experiment(AnExampleExperiment(G_EE_AMPA=0, G_EE_NMDA=0, G_EI=0, G_IE=0, G_II=0, NE=800, NI=200,
                            label="Compare subcritical Up for broader vs more compact stimulus"))

        self.assertLess(10, res.stats().end_rate)

    def test_set_E_I_ratio_for_stability(self):

        ratio_inhibition_to_excitation = 1.5
        G_IE = 1.336
        G_EE = 0.381

        G_II = 1.024
        G_EI = 0.292

        object_under_test = it_8.execute_compte_experiment(AnExampleExperiment(G_EE_AMPA=G_EE, G_EE_NMDA=0, G_EI=G_EI, G_IE=G_IE, G_II=G_II, NE=800, NI=200,
                            label="Compare subcritical Up for broader vs more compact stimulus"))

        self.assertGreater(1, object_under_test.stats().end_rate)

    def test_appropriate_nmda_level(self):

        ratio_inhibition_to_excitation = 1.5
        G_IE = 1.336
        G_EE = 0.381

        G_II = 1.024
        G_EI = 0.292

        not_enough_nmda = it_8.execute_compte_experiment(AnExampleExperiment(G_EE_AMPA=G_EE, G_EE_NMDA=0.5, G_EI=G_EI, G_IE=G_IE, G_II=G_II, NE=800, NI=200,
                            label="Not enough NDMA for sustained activity 0.5 ns"))
        critical_nmda = it_8.execute_compte_experiment(AnExampleExperiment(G_EE_AMPA=G_EE, G_EE_NMDA=0.6, G_EI=G_EI, G_IE=G_IE, G_II=G_II, NE=800, NI=200,
                            label="Critical NDMA for sustained activity 0.6 ns"))
        rates = compute_binned_firing_rate(critical_nmda)

        supercritical_nmda = it_8.execute_compte_experiment(
            AnExampleExperiment(G_EE_AMPA=G_EE, G_EE_NMDA=0.7, G_EI=G_EI, G_IE=G_IE, G_II=G_II, NE=800, NI=200,
                                label="Critical NDMA for sustained activity 0.6 ns"))

        self.assertGreater(1, not_enough_nmda.stats().end_rate)

        self.assertLess(150, rates[200, -50:].mean(), "cue neuron must have high rate")
        self.assertGreater(300, rates[200, -50:].mean(), "cue neuron must not fire at max rate")

        self.assertGreater(2, rates[0, -50:].mean(), "non-cue neuron must be mostly quiescent")

        self.assertLess( 10, critical_nmda.stats().end_rate)
        self.assertGreater( 20, critical_nmda.stats().end_rate)
        self.assertLess( 3, critical_nmda.stats().end_rate)

        self.assertGreater(450, supercritical_nmda.stats().end_rate)


    def test_run_parallel(self):

        ratio_inhibition_to_excitation = 1.5
        G_IE = 1.336
        G_EE = 0.381

        G_II = 1.024
        G_EI = 0.292

        nmda = np.linspace(0.5, 0.56, num=10)

        def run_experiment(g_nmda):
            experiment = AnExampleExperiment(
                G_EE_AMPA=G_EE,
                G_EE_NMDA=g_nmda,
                G_EI=G_EI,
                G_IE=G_IE,
                G_II=G_II,
                NE=800,
                NI=200,
                label=f"NMDA={g_nmda:.3f} Compare subcritical Up"
            )

            return it_8.execute_compte_experiment(experiment)

        # Run in parallel
        results = Parallel(n_jobs=-1)(  # -1 uses all available cores
            delayed(run_experiment)(g) for g in nmda
        )

        print(results)


if __name__ == '__main__':
    unittest.main()
