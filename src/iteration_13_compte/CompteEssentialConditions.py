import unittest

import numpy as np
from brian2 import ms, Hz
from joblib import Parallel, delayed

import _8_compare_queues_at_different_position_for_subcritical_activity as it_8
from iteration_13_compte.compte_utils_deserialized import compute_binned_firing_rate, plot_compte_results
from iteration_13_compte.configs import AnExampleExperiment, CueInfo


def successful(results):
    pass


class MyTestCase(unittest.TestCase):

    def test_ensure_external_input_induces_firing(self):
        res = it_8.execute_compte_experiment(AnExampleExperiment(G_EE_AMPA=0, G_EE_NMDA=0, G_EI=0, G_IE=0, G_II=0, NE=800, NI=200,
                            label="Compare subcritical Up for broader vs more compact stimulus", cues=[]))

        self.assertLess(10, res.stats().end_rate)

    def test_set_E_I_ratio_for_stability(self):

        ratio_inhibition_to_excitation = 1.5
        G_IE = 1.336
        G_EE = 0.381

        G_II = 1.024
        G_EI = 0.292

        object_under_test = it_8.execute_compte_experiment(AnExampleExperiment(G_EE_AMPA=G_EE, G_EE_NMDA=0, G_EI=G_EI, G_IE=G_IE, G_II=G_II, NE=800, NI=200,
                            label="Compare subcritical Up for broader vs more compact stimulus", cues=[]))

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
        G_IE = 1.4
        G_EE = 0.381

        G_II = 1.2
        G_EI = 0.292

        nmda = np.linspace(0.5, 0.56, num=10)

        def run_experiment(G_IE, G_EE, G_II, G_EI, g_nmda):
            experiment = AnExampleExperiment(
                G_EE_AMPA=G_EE,
                G_EE_NMDA=g_nmda,
                G_EI=G_EI,
                G_IE=G_IE,
                G_II=G_II,
                NE=800,
                NI=200,
                label=f"Look for stability under cue: G_IE={G_IE: .3f}, G_EI={G_EI: .3f}, G_II={G_II: .3f}, G_EI={G_EI: .3f}, g_nmda={g_nmda: .3f}"
            )

            results = it_8.run_compte_experiment(experiment)

            if successful(results):
                plot_compte_results(results)
                print()

            return results

        # Run in parallel
        results = Parallel(n_jobs=-1)(  # -1 uses all available cores
            delayed(run_experiment)(g) for g in nmda
        )

        print(results)

    def test_check_for_two_inputs(self):
        G_IE = 1.336
        G_EE = 0.381

        G_II = 1.024
        G_EI = 0.292

        cue_1 = CueInfo(degree=90, delay=200 * ms, duration=200 * ms, sigma=2, spread=10, cue_rate=50 * Hz)
        cue_2 = CueInfo(degree=270, delay=600 * ms, duration=400 * ms, sigma=4, spread=20, cue_rate=2000 * Hz)

        critical_nmda = it_8.run_compte_experiment(
            AnExampleExperiment(G_EE_AMPA=G_EE, G_EE_NMDA=0.55, G_EI=G_EI, G_IE=G_IE, G_II=G_II, NE=800, NI=200,
                                label="Are both cues visible?", cues=[cue_1, cue_2]))
        plot_compte_results(results=critical_nmda, plot_current_for_neurons=[(200, "cue 1"), (600, "cue 2")])

        critical_nmda = it_8.run_compte_experiment(
            AnExampleExperiment(G_EE_AMPA=G_EE, G_EE_NMDA=0.55, G_EI=G_EI, G_IE=G_IE, G_II=G_II, NE=800, NI=200,
                                label="Are both cues visible?", cues=[cue_1]))
        plot_compte_results(results=critical_nmda, plot_current_for_neurons=[(200, "cue 1"), (600, "cue 2")])

        critical_nmda = it_8.run_compte_experiment(
            AnExampleExperiment(G_EE_AMPA=G_EE, G_EE_NMDA=0.55, G_EI=G_EI, G_IE=G_IE, G_II=G_II, NE=800, NI=200,
                                label="Are both cues visible?", cues=[cue_2]))
        plot_compte_results(results=critical_nmda, plot_current_for_neurons=[(200, "cue 1"), (600, "cue 2")])


if __name__ == '__main__':
    unittest.main()
