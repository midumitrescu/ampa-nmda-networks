import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import Hz, ms
from numpy.testing import assert_array_equal, assert_allclose

from Plotting import show_plots_non_blocking, prepare_bigger_fonts
from iteration_16.model import calibrated_configuration
from iteration_16.nmda_compartment_model import NMDASimulationWangCompartments


def plot_k_sweep_results(result_by_k, k_s, config):

    fig = plt.figure(figsize=(14, 18))
    number_of_axes = len(k_s) + 4
    outer = fig.add_gridspec(number_of_axes, 1, height_ratios=[1]  * number_of_axes)

    # =========================================================
    # ROW 1: VOLTAGES (all k on same axes)
    # =========================================================
    ax_v = fig.add_subplot(outer[0])

    for k in k_s:
        r = result_by_k[k]

        t = r.neuron_monitor.t
        v = r.neuron_monitor.v[0]

        ax_v.plot(t, v, label=f"k={k}")

    ax_v.set_title("$V_m$ - Membrane voltage comparison")
    ax_v.set_ylabel("$V_m$ (mV)")
    ax_v.legend()

    # =========================================================
    # ROW 2: NMDA CURRENTS (all k on same axes)
    # =========================================================
    ax_i = fig.add_subplot(outer[1])

    for k in k_s:
        r = result_by_k[k]

        t = r.current_monitor.t
        i_nmda = r.current_monitor.i_nmda_total[0]

        ax_i.plot(t, i_nmda, label=f"k={k}")

    ax_i.set_title("NMDA current comparison")
    ax_i.set_ylabel("I_NMDA (nA)")
    ax_i.legend()

    
    # =========================================================
    # ROWS 3+: NMDA COMPARTMENTS PER K
    # =========================================================
    for idx, k in enumerate(k_s):
        r = result_by_k[k]

        t =  r.nmda_monitor.t

        ax = fig.add_subplot(outer[2 + idx])

        for c in range(len(r.nmda_monitor.s_nmda)):
            ax.plot(
                t,
                r.nmda_monitor.s_nmda[c],
                alpha=0.6,
            )

        ax.set_title(f"NMDA compartments (k={k})")
        ax.set_ylabel("s_nmda")

    # =========================================================
    # ROWS second to last+: Sum of S_NMDA variables
    # =========================================================
    ax_s_sum = fig.add_subplot(outer[number_of_axes - 2])

    for k in k_s:
        r = result_by_k[k]

        t = r.nmda_monitor.t
        s_nmda_sum = np.sum(r.nmda_monitor.s_nmda, axis=0)

        ax_s_sum.plot(t, s_nmda_sum, label=f"k={k}")

    ax_s_sum.set_title(r"Total NMDA activation $\sum_i s_i^{(k)}(t)$")
    ax_s_sum.set_ylabel("Σ $s_\mathrm{nmda}$")
    ax_s_sum.legend()

    # =========================================================
    # ROWS last: Ration #k=1 to #k=current
    # =========================================================

    ax_ratio = fig.add_subplot(outer[number_of_axes - 1])
    ref_k = 1

    r_ref = result_by_k[ref_k]
    s_ref = np.sum(r_ref.nmda_monitor.s_nmda, axis=0)

    for k in k_s:
        r = result_by_k[k]

        t = r.nmda_monitor.t
        s_cur = np.sum(r.nmda_monitor.s_nmda, axis=0)

        if k == ref_k:
            ratio = np.ones_like(s_ref)  # control
        else:
            ratio = s_cur / (s_ref + 1e-12)

        ax_ratio.plot(t, ratio, label=f"{k} Compartments")

    ax_ratio.set_title(r"$R_k(t) = \frac{ \sum_i s_i^{(k)}(t)}{s_i^{(1)}(t)}$")
    ax_ratio.set_ylabel("ratio$")
    ax_ratio.legend()

    ax_ratio.set_xlabel("Time (ms)")

    plt.tight_layout()
    prepare_bigger_fonts()
    show_plots_non_blocking()

class NMDACompartmentsSanityTestCases(unittest.TestCase):

    def test_config_can_be_initialized_with_NE_and_NI(self):
        object_under_test = calibrated_configuration.with_property(r_e=50 * Hz, r_i=100 * Hz, N_E=1, N_I=1)
        self.assertEqual(1, object_under_test.N_E)
        self.assertEqual(1, object_under_test.N_I)
        self.assertEqual(2, object_under_test.N)
        self.assertEqual(0.5, object_under_test.gamma)

    def test_config_can_be_initialized_with_N_and_default_gamma(self):
        object_under_test = calibrated_configuration.with_property(r_e=50 * Hz, r_i=100 * Hz, N=10)
        self.assertEqual(10, object_under_test.N)
        self.assertEqual(0.8, object_under_test.gamma)
        self.assertEqual(8, object_under_test.N_E)
        self.assertEqual(2, object_under_test.N_I)

    def test_presynaptic_spikes_and_indes_in_compartments_are_disjoint_two_compartments(self):
        config = calibrated_configuration.with_property(r_e=2 * Hz, r_i = 2 * Hz, N_E = 10, N_I = 5, seed=200)
        object_under_test = NMDASimulationWangCompartments.run_and_plot(config, k=2)

        ampa_spikes = object_under_test.ampa_spikes
        gaba_spikes = object_under_test.gaba_spikes

        nmda_compartments = np.asarray(object_under_test.nmda_spikes.compartments)

        compartments = np.unique(nmda_compartments)
        assert compartments.size == 2

        spikes = {
            c: np.sort(object_under_test.nmda_spikes.t[nmda_compartments == c])
            for c in compartments
        }

        indexes_first_cluster = range(0, 5)
        indexes_second_cluster = range(5, 10)

        first_cluster_spikes = np.sort(np.concatenate([ ampa_spikes.all_values["t"][i] / ms for i in indexes_first_cluster]))
        second_cluster_spikes = np.sort(np.concatenate([ ampa_spikes.all_values["t"][i] / ms for i in indexes_second_cluster]))
        spikes_c0 = spikes[compartments[0]]
        spikes_c1 = spikes[compartments[1]]

        assert_allclose(first_cluster_spikes, spikes_c0)
        assert_allclose(second_cluster_spikes, spikes_c1)

        union = np.sort(np.concatenate([spikes_c0, spikes_c1]))
        ampa_sorted = np.sort(ampa_spikes.t)

        np.testing.assert_array_equal(ampa_sorted, union)

        self.assertEqual(0, len(set(spikes_c0).intersection(set(spikes_c1))), "There is no reason for spikes to overlap")

        # Must not be identical event counts either
        assert spikes[compartments[0]].size != spikes[compartments[1]].size or not np.array_equal(
            spikes[compartments[0]], spikes[compartments[1]]
        )

        """
            Ensure no GABA spike time coincides with any NMDA event time.
            """

        gaba_t = np.asarray(gaba_spikes.t)
        nmda_t = np.asarray(object_under_test.nmda_spikes.t)

        for t in gaba_t:
            assert not np.any(np.isclose(nmda_t, t, atol=1e-12)), (
                f"GABA spike at time {t} overlaps with NMDA events"
            )

    def test_simulation_results_are_correctly_extracted_to_numpy(self):
        config = calibrated_configuration.with_property(r_e=2 * Hz, r_i = 2 * Hz, N_E = 10, N_I = 5, seed=200)
        object_under_test = NMDASimulationWangCompartments.run_and_plot(config, k=2)

        ampa_spikes = object_under_test.ampa_spikes

        self.assertEqual(7, len(ampa_spikes.t))
        assert_allclose([288.3 , 298.15, 356.45, 371.45, 409.95, 429.6 , 460.1 ], ampa_spikes.t)
        assert_array_equal([8, 3, 6, 8, 7, 5, 1], ampa_spikes.i)
        self.assertEqual(1.4, ampa_spikes.mean_rate)
        self.assertEqual(7, ampa_spikes.num_spikes)
        assert np.all(ampa_spikes.i < 10)
        assert np.all(ampa_spikes.i >= 0)


        gaba_spikes = object_under_test.gaba_spikes
        self.assertEqual(10, len(gaba_spikes.t))
        assert_allclose([133.65, 164.05, 182.25, 197.15, 203.15, 247.3, 306.3, 306.70000000000005, 422.05, 493.6], gaba_spikes.t)
        assert_array_equal([1, 4, 3, 0, 4, 3, 2, 2, 1, 3], gaba_spikes.i)

        self.assertEqual(4,gaba_spikes.mean_rate)

        assert np.all(gaba_spikes.i < 5)
        assert np.all(gaba_spikes.i >= 0)

        assert_allclose([298.15, 460.1, 288.3, 356.45, 371.45, 409.95, 429.6], object_under_test.nmda_spikes.t)
        assert_allclose([0, 0, 1, 1, 1, 1, 1], object_under_test.nmda_spikes.compartments)

    def test_simulation_works_when_k_equals_N_E(self):
        config = calibrated_configuration.with_property(r_e=2 * Hz, r_i = 2 * Hz, N_E = 10, N_I = 5, seed=200)
        object_under_test = NMDASimulationWangCompartments.run_and_plot(config, k=10)










if __name__ == '__main__':
    unittest.main()
