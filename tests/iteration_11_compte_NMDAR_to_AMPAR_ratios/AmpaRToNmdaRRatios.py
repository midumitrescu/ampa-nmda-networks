import unittest

import matplotlib.pyplot as plt
import numpy as np

from Configuration import Experiment
from Plotting import show_plots_non_blocking
from iteration_11_compte_NMDAR_to_AMPAR_ratios.simulate_one_EPSP import simulate_one_epsp
from iteration_8_compute_mean_steady_state.models_and_configs import palmer_experiment


class AmpaRtoNmdaREpspTestCase(unittest.TestCase):

    def test_one_EPSP(self):
        results = simulate_one_epsp(palmer_experiment.with_property(Experiment.KEY_CURRENTS_TO_RECORD, ["I_ampa", "I_gaba", "I_nmda"]).with_property("t_range", [0, 1000]))

        plt.plot(results.currents.t, results.currents.I_nmda[0], label="I_nmda")
        plt.plot(results.currents.t, results.currents.I_ampa[0], label="I_ampa")
        plt.plot(results.currents.t, results.currents.I_gaba[0], label="I_gaba")
        plt.legend()

        print(results)
        print("NMDA ", results.currents.q_nmda)
        print("AMPA ", results.currents.q_ampa)
        print("GABA ", results.currents.q_gaba)
        print("NMDAR/(NMDAR + AMPAR)",  results.currents.q_nmda/(results.currents.q_nmda +  results.currents.q_ampa))

        print("Peak NMDAR/AMPAR currents", np.max(np.abs(results.currents.I_nmda)) / np.max(np.abs(results.currents.I_ampa)))
        show_plots_non_blocking()


if __name__ == '__main__':
    unittest.main()
