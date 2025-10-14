import unittest

from brian2 import ms
from matplotlib import pyplot as plt

import iteration_4_conductance_based_model.conductance_based_model
from Configuration import NetworkParams, SynapticParams, PlotParams, Experiment
from iteration_5_nmda.compare_network_w_and_w_o_nmda import question_1_network_with_nmda_wang_model, \
    question_1_network_ampa_gaba_only, question_1_network_with_translated_nmda_model


class TestsForMeeting(unittest.TestCase):

    def test_q_1_ampa_only(self):
        question_1_network_ampa_gaba_only()
        plt.close()

    def test_q_1_nmda_wang(self):
        question_1_network_with_nmda_wang_model()
        plt.close()

    def test_q_1_nmda_translated(self):
        question_1_network_with_translated_nmda_model()
        plt.close()

    def test_various_g_s(self):

        for g in [2, 3, 4, 5]:
            question_1_network_ampa_gaba_only(g=g)
            plt.close()
            question_1_network_with_nmda_wang_model(g=g)
            plt.close()
            question_1_network_with_translated_nmda_model(g=g)
            plt.close()

'''Take some special cases and to long runs
'''
class SpecialRuns(unittest.TestCase):

    #
    def test_run_1(self):
        simulation = {
            "sim_time": 5_000,
            "sim_clock": 0.1 * ms,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            NetworkParams.KEY_G: 5,
            NetworkParams.KEY_NU_E_OVER_NU_THR: 1.575,
            SynapticParams.KEY_G_AMPA: 2.875e-6,

            PlotParams.KEY_PANEL: "Asynchronous Irregular?",
            "t_range": [[0, 500], [3500, 3600], [3000, 5000]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 3 * ms
        }
        experiment = Experiment(simulation)
        iteration_4_conductance_based_model.conductance_based_model.sim_and_plot(experiment)
        plt.close()




if __name__ == '__main__':
    unittest.main()
