import unittest

from brian2 import ms
from matplotlib import pyplot as plt

import iteration_4_conductance_based_model.conductance_based_model
from Configuration import NetworkParams, SynapticParams, PlotParams, Experiment
from iteration_5_nmda.compare_network_w_and_w_o_nmda import question_1_network_with_nmda_wang_model, \
    question_1_network_ampa_gaba_only, question_1_network_with_translated_nmda_model
from iteration_5_nmda.network_with_nmda import wang_model, sim_and_plot, translated_model, wang_model_with_extra_variables, \
    translated_model_with_extra_variables

import numpy as np


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

    # issue here is the very regular firing patters for input with NMDA and quite high 5.
    # Look at them in detail
    def test_run_2(self):
        simulation = {
            "sim_time": 600,
            "sim_clock": 0.1 * ms,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            NetworkParams.KEY_G: 5,
            NetworkParams.KEY_NU_E_OVER_NU_THR: 1.49,
            SynapticParams.KEY_G_AMPA: 2.875e-6,

            PlotParams.KEY_PANEL: "(very) ? ",
            "t_range": [[0, 500], [450, 500]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 1 * ms,

            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["sigmoid_v", "x", "g_nmda", "I_nmda", "one_minus_g_nmda"],

            PlotParams.KEY_PLOT_NEURONS_W_HIDDEN_VARIABLES: [1, 1_000, 10_001, 11_000]
        }

        sim_and_plot(Experiment(simulation).with_property(Experiment.KEY_SELECTED_MODEL, wang_model_with_extra_variables).with_property(PlotParams.KEY_PANEL, "Synchronous regular? Wang model"))
        plt.close()

        sim_and_plot(Experiment(simulation).with_property(Experiment.KEY_SELECTED_MODEL, translated_model_with_extra_variables).with_property(PlotParams.KEY_PANEL, "Synchronous regular? Model with NMDA 1/2 Translated"))
        plt.close()

    def test_run_2_with_less_external_input(self):

        for external_input in np.linspace(1.6, 1.4, 10):
            simulation = {
                "sim_time": 600,
                "sim_clock": 0.1 * ms,
                "epsilon": 0.1,
                "C_ext": 1000,

                "g_L": 0.00004,

                NetworkParams.KEY_G: 5,
                NetworkParams.KEY_NU_E_OVER_NU_THR: external_input,
                SynapticParams.KEY_G_AMPA: 2.875e-6,

                PlotParams.KEY_PANEL: "(very) ? ",
                "t_range": [[0, 500], [450, 500]],
                "voltage_range": [-70, -30],
                "smoothened_rate_width": 1 * ms,

                Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["sigmoid_v", "x", "g_nmda", "I_nmda", "one_minus_g_nmda"],

                PlotParams.KEY_PLOT_NEURONS_W_HIDDEN_VARIABLES: [1, 1_000, 10_001, 11_000],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.PSD_AND_CVS, PlotParams.AvailablePlots.RASTER_AND_RATE]
            }


            sim_and_plot(Experiment(simulation).with_property(Experiment.KEY_SELECTED_MODEL, translated_model_with_extra_variables).with_property(PlotParams.KEY_PANEL, "Synchronous regular? Model with NMDA 1/2 Translated"))
            plt.close()






if __name__ == '__main__':
    unittest.main()
