import copy
import unittest
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from brian2 import ms, ufarad, cm, siemens
import numpy as np

from Configuration import Experiment
from Plotting import plot_non_blocking

some_params = {
    "panel": f"Testing matplotlib Text generation, sim clock = 0.05 ms",
    "g": 0,
    "sim_time": 1500,
    "sim_clock": 0.05 * ms,
    "nu_ext_over_nu_thr": 1,
    "g_ampa": 1e-05,
    "t_range": [100, 120],
    "voltage_range": [-70, -30],
    "epsilon": 0,
    "C_ext": 1000,
    "smoothened_rate_width": 0.5 * ms
}

experiment = Experiment(some_params)

class SimpleIntegrationstCase(unittest.TestCase):

    @staticmethod
    def test_simple_plotting_works():
        plt.plot(np.linspace(0, 1000), np.linspace(0, 1000))
        plt.title("Test")
        plot_non_blocking()

    @staticmethod
    def test_latex_plotting_works():
        plt.rcParams['text.usetex'] = True
        plt.plot(np.linspace(0, 1000), np.linspace(0, 1000))
        plt.title(r'$\gamma$, $\frac{\nu_1}{\mu_2}$')
        plot_non_blocking()

class ComplexLatexTextsTestCase(unittest.TestCase):

    def test_matplotlib_text_generation(self):
        plt.figure(figsize=(10, 10))
        plt.title(experiment.gen_plot_title())
        plt.subplots_adjust(top=0.7)
        plot_non_blocking()

    def test_matplotlib_text_generation_for_g_L_004_siemens(self):
        plt.figure(figsize=(8, 8))

        config_for_siemens_magnitude = copy.copy(some_params)
        config_for_siemens_magnitude["g_L"] = 0.004
        object_under_test = Experiment(config_for_siemens_magnitude)
        plt.title(f"$g_L=0.004$ S - {object_under_test.gen_plot_title()}")
        plt.subplots_adjust(top=0.7)
        plot_non_blocking()

    def test_matplotlib_text_generation_for_nu_ext_over_nu_thr(self):
        # plt.plot()
        plt.figure(figsize=(10, 10))
        plt.title(r'$\frac{\nu_\mathrm{Ext}}{\nu_\mathrm{Thr}}=$'f"{1}")
        plt.xlabel(r"$\nu_\mathrm{Ext}$")
        plt.subplots_adjust(top=0.7)
        plot_non_blocking()

    def test_texts_for_it_5_hidden_variables(self):

        for text in [r'$ g_\mathrm{NMDA}$''\n'r'[$\frac{{nS}}{{\mathrm{{cm}}^2}}]$', r"$I_\mathrm{NMDA}$""\n""[nA]"]:
            plt.figure(figsize=(10, 10))
            plt.title(text)
            plt.xlabel(r"$\nu_\mathrm{Ext}$")
            plt.subplots_adjust(top=0.7)
            plot_non_blocking()

    def test_texts_for_nmda_refactoring_and_understanding_why_s_is_above_1(self):

        for text in [r'Driving force of $s_{\text{NMDA}}$ = $\alpha \cdot x \cdot (1-s_{\text{NMDA}})$',
                     'Driving force of $s_{\\text{NMDA}}$ = $\\alpha \\cdot x \\cdot (1-s_{\\text{NMDA}})$']:
            plt.figure(figsize=(10, 10))
            plt.title(text)
            plt.xlabel(r"$\nu_\mathrm{Ext}$")
            plt.subplots_adjust(top=0.7)
            plot_non_blocking()

    def test_plot_name_generation(self):
        config = {
            "sim_time": 5000,
            "sim_clock": 0.1 * ms,
            "g": 1,
            "g_ampa": 2.518667367869784e-06,
            # binary search result 2.0100483413989423 binary_search_for_target_value(lower_value=0, upper_value=10, func=look_for_rate_of_input_value, target_result=1))
            # i.e. 1 Hz
            # "nu_ext_over_nu_thr": 1.75,
            "nu_ext_over_nu_thr": 1.9,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            "panel": f"Scan $\\frac{{\\nu_E}}{{\\nu_T}}$ and g",
            "t_range": [[0, 3000], [4000, 5000], [4500, 4800]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 2 * ms
        }

        object_under_test = Experiment(config)
        plt.title(object_under_test.gen_plot_title())
        # \\text\u007b cm \u007d^2
        plt.subplots_adjust(top=0.7)
        plt.tight_layout()
        plot_non_blocking()

        self.assertEqual("""Scan $\\frac{\\nu_E}{\\nu_T}$ and g
    Network: [N=12500, $N_E=10000$, $N_I=2500$, $\gamma=0.25$, $\epsilon=0.1$]
    Input: [$\\nu_T=2. Hz$, $\\frac{\\nu_E}{\\nu_T}=1.90$, $\\nu_E=3.80$ Hz]
    Neuron: [$C=1. uF$, $g_L=40. uS$, $\\theta=-40. mV$, $V_R=-65. mV$, $E_L=-65. mV$, $\\tau_M=25. ms$, $\\tau_{\\mathrm{ref}}=2. ms$]
    Synapse: [$g_{\\mathrm{AMPA}}=2.52\\,\\mu\\mathrm{S}$, $g_{\\mathrm{GABA}}=2.52\\,\\mu\\mathrm{S}$, $g=1$]""", object_under_test.gen_plot_title())


if __name__ == '__main__':
    unittest.main()
