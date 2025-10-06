import itertools
import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms, siemens, cm, second, Hz

from BinarySeach import binary_search_for_target_value
from Configuration import Experiment
from iteration_4_conductance_based_model.conductance_based_model import sim_and_plot, sim


def compute_rate_for_nu_ext_over_nu_thr(nu_ext_over_nu_thr, wait_for=4_000):
    conductance_based_simulation = {

        "sim_time": 5_000,
        "sim_clock": 0.1 * ms,
        "g": 15,
        "g_ampa": 2.518667367869784e-06,
        "g_gaba": 2.518667367869784e-06,
        "nu_ext_over_nu_thr": nu_ext_over_nu_thr,
        "epsilon": 0.1,
        "C_ext": 1000,

        "g_L": 0.00004,

        "panel": f"Testing conductance based model",
        "t_range": [[100, 120], [100, 300], [0, 1000], [1000, 2000]],
        "voltage_range": [-70, -30],
        "smoothened_rate_width": 3 * ms
    }

    experiment = Experiment(conductance_based_simulation)

    skip_iterations = int(wait_for / (experiment.sim_clock / ms))
    rate_monitor, _, _, _ = sim(experiment)

    mean_unsmoothened = np.mean(rate_monitor.rate[skip_iterations:])
    mean_smothened = np.mean(rate_monitor.smooth_rate(width=experiment.plot_params.smoothened_rate_width)[skip_iterations:])
    print(f"For {nu_ext_over_nu_thr : .5f}, we get {mean_smothened}, {mean_unsmoothened}")
    print(f"For {nu_ext_over_nu_thr : .5f}, we get without units {mean_smothened / Hz}, {mean_unsmoothened / Hz}")
    return mean_unsmoothened / Hz, mean_smothened / Hz


class MyTestCase(unittest.TestCase):

    def test_new_configuration_is_parsed(self):
        new_config = {

            "g": 1,
            "nu_ext_over_nu_thr": 1,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_ampa": 0.002,
            "g_gaba": 0.002,

            "tau_ampa_ms": 5,
            "tau_gaba_ms": 5,
        }

        object_under_test = Experiment(new_config)
        self.assertEqual(0.002 * siemens / cm ** 2, object_under_test.synaptic_params.g_ampa)
        self.assertEqual(0.002 * siemens / cm ** 2, object_under_test.synaptic_params.g_gaba)

        self.assertEqual(5 * ms, object_under_test.synaptic_params.tau_ampa)
        self.assertEqual(5 * ms, object_under_test.synaptic_params.tau_gaba)

        self.assertEqual(1, object_under_test.network_params.g)

    def test_model_runs_with_default_eq(self):
        conductance_based_simulation = {

            "sim_time": 100,
            "sim_clock": 0.05 * ms,

            "g": 0,
            "nu_ext_over_nu_thr": 1,
            "epsilon": 0.1,
            "C_ext": 1000,

            "panel": f"Testing conductance based model",
            "t_range": [100, 120],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 0.5 * ms
        }

        experiment = Experiment(conductance_based_simulation)
        sim_and_plot(experiment)
        plt.show()

    def test_model_runs_only_with_exc_current(self):
        conductance_based_simulation = {

            "sim_time": 100,
            "sim_clock": 0.05 * ms,

            "g": 1,
            "nu_ext_over_nu_thr": 2,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_ampa": 0.002,
            "g_gaba": 0.002,

            "panel": f"Testing conductance based model only with excitatory current",
            "t_range": [100, 120],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 0.5 * ms
        }

        experiment = Experiment(conductance_based_simulation)
        sim_and_plot(experiment, eq="""        
                            dv/dt = - 1/C * g_e * (v-E_ampa) : volt
                            dg_e/dt = -g_e / tau_ampa : siemens / meter**2
                            dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
                        """)
        plt.show()

    def test_model_scan_nu_ext_for_excitation_balance(self):

        for nu_ext_over_nu_thr in np.linspace(start=1, stop=3, num=3):
            conductance_based_simulation = {

                "sim_time": 1000,
                "sim_clock": 0.05 * ms,

                "g": 1,
                "nu_ext_over_nu_thr": nu_ext_over_nu_thr,
                "epsilon": 0.1,
                "C_ext": 1000,

                "g_ampa": 0.002,
                "g_gaba": 0.002,

                "panel": f"Scanning ",
                "t_range": [100, 120],
                "voltage_range": [-70, -30],
                "smoothened_rate_width": 0.5 * ms
            }
            experiment = Experiment(conductance_based_simulation)
            sim_and_plot(experiment, eq="""        
                                        dv/dt = - 1/C * g_e * (v-E_ampa) : volt
                                        dg_e/dt = -g_e / tau_ampa : siemens / meter**2
                                        dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
                                    """)
            plt.show()

    def test_model_runs_only_with_inhibitory_current(self):
        conductance_based_simulation = {

            "sim_time": 1500,
            "sim_clock": 0.05 * ms,

            "g": 1,
            "nu_ext_over_nu_thr": 1,
            "epsilon": 0.1,
            "C_ext": 1000,

            "panel": f"Testing conductance based model",
            "t_range": [100, 120],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 0.5 * ms
        }


        experiment = Experiment(conductance_based_simulation)

        self.assertEqual(1, experiment.network_params.g)

        sim_and_plot(experiment, eq="""        
                            dv/dt = - 1/C * g_i * (v-E_gaba) : volt
                            dg_e/dt = -g_e / tau_ampa : siemens / meter**2
                            dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
                        """)
        plt.show()

    def test_default_model_works(self):
        for current_nu_ext_over_nu_thr, current_g in itertools.product(np.linspace(start=0.01, stop=0.2, num=20), np.linspace(start=10, stop=20, num=10)):
            conductance_based_simulation = {

                "sim_time": 5000,
                "sim_clock": 0.1 * ms,
                "g": current_g,
                "g_ampa": 2.518667367869784e-06,
                "g_gaba": 2.518667367869784e-06,
                "nu_ext_over_nu_thr": current_nu_ext_over_nu_thr,
                "epsilon": 0.1,
                "C_ext": 1000,

                "g_L": 0.0004,

                "panel": f"Testing conductance based model",
                "t_range": [[100, 120], [100, 300], [0, 1500], [2500, 3000], [4500, 5000]],
                "voltage_range": [-70, -30],
                "smoothened_rate_width": 3 * ms
            }


            experiment = Experiment(conductance_based_simulation)

            sim_and_plot(experiment)
            plt.show()

    def test_understand_why_most_firing_is_external(self):
        for current_g in np.linspace(start=0, stop=20, num=5):
            conductance_based_simulation = {

                "sim_time": 7000,
                "sim_clock": 0.1 * ms,
                "g": current_g,
                "g_ampa": 1.518667367869784e-06,
                "g_gaba": 1.518667367869784e-06,
                "nu_ext_over_nu_thr": 0.007029794622212648,
                "epsilon": 0.1,
                "C_ext": 1000,

                "g_L": 0.00004,

                "panel": f"Testing conductance based model",
                #"t_range": [[2000, 4000], [4500, 5000], [4500, 4600], [4500, 4520], [6000, 7000]],
                "t_range": [[6000, 7000]],
                "voltage_range": [-70, -30],
                "smoothened_rate_width": 3 * ms
            }

            experiment = Experiment(conductance_based_simulation)

            sim_and_plot(experiment)
            plt.show()

    def test_find_nu_ext_over_nu_thr_binary_search(self):

        def look_for_rate_of_input_value(value):
            return compute_rate_for_nu_ext_over_nu_thr(value)[1]
        #(0.007029794622212648, 0.007029795087873936)
        print(binary_search_for_target_value(lower_value=0, upper_value=1, func=look_for_rate_of_input_value, target_result=1))





if __name__ == '__main__':
    unittest.main()
