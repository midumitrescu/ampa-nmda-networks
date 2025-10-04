import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms, siemens, cm

from Configuration import Experiment
from iteration_4_conductance_based_model.conductance_based_model import sim


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
        sim(experiment)
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
        sim(experiment, eq="""        
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
            sim(experiment, eq="""        
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

        sim(experiment, eq="""        
                            dv/dt = - 1/C * g_i * (v-E_gaba) : volt
                            dg_e/dt = -g_e / tau_ampa : siemens / meter**2
                            dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
                        """)
        plt.show()

    def test_default_model_works(self):

        for mult in np.linspace(start=0.1, stop=2, num=20):
            conductance_based_simulation = {

                "sim_time": 1500,
                "sim_clock": 0.05 * ms,

                "g": 1,
                "g_ampa": 0,
                "nu_ext_over_nu_thr": 1 * mult,
                "epsilon": 0.1,
                "C_ext": 1000,

                "panel": f"Testing conductance based model",
                "t_range": [100, 120],
                "voltage_range": [-70, -30],
                "smoothened_rate_width": 0.5 * ms
            }


            experiment = Experiment(conductance_based_simulation)

            self.assertEqual(1, experiment.network_params.g)

            sim(experiment)
            plt.show()




if __name__ == '__main__':
    unittest.main()
