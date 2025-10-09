import itertools
import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms, siemens, cm, second, Hz

from BinarySeach import binary_search_for_target_value
from Configuration import Experiment, SynapticParams, NetworkParams, PlotParams
from iteration_4_conductance_based_model.conductance_based_model import sim_and_plot, sim


def compute_rate_for_nu_ext_over_nu_thr(nu_ext_over_nu_thr, g = 1, wait_for=4_000):
    conductance_based_simulation = {

        "sim_time": 5_000,
        "sim_clock": 0.1 * ms,
        "g": g,
        "g_ampa": 2.518667367869784e-06,
        #g_gaba": 2.518667367869784e-06,
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
                "nu_ext_over_nu_thr": 10,
                "epsilon": 0.1,
                "C_ext": 1000,

                "g_L": 0.00004,

                "panel": f"Testing conductance based model",
                "t_range": [[100, 120], [100, 300], [0, 1500], [2500, 3000], [4500, 5000]],
                "voltage_range": [-70, -30],
                "smoothened_rate_width": 3 * ms
            }


            experiment = Experiment(conductance_based_simulation)

            sim_and_plot(experiment)
            plt.show()


    def test_show_model_from_binary_search_value(self):
        conductance_based_simulation = {

            "sim_time": 5000,
            "sim_clock": 0.1 * ms,
            "g": 1,
            "g_ampa": 2.518667367869784e-06,
            "nu_ext_over_nu_thr": 1.85,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            "panel": f"Scan for increasing values of g (ratio inh nu excitation)",
            "t_range": [[3000, 3200], [4000, 5000]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 3 * ms
        }


        experiment = Experiment(conductance_based_simulation)

        sim_and_plot(experiment)
        plt.show()

    def test_understand_why_more_g_produces_more_firing(self):

        three_g_s = [0, 1, 3, 5, 10]
        nu_ext_over_nu_thrs = [1.8, 1.85, 1.9, 1.95, 2]

        # i.e. 1 Hz
        for current_g, nu_ext_over_nu_thr in itertools.product(three_g_s, nu_ext_over_nu_thrs):
            conductance_based_simulation = {

                "sim_time": 5000,
                "sim_clock": 0.1 * ms,
                "g": current_g,
                "g_ampa": 2.518667367869784e-06,
                # binary search result 2.0100483413989423 binary_search_for_target_value(lower_value=0, upper_value=10, func=look_for_rate_of_input_value, target_result=1))
                # i.e. 1 Hz
                #"nu_ext_over_nu_thr": 1.75,
                "nu_ext_over_nu_thr": nu_ext_over_nu_thr,
                "epsilon": 0.1,
                "C_ext": 1000,

                "g_L": 0.00004,

                "panel": f"Scan $\\frac{{\\nu_E}}{{\\nu_T}}$ and g",
                "t_range": [[0, 3000], [4000, 5000], [4500, 4800]],
                "voltage_range": [-70, -30],
                "smoothened_rate_width": 5 * ms
            }

            experiment = Experiment(conductance_based_simulation)

            sim_and_plot(experiment)
            plt.show()

    def test_understand_why_most_firing_is_external(self):
        for current_g in np.linspace(start=0, stop=50, num=6):
            conductance_based_simulation = {

                "sim_time": 2000,
                "sim_clock": 0.1 * ms,
                "g": current_g,
                #"g_ampa": 1.518667367869784e-06,
                #"g_ampa": 1.518667367869784e-06,
                "g_ampa": 1e-05,
                #"g_gaba": 1.518667367869784e-06,
                "nu_ext_over_nu_thr": 1,
                "epsilon": 0.1,
                "C_ext": 1000,

                "g_L": 0.00004,

                "panel": f"Testing conductance based model",
                "t_range": [[0, 100], [100, 300], [300, 500], [500, 1000], [1000, 2000], [1900, 1950]],
                "voltage_range": [-70, -30],
                "smoothened_rate_width": 10 * ms
            }

            experiment = Experiment(conductance_based_simulation)

            rate_monitor, spike_monitor, _, _, = sim_and_plot(experiment)
            plt.show()

    def test_model_below_threshold_for_very_long_time(self):

        conductance_based_simulation = {

            "sim_time": 20_000,
            "sim_clock": 0.1 * ms,
            "g": 4,
            "g_ampa": 2.518667367869784e-06,
            "nu_ext_over_nu_thr": 1.8,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            "panel": f"Testing conductance based model",
            "t_range": [10_000, 19_000],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 2 * ms
        }

        experiment = Experiment(conductance_based_simulation)

        rate_monitor, spike_monitor, _, _, = sim_and_plot(experiment)
        plt.show()

    #@unittest.skip("too long rn")
    # Results:
    # g = 7 => (2.3248291021445766, 2.324829102435615)
    # g = 1 => (1.868896484375, 1.868896484384095)
    # g = 4 => (1.8871950361062773, 1.8871950363973156)
    def test_find_nu_ext_over_nu_thr_binary_search(self):

        def look_for_rate_of_input_value(value):
            return compute_rate_for_nu_ext_over_nu_thr(value, g=4)[1]
        #(0.007029794622212648, 0.007029795087873936)
        # (2.3248291021445766, 2.324829102435615)+
        print(binary_search_for_target_value(lower_value=0, upper_value=10, func=look_for_rate_of_input_value, target_result=1))

    def test_model_from_q_0_is_stable_on_the_long_run(self):
        simulation = {
            "sim_time": 5_000,
            "sim_clock": 0.1 * ms,
            "g": 0,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,
            "t_range": [[0, 200], [200, 500], [500, 1000], [3000, 5000]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 0.5 * ms
        }
        experiment = Experiment(simulation)

        long_run_to_check_stability_and_cv = experiment.with_property(SynapticParams.KEY_G_AMPA, 3e-06) \
            .with_property(NetworkParams.KEY_NU_E_OVER_NU_THR, 1.5) \
            .with_property(Experiment.KEY_SIM_TIME, 30_000) \
            .with_property(PlotParams.KEY_T_RANGE, [[10_000, 30_000]])\
            .with_property(PlotParams.KEY_PANEL, "Model is stables")

        sim_and_plot(long_run_to_check_stability_and_cv)


if __name__ == '__main__':
    unittest.main()
