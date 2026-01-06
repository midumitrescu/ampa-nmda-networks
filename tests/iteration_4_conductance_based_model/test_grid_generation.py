import unittest

from brian2 import ms
from matplotlib import pyplot as plt

from Configuration import Experiment
from iteration_4_conductance_based_model.grid_computations import compare_g_s_vs_nu_ext_over_nu_thr, \
    compare_g_ampa_vs_nu_ext_over_nu_thr


class GridTestCases(unittest.TestCase):

    def test_grid_should_also_work_for_1x1(self):
        g_s = [3]
        nu_ext_over_nu_thrs = [10]

        conductance_based_simulation = {
            "sim_time": 100,
            "sim_clock": 0.1 * ms,
            "g": 0,
            "g_ampa": 2.518667367869784e-06,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            "panel": f"Test g vs $\\frac{{\\nu_E}}{{\\nu_T}}$. Plot 1 item, 1 timestep",
            "t_range": [[0, 50]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 1 * ms
        }
        experiment = Experiment(conductance_based_simulation)
        compare_g_s_vs_nu_ext_over_nu_thr(experiment, g_s, nu_ext_over_nu_thrs)
        plt.close()

    def test_grid_composition(self):
        g_s = [3, 6]
        nu_ext_over_nu_thrs = [2, 10]

        conductance_based_simulation = {
            "sim_time": 100,
            "sim_clock": 0.1 * ms,
            "g": 0,
            "g_ampa": 2.518667367869784e-06,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            "panel": f"Test g vs $\\frac{{\\nu_E}}{{\\nu_T}}$. Plot 2x2 items, 1 timestep",
            "t_range": [[0, 50]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 1 * ms
        }
        experiment = Experiment(conductance_based_simulation)
        compare_g_s_vs_nu_ext_over_nu_thr(experiment, g_s, nu_ext_over_nu_thrs)
        plt.close()

    def test_grid_composition_4x2(self):
        g_s = [3, 6, 9, 12]
        nu_ext_over_nu_thrs = [1, 2, 3]

        conductance_based_simulation = {
            "sim_time": 1000,
            "sim_clock": 0.1 * ms,
            "g": 0,
            "g_ampa": 2.518667367869784e-06,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            "panel": f"Test g vs $\\frac{{\\nu_E}}{{\\nu_T}}$. Plot 4x3, 1 timestep",
            "t_range": [[0, 50]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 1 * ms
        }
        experiment = Experiment(conductance_based_simulation)
        compare_g_s_vs_nu_ext_over_nu_thr(experiment, g_s, nu_ext_over_nu_thrs)
        plt.close()

    def test_q_0(self):
        g_ampas = [2e-06, 2.5e-06, 3e-06]
        simulation = {
            "sim_time": 500,
            "sim_clock": 0.1 * ms,
            "g": 0,
            "g_ampa": 0,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            "panel": f"g vs nu ext for nu ext producing close to 1 Hz",
            "t_range": [[0, 50]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 1 * ms
        }
        experiment = Experiment(simulation)
        compare_g_ampa_vs_nu_ext_over_nu_thr(experiment, g_ampas, [1.6, 1.7, 1.8])
        plt.close()


if __name__ == '__main__':
    unittest.main()
