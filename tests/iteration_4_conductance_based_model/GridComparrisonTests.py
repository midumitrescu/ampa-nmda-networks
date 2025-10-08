import unittest

from brian2 import ms

from Configuration import Experiment
from iteration_4_conductance_based_model.grid_computations import produce_comparrison_plot


class GridTestCases(unittest.TestCase):

    def test_grid_composition(self):
        g_s = [3, 6, 9]
        nu_ext_over_nu_thrs = [2, 10]

        conductance_based_simulation = {
            "sim_time": 1000,
            "sim_clock": 0.1 * ms,
            "g": 0,
            "g_ampa": 2.518667367869784e-06,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            "panel": f"Scan $\\frac{{\\nu_E}}{{\\nu_T}}$ and g",
            "t_range": [[0, 50]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 10 * ms
        }
        experiment = Experiment(conductance_based_simulation)
        produce_comparrison_plot(experiment, g_s, nu_ext_over_nu_thrs)

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

            "panel": f"Scan $\\frac{{\\nu_E}}{{\\nu_T}}$ and g",
            "t_range": [[500, 1000]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 5 * ms
        }
        experiment = Experiment(conductance_based_simulation)
        produce_comparrison_plot(experiment, g_s, nu_ext_over_nu_thrs)

    def test_grid__composition_4x3_with_multiple_time_slots(self):
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

            "panel": f"Scan $\\frac{{\\nu_E}}{{\\nu_T}}$ and g",
            "t_range": [[150, 300], [500, 1000]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 5 * ms
        }
        experiment = Experiment(conductance_based_simulation)
        produce_comparrison_plot(experiment, g_s, nu_ext_over_nu_thrs)

    def test_plot_relevant_example(self):
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

            "panel": f"Scan $\\frac{{\\nu_E}}{{\\nu_T}}$ and g",
            "t_range": [[500, 1000]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 5 * ms
        }
        experiment = Experiment(conductance_based_simulation)
        produce_comparrison_plot(experiment, g_s, nu_ext_over_nu_thrs)





if __name__ == '__main__':
    unittest.main()
