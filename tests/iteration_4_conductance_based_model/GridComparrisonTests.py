import unittest

from brian2 import ms

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

            "panel": f"Scan $\\frac{{\\nu_E}}{{\\nu_T}}$ and g",
            "t_range": [[0, 50]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 1 * ms
        }
        experiment = Experiment(conductance_based_simulation)
        compare_g_s_vs_nu_ext_over_nu_thr(experiment, g_s, nu_ext_over_nu_thrs)

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

            "panel": f"Scan $\\frac{{\\nu_E}}{{\\nu_T}}$ and g",
            "t_range": [[0, 50]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 1 * ms
        }
        experiment = Experiment(conductance_based_simulation)
        compare_g_s_vs_nu_ext_over_nu_thr(experiment, g_s, nu_ext_over_nu_thrs)

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
            "smoothened_rate_width": 1 * ms
        }
        experiment = Experiment(conductance_based_simulation)
        compare_g_s_vs_nu_ext_over_nu_thr(experiment, g_s, nu_ext_over_nu_thrs)

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
            "smoothened_rate_width": 1 * ms
        }
        experiment = Experiment(conductance_based_simulation)
        compare_g_s_vs_nu_ext_over_nu_thr(experiment, g_s, nu_ext_over_nu_thrs)

    def test_plot_relevant_example(self):
        g_s = [4, 6, 8, 16]
        nu_ext_over_nu_thrs = [1.7, 1.8, 1.9, 2]

        conductance_based_simulation = {
            "sim_time": 5000,
            "sim_clock": 0.1 * ms,
            "g": 0,
            "g_ampa": 2.518667367869784e-06,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            "panel": f"g vs nu ext for nu ext producing close to 1 Hz",
            "t_range": [[500, 1000], [4000, 5000], [4500, 4600]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 1 * ms
        }
        experiment = Experiment(conductance_based_simulation)
        compare_g_s_vs_nu_ext_over_nu_thr(experiment, g_s, nu_ext_over_nu_thrs)

    def test_q_0(self):
        g_ampas = [2e-06, 2.5e-06, 3e-06]
        simulation = {
            "sim_time": 5000,
            "sim_clock": 0.1 * ms,
            "g": 0,
            "g_ampa": 0,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            "panel": f"g vs nu ext for nu ext producing close to 1 Hz",
            "t_range": [[100, 200], [0, 500]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 1 * ms
        }
        experiment = Experiment(simulation)
        compare_g_ampa_vs_nu_ext_over_nu_thr(experiment, g_ampas, [1.6, 1.7, 1.8])

    def test_grid_and_plot_g_ampa_nu_thrs(self):
        g_ampas = [2e-06, 2.5e-06, 2.625e-6, 2.75e-06, 3e-06]
        nu_thresholds = [1.5, 1.7, 1.8]
        simulation = {
            "sim_time": 1000,
            "sim_clock": 0.1 * ms,
            "g": 0,
            "g_ampa": 0,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            "panel": f"g vs nu ext for nu ext producing close to 1 Hz",
            "t_range": [[0, 200], [200, 500], [500, 1000], [4000, 4000]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 0.5 * ms
        }
        experiment = Experiment(simulation)
        compare_g_ampa_vs_nu_ext_over_nu_thr(experiment, g_ampas, nu_thresholds)


    def compare_two_experiments(self):
        g_ampas = [0, 1]

    def test_question_1_q_to_s_r_for_ampa_0(self):
        g_s = [3, 4, 5, 6]
        nu_ext_over_nu_thrs = [1.5, 2, 2.5]

        conductance_based_simulation = {
            "sim_time": 5000,
            "sim_clock": 0.1 * ms,
            "g": 0,
            "g_ampa": 2.518667367869784e-06,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            "panel": f"g vs nu ext for nu ext producing close to 1 Hz",
            "t_range": [[3000, 5000]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 0.5 * ms
        }
        experiment = Experiment(conductance_based_simulation)
        compare_g_s_vs_nu_ext_over_nu_thr(experiment, g_s, nu_ext_over_nu_thrs)

    # Look for question from script:
    def test_explore_model_in_AI_state(self):
        g_ampas = [3e-06]
        nu_thresholds = [1.5]
        simulation = {
            "sim_time": 5_000,
            "sim_clock": 0.1 * ms,
            "g": 0,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,
            "t_range": [[500, 1000], [3000, 5000]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 0.5 * ms
        }
        compare_g_ampa_vs_nu_ext_over_nu_thr(Experiment(simulation), g_ampas, nu_thresholds)


if __name__ == '__main__':
    unittest.main()
