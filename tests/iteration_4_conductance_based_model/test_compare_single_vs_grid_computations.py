import unittest
from loguru import logger
import sys

import matplotlib.pyplot as plt
from brian2 import ms

from Configuration import Experiment
from iteration_4_conductance_based_model.conductance_based_model import sim_and_plot
from iteration_4_conductance_based_model.grid_computations import compare_g_s_vs_nu_ext_over_nu_thr

logger.remove()
logger.add(sys.stderr, level="INFO")


class SingleVSGridComputationTestCase(unittest.TestCase):
    def test_single_simulation_and_grid_produce_same_results(self):
        conductance_based_simulation = {
            "sim_time": 5_000,
            "sim_clock": 0.1 * ms,
            "g": 4,
            "g_ampa": 2.518667367869784e-06,
            "nu_ext_over_nu_thr": 1.88705,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            "panel": f"Compare single computation vs grid",
            "t_range": [[100, 120]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 3 * ms
        }
        experiment = Experiment(conductance_based_simulation)
        sim_and_plot(experiment)
        plt.close()

        g_s = [4]
        nu_ext_over_nu_thrs = [1.88705]

        conductance_based_simulation = {
            "sim_time": 5_000,
            "sim_clock": 0.1 * ms,
            "g_ampa": 2.518667367869784e-06,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            "panel": f"Compare single computation vs grid",
            "t_range": [[100, 120]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 3 * ms
        }
        experiment = Experiment(conductance_based_simulation)
        compare_g_s_vs_nu_ext_over_nu_thr(experiment, g_s, nu_ext_over_nu_thrs)

        plt.close()

if __name__ == '__main__':
    unittest.main()
