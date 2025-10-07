import itertools
import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms, siemens, cm, second, Hz

from BinarySeach import binary_search_for_target_value
from Configuration import Experiment
from iteration_4_conductance_based_model.conductance_based_model import sim_and_plot, sim, produce_comparrison_plot


class GridTestCases(unittest.TestCase):

    def test_grid_composition(self):
        g_s = [0, 1]
        nu_ext_over_nu_thrs = [1.8, 1.85]

        conductance_based_simulation = {
            "sim_time": 100,
            "sim_clock": 0.1 * ms,
            "g": 0,
            "g_ampa": 2.518667367869784e-06,
            "nu_ext_over_nu_thr": 0,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,

            "panel": f"Scan $\\frac{{\\nu_E}}{{\\nu_T}}$ and g",
            "t_range": [[0, 50]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 5 * ms
        }


        experiment = Experiment(conductance_based_simulation)
        produce_comparrison_plot(experiment, g_s, nu_ext_over_nu_thrs)


if __name__ == '__main__':
    unittest.main()
