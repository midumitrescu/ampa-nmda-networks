import copy
import unittest
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from brian2 import ms, ufarad, cm, siemens

from Configuration import Experiment

some_params = {
                "panel": f"Testing matplotlib Text generation, sim clock = 0.05 ms",
                "g": 0,
                "sim_time": 1500,
                "sim_clock": 0.05 * ms,
                "nu_ext_over_nu_thr": 1,
                "t_range": [100, 120],
                "voltage_range": [-70, -30],
                "epsilon": 0,
                "C_ext": 1000,
                "smoothened_rate_width": 0.5*ms
            }

experiment = Experiment(some_params)

class MyTestCase(unittest.TestCase):
    def test_matplotlib_text_generation(self):

        #plt.plot()
        plt.figure(figsize=(10, 10))
        plt.title(experiment.gen_plot_title())
        #\\text\u007b cm \u007d^2
        plt.subplots_adjust(top=0.7)
        plt.show()

    def test_matplotlib_text_generation_for_g_L_004_siemens(self):

        plt.figure(figsize=(8, 8))

        config_for_siemens_magnitude = copy.copy(some_params)
        config_for_siemens_magnitude["g_L"] = 0.004 * siemens / cm**2
        object_under_test = Experiment(config_for_siemens_magnitude)
        plt.title(f"$g_L=0.004$ S - {object_under_test.gen_plot_title()}")
        #\\text\u007b cm \u007d^2
        plt.subplots_adjust(top=0.7)
        plt.show()


if __name__ == '__main__':
    unittest.main()
