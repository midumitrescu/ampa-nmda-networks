import unittest

from brian2 import ms

from Configuration import PlotParams

'''
 looking for stability for:
 
 AMPA/GABA only,. g_ampa = 2.625, nu_ext/nu_thr = 1.7 -> 1.8. Model goes from SI slow directly to S
'''

class MyTestCase(unittest.TestCase):

    def test_for_g_in_parameter_range(self):
        g_ampas = [2e-06, 2.5e-06, 2.625e-6, 2.75e-06, 3e-06]
        nu_thresholds = [1.5, 1.6, 1.7, 1.8]
        simulation = {
            "sim_time": 5_000,
            "sim_clock": 0.1 * ms,
            "g": 2,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_L": 0.00004,
            PlotParams.KEY_PANEL: r"Compare $g_\mathrm{AMPA}$ vs $\frac{\nu_\mathrm{Ext}}{\nu_\mathrm{Thr}}$  such that network is stable\\",
            "t_range": [[0, 500], [4000, 5000]],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 0.5 * ms
        }
        experiment = Experiment(simulation)
        compare_g_ampa_vs_nu_ext_over_nu_thr(experiment, g_ampas, nu_thresholds)
        # add assertion here


if __name__ == '__main__':
    unittest.main()
