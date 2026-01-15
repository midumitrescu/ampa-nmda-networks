import unittest

from brian2 import msiemens, cm, ms, siemens

from Configuration import Experiment


class Iteration4TestCases(unittest.TestCase):
    def test_tau_membrane_is_25_ms(self):
        tau_membrane_25_ms = {
            "panel": self._testMethodName,
            "sim_time": 150,
            "sim_clock": 0.05,
            "g": 0,
            "nu_ext_over_nu_thr": 1,
            "epsilon": 0,
            "C_ext": 1000,

            "g_L": 0.00004,

            "t_range": [100, 120],
            "voltage_range": [-70, -30]
        }

        object_under_test = Experiment(params=tau_membrane_25_ms)
        self.assertEqual(0.00004 * siemens * (cm ** -2), object_under_test.neuron_params.g_L)
        self.assertAlmostEqual(25, object_under_test.neuron_params.tau/ms)


if __name__ == '__main__':
    unittest.main()
