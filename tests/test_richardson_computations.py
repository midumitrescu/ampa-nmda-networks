import unittest

from brian2 import second, siemens, cm, mV

from Configuration import Experiment, NetworkParams, PlotParams


class EffectiveTimeConstantComputationsTestCases(unittest.TestCase):

    # computed in Effective Reversal Computation
    def test_mean_conductance_computation_6000_pre_syn(self):
        config = {
            NetworkParams.KEY_N_E: 1,
            NetworkParams.KEY_C_EXT: 6000,
            NetworkParams.KEY_NU_E_OVER_NU_THR: 5 * 1e-2,

            "g": 1.5,
            "g_ampa": 2.4e-05,
            "g_gaba": 2.4e-05,
            "g_nmda": 0,

            "t_range": [[0, 10]],
        }

        object_under_test = Experiment(config)

        self.assertAlmostEqual(object_under_test.effective_timeconstant_estimation.mean_excitatory_conductance() / siemens * cm**2, 2.88E-2)
        self.assertAlmostEqual(object_under_test.effective_timeconstant_estimation.mean_inhibitory_conductance() / siemens * cm**2, 4.32E-2)
        self.assertAlmostEqual(object_under_test.effective_timeconstant_estimation.mean_total_conductance() / siemens * cm**2, 7.204E-2 )
        self.assertAlmostEqual(object_under_test.effective_timeconstant_estimation.E_0() / mV, -48.0094, places=4)

    def test_mean_conductance_computation_1000_pre_syn(self):
        config = {
            NetworkParams.KEY_N_E: 1,
            NetworkParams.KEY_C_EXT: 1000,
            NetworkParams.KEY_NU_E_OVER_NU_THR: 5 * 1e-2,

            "g": 1,
            "g_ampa": 2.4e-05,
            "g_gaba": 2.4e-05,
            "g_nmda": 0,

            "t_range": [[0, 10]],
        }

        object_under_test = Experiment(config)

        self.assertAlmostEqual(object_under_test.effective_timeconstant_estimation.mean_excitatory_conductance() / siemens * cm**2, 4.8E-3)
        self.assertAlmostEqual(object_under_test.effective_timeconstant_estimation.mean_inhibitory_conductance() / siemens * cm**2, 4.8E-3)
        self.assertAlmostEqual(object_under_test.effective_timeconstant_estimation.mean_total_conductance() / siemens * cm**2, 9.64000E-03)
        self.assertAlmostEqual(object_under_test.effective_timeconstant_estimation.E_0() / mV, -4.01037E+01, places=4)

    def test_std_voltage(self):
        config = {
            NetworkParams.KEY_N_E: 1,
            NetworkParams.KEY_C_EXT: 1000,
            NetworkParams.KEY_NU_E_OVER_NU_THR: 5 * 1e-2,

            "g": 1,
            "g_ampa": 2.4e-05,
            "g_gaba": 2.4e-05,
            "g_nmda": 0,

            "t_range": [[0, 10]],
        }

        object_under_test = Experiment(config)
        self.assertAlmostEqual(object_under_test.effective_timeconstant_estimation.std_excitatory_conductance() / siemens * cm**2, 2.4E-4)
        self.assertAlmostEqual(object_under_test.effective_timeconstant_estimation.std_inhibitory_conductance() / siemens * cm**2, 2.4E-4)
        self.assertAlmostEqual(object_under_test.effective_timeconstant_estimation.std_voltage() / mV, 1.373188624 )



if __name__ == '__main__':
    unittest.main()
