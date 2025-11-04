import unittest

from brian2 import second, siemens, cm

from Configuration import Experiment, NetworkParams


class ConfigurationParsingTestCases(unittest.TestCase):

    def test_error_should_be_raised_when_neither_sim_time_not_t_range_are_provided(self):
        self.assertRaises(ValueError, lambda: Experiment({}))

    def test_sim_time_is_correctly_extracted(self):
        sim_time_present = {
            "sim_time": 7000
        }

        object_under_test = Experiment(sim_time_present)
        self.assertEqual(7 * second, object_under_test.sim_time)

    def test_sim_time_is_extracted_before_time_range(self):
        sim_time_present = {
            "t_range": [0, 1000],
        }

        object_under_test = Experiment(sim_time_present)
        self.assertEqual(1 * second, object_under_test.sim_time)

    def test_when_g_gaba_not_present_then_g_gaba_should_be_computed(self):
        conductance_based_config = {
            "sim_time": 1,
            "g": 2,
            "g_ampa": 1e-05,
        }

        object_under_test = Experiment(conductance_based_config)
        self.assertEqual(1e-5 * siemens / (cm ** 2), object_under_test.synaptic_params.g_ampa)
        self.assertEqual(2e-5 * siemens / (cm ** 2), object_under_test.synaptic_params.g_gaba)

    def test_when_g_is_0_g_gaba_is_0(self):
        conductance_based_config = {
            "sim_time": 1,
            "g": 0,
            "g_ampa": 1e-05,
        }

        object_under_test = Experiment(conductance_based_config)
        self.assertEqual(1e-5 * siemens / (cm ** 2), object_under_test.synaptic_params.g_ampa)
        self.assertEqual(0 * siemens / (cm ** 2), object_under_test.synaptic_params.g_gaba)

    def test_copy_works(self):
        conductance_based_config = {
            "sim_time": 1,
            "g": 0,
            "g_ampa": 1e-05,
        }
        object_under_test = Experiment(conductance_based_config)
        object_under_test_2 = object_under_test.with_property(NetworkParams.KEY_G, 1)
        self.assertEqual(0, object_under_test.network_params.g)
        self.assertEqual(1, object_under_test_2.network_params.g)



if __name__ == '__main__':
    unittest.main()
