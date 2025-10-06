import unittest

from brian2 import second

from Configuration import Experiment


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



if __name__ == '__main__':
    unittest.main()
