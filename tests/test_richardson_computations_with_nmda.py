import unittest

from brian2 import second, siemens, cm, mV, nsiemens

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_8_compute_mean_steady_state.test_wang_numbers import wang_recurrent_config


class EffectiveTimeConstantComputationsWithNMDATestCases(unittest.TestCase):

    def test_mean_g(self):
        object_under_test = Experiment(wang_recurrent_config)
        nmda = object_under_test.effective_time_constant_up_state.compute_mean_g_nmda()
        self.assertEqual(0.021982012163742008, nmda / nsiemens)


if __name__ == '__main__':
    unittest.main()
