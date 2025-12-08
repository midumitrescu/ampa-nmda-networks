import unittest

from brian2 import mV, siemens, cm, mvolt

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import State, Experiment


class ConfigurationWithUpStateTestCases(unittest.TestCase):

    def test_N_and_gamma(self):
        up_state = {
            "N": 1_000,
            "gamma": 0.25

        }
        object_under_test = State(up_state)

        self.assertEqual(1_000, object_under_test.N)
        self.assertEqual(800, object_under_test.N_E)
        self.assertEqual(200, object_under_test.N_I)

    def test_N_E_and_gamma(self):
        up_state = {
            "N_E": 1_000,
            "gamma": 0.25

        }
        object_under_test = State(up_state)

        self.assertEqual(1_250, object_under_test.N)
        self.assertEqual(1000, object_under_test.N_E)
        self.assertEqual(250, object_under_test.N_I)

    def test_either_N_or_N_E_can_be_present_in_configuration(self):
        up_state = {
            "N": 1_000,
            "N_E": 800,
        }
        self.assertRaises(ValueError, lambda: State(up_state))

    def test_configuration_with_up_and_down_state_can_be_created(self):
        config = {
            "N": 1,
            "t_range": [[0, 10]],
            "up_state": {
                "N": 1_000,
                "gamma": 0.25
            },
            "down_state": {
                "N": 500,
                "gamma": 0.25
            }
        }
        object_under_test = Experiment(config)
        self.assertEqual(1_000, object_under_test.network_params.up_state.N)
        self.assertEqual(800, object_under_test.network_params.up_state.N_E)
        self.assertEqual(200, object_under_test.network_params.up_state.N_I)

        self.assertEqual(500, object_under_test.network_params.down_state.N)
        self.assertEqual(400, object_under_test.network_params.down_state.N_E)
        self.assertEqual(100, object_under_test.network_params.down_state.N_I)

        self.assertEqual(-65, object_under_test.effective_time_constant_up_state.E_0() / mV)
        self.assertEqual(0, object_under_test.effective_time_constant_up_state.mean_excitatory_conductance() / siemens * cm**2)
        self.assertEqual(0, object_under_test.effective_time_constant_up_state.mean_inhibitory_conductance() / siemens * cm**2)

    def test_effective_time_constant_up_and_down_state(self):
        config = {
            "N": 1,
            "t_range": [[0, 10]],

            "g": 1,
            "g_ampa": 2.40E-05,

            "up_state": {
                "N_E": 10_000,
                "gamma": 1,
                "nu": 100,
            },
            "down_state": {
                "N_E": 100,
                "gamma": 1,
                "nu": 10,
            }
        }
        object_under_test = Experiment(config)

        self.assertEqual(20_000, object_under_test.network_params.up_state.N)
        self.assertEqual(10_000, object_under_test.network_params.up_state.N_E)
        self.assertEqual(10_000, object_under_test.network_params.up_state.N_I)

        self.assertAlmostEqual(-40.01041233, object_under_test.effective_time_constant_up_state.E_0() / mV)

        self.assertAlmostEqual(4.8E-2, object_under_test.effective_time_constant_up_state.mean_excitatory_conductance() / siemens * cm**2)
        self.assertAlmostEqual(4.8E-2, object_under_test.effective_time_constant_up_state.mean_inhibitory_conductance() / siemens * cm**2)
        self.assertAlmostEqual(7.58947E-04,
                               object_under_test.effective_time_constant_up_state.std_excitatory_conductance() / siemens * cm ** 2)
        self.assertAlmostEqual(7.58947E-04,
                               object_under_test.effective_time_constant_up_state.std_inhibitory_conductance() / siemens * cm ** 2)
        self.assertAlmostEqual(0.4458682244,
                               object_under_test.effective_time_constant_up_state.std_voltage() / mV)


        self.assertEqual(200, object_under_test.network_params.down_state.N)
        self.assertEqual(100, object_under_test.network_params.down_state.N_E)
        self.assertEqual(100, object_under_test.network_params.down_state.N_I)

        self.assertAlmostEqual(-4.73529412E+01, object_under_test.effective_time_constant_down_state.E_0() / mV)

        self.assertAlmostEqual(4.8E-5,
                               object_under_test.effective_time_constant_down_state.mean_excitatory_conductance() / siemens * cm ** 2)
        self.assertAlmostEqual(4.8E-5,
                               object_under_test.effective_time_constant_down_state.mean_inhibitory_conductance() / siemens * cm ** 2)
        self.assertAlmostEqual(2.40E-05,
                               object_under_test.effective_time_constant_down_state.std_excitatory_conductance() / siemens * cm ** 2)
        self.assertAlmostEqual(2.40E-05,
                               object_under_test.effective_time_constant_down_state.std_inhibitory_conductance() / siemens * cm ** 2)
        self.assertAlmostEqual(4.693584178,
                               object_under_test.effective_time_constant_down_state.std_voltage() / mV)

if __name__ == '__main__':
    unittest.main()
