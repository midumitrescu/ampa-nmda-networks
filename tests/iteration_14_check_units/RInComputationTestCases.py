import unittest

import matplotlib.pyplot as plt
from brian2 import mV
from brian2.units.allunits import nsiemens

from iteration_14_check_units.simulate_V_Clamp import no_presynaptic_input
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, CurrentClampParams
from iteration_8_compute_mean_steady_state.models_and_configs import wang_recurrent_config
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_steady_state

plt.rcParams['text.usetex'] = False


class MyTestCase(unittest.TestCase):

    def test_r_in_computation_no_state(self):
        experiment_under_test = Experiment(wang_recurrent_config)
        object_under_test = sim_steady_state(experiment_under_test)
        self.assertEqual(40, object_under_test.r_in)
        self.assertEqual(-70, object_under_test.v_steady)
        self.assertEqual(0, object_under_test.g_e_steady)
        self.assertEqual(0, object_under_test.g_i_steady)
        self.assertEqual(0, object_under_test.g_nmda_steady)


    def test_r_in_computation_no_current(self):
        experiment_under_test = Experiment(wang_recurrent_config)
        object_under_test = sim_steady_state(experiment_under_test, no_presynaptic_input)

        # why is that the case?
        print(1/experiment_under_test.neuron_params.g_L)
        # we have to
        self.assertEqual(40, object_under_test.r_in)
        self.assertEqual(-70, object_under_test.v_steady)
        self.assertEqual(0, object_under_test.g_e_steady)
        self.assertEqual(0, object_under_test.g_i_steady)
        self.assertEqual(0, object_under_test.g_nmda_steady)

    def test_r_in_computation_10_pA(self):
        experiment_under_test = Experiment(wang_recurrent_config).with_property(CurrentClampParams.KEY_I_INJECTED, 10)
        object_under_test = sim_steady_state(experiment_under_test, no_presynaptic_input)

        # we have to
        self.assertEqual(39.99999999972359, object_under_test.r_in)
        self.assertEqual(-69.60000000000277, object_under_test.v_steady)
        self.assertEqual(0, object_under_test.g_e_steady)
        self.assertEqual(0, object_under_test.g_i_steady)
        self.assertEqual(0, object_under_test.g_nmda_steady)

    def test_r_should_compute_r_in_when_p_is_10_pA(self):
        experiment_under_test = Experiment(wang_recurrent_config)
        previously_computed_steady_state = sim_steady_state(experiment_under_test, no_presynaptic_input)

        experiment_under_test = Experiment(wang_recurrent_config).with_property(CurrentClampParams.KEY_I_INJECTED, 10)
        object_under_test = sim_steady_state(experiment_under_test, no_presynaptic_input).recompute_r_in(previously_computed_steady_state)

        # we have to
        self.assertAlmostEqual(39.99999999972359, object_under_test.r_in)
        self.assertEqual(-69.60000000000277, object_under_test.v_steady)
        self.assertEqual(0, object_under_test.g_e_steady)
        self.assertEqual(0, object_under_test.g_i_steady)
        self.assertEqual(0, object_under_test.g_nmda_steady)

    def test_r_should_compute_r_in_when_p_is_20_pA_and_dv_should_double(self):
        experiment_under_test = Experiment(wang_recurrent_config)
        previously_computed_steady_state = sim_steady_state(experiment_under_test, no_presynaptic_input)

        experiment_under_test = Experiment(wang_recurrent_config).with_property(CurrentClampParams.KEY_I_INJECTED, 20)
        object_under_test = sim_steady_state(experiment_under_test, no_presynaptic_input)
        object_under_test.recompute_r_in(previously_computed_steady_state)

        # we have to
        self.assertAlmostEqual(40, object_under_test.r_in)
        self.assertEqual(-69.20000000000277, object_under_test.v_steady)
        self.assertEqual(0, object_under_test.g_e_steady)
        self.assertEqual(0, object_under_test.g_i_steady)
        self.assertEqual(0, object_under_test.g_nmda_steady)

    def test_r_should_compute_r_in_when_p_is_50_pA_and_dv_should_double_be_plus_2_mV(self):
        experiment_under_test = Experiment(wang_recurrent_config)
        previously_computed_steady_state = sim_steady_state(experiment_under_test, no_presynaptic_input)

        experiment_under_test = Experiment(wang_recurrent_config).with_property(CurrentClampParams.KEY_I_INJECTED, 50)
        object_under_test = sim_steady_state(experiment_under_test, no_presynaptic_input, steady_results_for_r_in=previously_computed_steady_state)

        # we have to
        self.assertAlmostEqual(40, object_under_test.r_in)
        self.assertAlmostEqual(-68, object_under_test.v_steady)
        self.assertEqual(0, object_under_test.g_e_steady)
        self.assertEqual(0, object_under_test.g_i_steady)
        self.assertEqual(0, object_under_test.g_nmda_steady)

    def test_r_should_compute_r_in_when_p_is_50_pA_and_dv_should_double_be_minus_2_mV(self):
        experiment_under_test = Experiment(wang_recurrent_config).with_property(CurrentClampParams.KEY_I_INJECTED, -50)
        object_under_test = sim_steady_state(experiment_under_test, no_presynaptic_input)

        # we have to
        self.assertAlmostEqual(40, object_under_test.r_in)
        self.assertAlmostEqual(-72, object_under_test.v_steady)
        self.assertEqual(0, object_under_test.g_e_steady)
        self.assertEqual(0, object_under_test.g_i_steady)
        self.assertEqual(0, object_under_test.g_nmda_steady)

    def test_steady_state_for_up_state(self):
        experiment_under_test = Experiment(wang_recurrent_config)
        object_under_test = sim_steady_state(experiment_under_test, state=experiment_under_test.network_params.up_state)

        # we have to
        self.assertAlmostEqual(20.398919510752673, object_under_test.r_in)
        self.assertEqual(-48.75341763069946, object_under_test.v_steady)
        self.assertAlmostEqual(16, object_under_test.g_e_steady)
        self.assertAlmostEqual(8, object_under_test.g_i_steady)
        self.assertAlmostEqual(0.02220431199227894, object_under_test.g_nmda_steady)

        print(f"Membrane voltage delta: {object_under_test.v_steady - experiment_under_test.effective_time_constant_up_state.E_0() / mV}")
        print(f"Delta to effective time constant: {object_under_test.g_nmda_steady - experiment_under_test.effective_time_constant_up_state.compute_mean_g_nmda() / nsiemens}" )






if __name__ == '__main__':
    unittest.main()
