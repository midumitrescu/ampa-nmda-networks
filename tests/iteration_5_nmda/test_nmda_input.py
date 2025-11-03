import unittest

from brian2 import ms

from Configuration import Experiment
from iteration_5_nmda import developing_network_with_nmda
from iteration_5_nmda.network_with_nmda import sim_and_plot, sim, wang_model, translated_model, wang_model_extended, \
    translated_model_extended


class MyTestCase(unittest.TestCase):
    def test_nmda_configuration_is_parsed(self):
        new_config = {
            "sim_time": 10,
            "record_N": 1,
        }

        object_under_test = Experiment(new_config)

        self.assertEqual(1, object_under_test.network_params.neurons_to_record)

    def test_sensible_defaults_for_new_parameters(self):
        new_config = {
            "sim_time": 10,
        }

        object_under_test = Experiment(new_config)

        self.assertTrue(object_under_test.plot_params.plot_smoothened_rate)
        self.assertEqual(25, object_under_test.network_params.neurons_to_record)

    def test_nmda_model_runs(self):
        nmda_based_simulation = {

            "sim_time": 100,
            "sim_clock": 0.05 * ms,

            "g": 0,
            "N_E": 10,
            "nu_ext_over_nu_thr": 1,
            "epsilon": 0.1,
            "C_ext": 10,

            "panel": f"Test model runs with default equation",
            "t_range": [80, 100],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 0.5 * ms
        }

        experiment = Experiment(nmda_based_simulation)
        sim(experiment)

    def test_nmda_model_runs_and_plots(self):
        nmda_based_simulation = {

            "sim_time": 100,
            "sim_clock": 0.05 * ms,

            "g": 0,
            "N_E": 10,
            "nu_ext_over_nu_thr": 1,
            "epsilon": 0.1,
            "C_ext": 10,

            "panel": f"Test model runs with default equation",
            "t_range": [80, 100],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 0.5 * ms,
            "record_N": 1,
        }

        experiment = Experiment(nmda_based_simulation)
        sim_and_plot(experiment)

    def test_unit_for_x_variable(self):
        new_config = {
            "sim_time": 10
        }

        experiment = Experiment(new_config)
        sim_and_plot(experiment, eq="""
            dv/dt = 1/C * (-g_L * (v-E_leak)) : volt (unless refractory)
            dg_e/dt = -g_e / tau_ampa : siemens / meter**2
            dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
            dg_nmda /dt = -g_nmda / tau_nmda_decay: siemens / meter**2
            dx/dt = - x / tau_nmda_rise : siemens / meter**2
        """)
        # units w and x must have same units, due to x += w on pre (jumps) => x must be siemens / meter**2
        #  sigmoid_v = 1/(1 + exp(-0.062 * v/mvolt)) * (MG_C/mmole / 3.57): 1

    def test_sigmoid_is_unitless(self):
        new_config = {
            "sim_time": 10
        }

        experiment = Experiment(new_config)
        sim(experiment, eq="""
            dv/dt = 1/C * (-g_L * (v-E_leak)) : volt (unless refractory)
            dg_e/dt = -g_e / tau_ampa : siemens / meter**2
            dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
            dg_nmda /dt = -g_nmda / tau_nmda_decay: siemens / meter**2
            dx/dt = - x / tau_nmda_rise : siemens / meter**2
            sigmoid_v = 1/(1 + exp(-0.062 * v/mvolt)) * (MG_C/mmole / 3.57): 1
        """)

    def test_unit_for_I_ampa_is_correct(self):
        new_config = {
            "sim_time": 10
        }

        experiment = Experiment(new_config)
        sim(experiment, eq="""
            dv/dt = 1/C * (-g_L * (v-E_leak) - g_nmda * sigmoid_v * (v-E_nmda)): volt (unless refractory)
            dg_e/dt = -g_e / tau_ampa : siemens / meter**2
            dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
            dg_nmda /dt = -g_nmda / tau_nmda_decay: siemens / meter**2
            dx/dt = - x / tau_nmda_rise : siemens / meter**2
            sigmoid_v = 1/(1 + exp(-0.062 * v/mvolt)) * (MG_C/mmole / 3.57): 1
        """)


    def test_unit_g_nmda_is_correct(self):
        new_config = {
            "sim_time": 10
        }

        experiment = Experiment(new_config)
        sim(experiment, eq="""
            dv/dt = 1/C * (-g_L * (v-E_leak) - g_nmda * sigmoid_v * (v-E_nmda)): volt (unless refractory)
            dg_e/dt = -g_e / tau_ampa : siemens / meter**2
            dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
            dg_nmda/dt = -g_nmda / tau_nmda_decay + alpha * x * one_minus_g_ampa: siemens / meter**2
            dx/dt = - x / tau_nmda_rise : siemens / meter**2 
            sigmoid_v = 1/(1 + exp(-0.062 * v/mvolt)) * (MG_C/mmole / 3.57): 1
            one_minus_g_ampa = 1- g_nmda/siemens * meter**2 : 1
        """)

    def test_wang_model_runs_successfully(self):
        new_config = {
            "sim_time": 10,
            "model": wang_model
        }

        experiment = Experiment(new_config)
        sim(experiment)

    def test_translated_model_runs_successfully(self):
        new_config = {
            "sim_time": 10,
            "model": translated_model
        }

        experiment = Experiment(new_config)
        sim(experiment)

    def test_all_saturation_variables_are_interval_0_1_wang_model(self):
        nmda_based_simulation = {

            "sim_time": 1000,
            "sim_clock": 0.1 * ms,
            "g": 1,
            "g_ampa": 2.518667367869784e-06,
            "g_gaba": 2.518667367869784e-06,
            "nu_ext_over_nu_thr": 0.5,

            "N_E": 10,
            "epsilon": 0.1,
            "C_ext": 100,

            "record_N": 1,

            "g_L": 0.00004,
            "panel": self._testMethodName,
        }

        _, _, _, _, internal_state = developing_network_with_nmda.sim(Experiment(nmda_based_simulation), eq="""        
                        dv/dt = 1/C * (-g_L * (v-E_leak) - g_e * (v-E_ampa) - g_i * (v-E_gaba) - g_nmda * sigmoid_v * (v-E_nmda)): volt (unless refractory)
                        dg_e/dt = -g_e / tau_ampa : siemens / meter**2
                        dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
                        dg_nmda/dt = -g_nmda / tau_nmda_decay + alpha * x * one_minus_g_nmda: siemens / meter**2
                        dx/dt = - x / tau_nmda_rise :  siemens / meter**2
                        sigmoid_v = 1/(1 + exp(-0.062 * v/mvolt) * (MG_C/mmole / 3.57)): 1
                        one_minus_g_nmda = 1- g_nmda/siemens * meter**2 : 1
                        I_nmda = g_nmda * sigmoid_v * (v-E_nmda): amp / meter**2
                    """)

        self.assertTrue((internal_state.one_minus_g_nmda[:] >= 0).all())
        self.assertTrue((internal_state.one_minus_g_nmda[:] <= 1).all())

    def test_all_saturation_variables_are_interval_0_1_translated_model(self):
        nmda_based_simulation = {

            "sim_time": 1000,
            "sim_clock": 0.1 * ms,
            "g": 1,
            "g_ampa": 2.518667367869784e-06,
            "g_gaba": 2.518667367869784e-06,
            "nu_ext_over_nu_thr": 0.5,

            "N_E": 10,
            "epsilon": 0.1,
            "C_ext": 100,

            "record_N": 1,

            "g_L": 0.00004,
            "panel": self._testMethodName,
        }

        _, _, _, _, internal_state = developing_network_with_nmda.sim(Experiment(nmda_based_simulation), eq="""        
                        dv/dt = 1/C * (-g_L * (v-E_leak) - g_e * (v-E_ampa) - g_i * (v-E_gaba) - g_nmda * sigmoid_v * (v-E_nmda)): volt (unless refractory)
                        dg_e/dt = -g_e / tau_ampa : siemens / meter**2
                        dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
                        dg_nmda/dt = -g_nmda / tau_nmda_decay + alpha * x * one_minus_g_nmda: siemens / meter**2
                        dx/dt = - x / tau_nmda_rise :  siemens / meter**2
                        sigmoid_v = 1/(1 + exp(-0.062 * (v/mvolt + 43)) * (MG_C/mmole / 3.57)): 1
                        one_minus_g_nmda = 1- g_nmda/siemens * meter**2 : 1
                        I_nmda = g_nmda * sigmoid_v * (v-E_nmda): amp / meter**2
                    """)

        self.assertTrue((internal_state.one_minus_g_nmda[:] >= 0).all())
        self.assertTrue((internal_state.one_minus_g_nmda[:] <= 1).all())

    def test_simple_models_are_correctly_written(self):
        nmda_based_simulation = {

            "sim_time": 1000,
            "sim_clock": 0.1 * ms,
            "g": 1,
            "g_ampa": 2.518667367869784e-06,
            "g_gaba": 2.518667367869784e-06,
            "nu_ext_over_nu_thr": 0.5,

            "N_E": 10,
            "epsilon": 0.1,
            "C_ext": 100,

            "record_N": 1,

            "g_L": 0.00004,
            "hidden_variables_to_record": ["sigmoid_v", "x", "g_nmda"],
            "panel": self._testMethodName,
        }
        experiment = Experiment(nmda_based_simulation)

        for model in [wang_model, translated_model]:
            current = experiment.with_property(Experiment.KEY_SELECTED_MODEL, model)
            sim(current)

    def test_extended_models_are_correctly_written(self):
        nmda_based_simulation = {

            "sim_time": 1000,
            "sim_clock": 0.1 * ms,
            "g": 1,
            "g_ampa": 2.518667367869784e-06,
            "g_gaba": 2.518667367869784e-06,
            "nu_ext_over_nu_thr": 0.5,

            "N_E": 10,
            "epsilon": 0.1,
            "C_ext": 100,

            "record_N": 1,

            "g_L": 0.00004,
            "panel": self._testMethodName,
        }
        experiment = Experiment(nmda_based_simulation)
        for model in [wang_model_extended, translated_model_extended]:
            current = experiment.with_property(Experiment.KEY_SELECTED_MODEL, model)
            sim(current)





if __name__ == '__main__':
    unittest.main()
