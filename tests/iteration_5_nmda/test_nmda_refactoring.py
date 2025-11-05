import unittest

from Configuration import Experiment, NetworkParams
from iteration_5_nmda_refactored.network_with_nmda import sim, sim_and_plot

'''
We have 3 big issue that we are having an issue with:

1. the model is incorrect! we need a g_nmda_bar for the max gnmda available. However, we want to use same notation
as Wang. Otherwise this will get cumbersome and not understandable.
2. We want to break all components of the model to make it very easy to understand what is what
3. We want, if we introduce some extra variables in another model, for the simulate function to still work!
'''

extended_model = """
dv/dt = 1/C * (- I_L - I_syn - I_Exc - I_Inh - I_nmda): volt (unless refractory)
        
I_L = g_L * (v-E_leak): amp / meter ** 2
I_syn = g_e_syn * (v-E_ampa): amp / meter ** 2
I_Exc = g_e * (v-E_ampa): amp / meter ** 2
I_Inh = g_i * (v-E_gaba): amp / meter ** 2
I_nmda = g_nmda * (v - E_ampa): amp / meter** 2

dg_e_syn/dt = -g_e_syn / tau_ampa  : siemens / meter**2
dg_e/dt = -g_e / tau_ampa : siemens / meter**2
dg_i/dt = -g_i / tau_gaba  : siemens / meter**2

g_nmda = g_nmda_max * sigmoid_v * s_nmda: siemens / meter**2
g_nmda_max: siemens / meter**2

ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha * x_nmda * (1 - s_nmda) : 1
dx_nmda/dt = - x_nmda / tau_nmda_rise : 1

sigmoid_v = 1/(1 + exp(-0.062 * v/mvolt) * (MG_C/mmole / 3.57)) : 1
one_minus_s_nmda = 1 - s_nmda : 1
"""


class RefactoredNMDAInput(unittest.TestCase):
    def test_nmda_configuration_is_parsed(self):
        new_config = {
            "sim_time": 10,
            "record_N": 1,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["sigmoid_v", "x_nmda", "g_nmda", "I_nmda", "one_minus_s_nmda"],
            Experiment.KEY_SELECTED_MODEL: extended_model
        }

        object_under_test = Experiment(new_config)
        sim(experiment=object_under_test)

    def test_sim_and_plot(self):
        new_config = {
            "sim_time": 100,
            "record_N": 2,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["sigmoid_v", "x_nmda", "g_nmda", "I_nmda", "one_minus_s_nmda"],
            Experiment.KEY_SELECTED_MODEL: extended_model
        }

        object_under_test = Experiment(new_config)
        sim_and_plot(object_under_test)

    def test_sim_and_plot_with_external_input(self):
        config = {
            "sim_time": 4000,

            NetworkParams.KEY_NU_E_OVER_NU_THR: 1,
            NetworkParams.KEY_EPSILON: 0.1,
            "g": 4,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,

            Experiment.KEY_SELECTED_MODEL: extended_model,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["sigmoid_v", "x_nmda", "g_nmda", "I_nmda", "one_minus_s_nmda"],
            "record_N": 10,

            "t_range": [[0, 500], [0, 4000]],
        }
        sim_and_plot(Experiment(config))


if __name__ == '__main__':
    unittest.main()
