import unittest

import pytest
from brian2 import ms

from Configuration import Experiment, PlotParams
from iteration_5_nmda import developing_network_with_nmda
from iteration_5_nmda.network_with_nmda import sim_and_plot
import matplotlib


matplotlib.use('Agg')

class NMDAScripts(unittest.TestCase):
    def test_nmda_model_runs_and_plots(self):
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

            "record_N": 2,

            "g_L": 0.00004,

            "panel": self._testMethodName,
            "t_range": [[900, 1000]],
            "voltage_range": [-70, -30],

            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE]
        }

        experiment = Experiment(nmda_based_simulation)
        sim_and_plot(experiment)

    def test_sigmoid_no_scaling_no_concentration_scaling(self):
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

            "record_N": 2,

            "g_L": 0.00004,

            "panel": self._testMethodName,
            "t_range": [[0, 500]],
            "voltage_range": [-70, -30]
        }
        # scaling * (MG_C/mmole / 3.57)

        experiment = Experiment(nmda_based_simulation)
        developing_network_with_nmda.sim_and_plot(experiment, eq="""        
    dv/dt = 1/(100*ms)*(-41 * mV - v): volt
    dg_e/dt = -g_e / tau_ampa : siemens / meter**2
    dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
    dg_nmda/dt = -g_nmda / tau_nmda_decay + alpha * x * one_minus_g_nmda: siemens / meter**2
    dx/dt = - x / tau_nmda_rise :  siemens / meter**2
    sigmoid_v = 1/(1 + exp(-0.062 * v/mvolt)): 1
    one_minus_g_nmda = 1- g_nmda/siemens * meter**2 : 1
    I_nmda = g_nmda * sigmoid_v * (v-E_nmda): amp / meter**2
""")


    def test_sigmoid_no_scaling_w_concentration(self):
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

            "record_N": 2,

            "g_L": 0.00004,

            "panel": self._testMethodName,
            "t_range": [[0, 500]],
            "voltage_range": [-70, -30]
        }

        experiment = Experiment(nmda_based_simulation)
        developing_network_with_nmda.sim_and_plot(experiment, eq="""        
        dv/dt = 1/(100*ms)*(-41 * mV - v): volt
        dg_e/dt = -g_e / tau_ampa : siemens / meter**2
        dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
        dg_nmda/dt = -g_nmda / tau_nmda_decay + alpha * x * one_minus_g_nmda: siemens / meter**2
        dx/dt = - x / tau_nmda_rise :  siemens / meter**2
        sigmoid_v = 1/(1 + exp(-0.062 * v/mvolt) * (MG_C/mmole / 3.57)): 1
        one_minus_g_nmda = 1- g_nmda/siemens * meter**2 : 1
        I_nmda = g_nmda * sigmoid_v * (v-E_nmda): amp / meter**2
    """)

    @pytest.mark.skip(reason="not working")
    def test_clock_driven_dynamics(self):
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

            "record_N": 2,

            "g_L": 0.00004,

            "panel": self._testMethodName,
            "t_range": [[0, 500]],
            "voltage_range": [-70, -30]
        }

        experiment = Experiment(nmda_based_simulation)
        developing_network_with_nmda.sim_and_plot(experiment, eq="""        
        dv/dt = 1/(100*ms)*(-41 * mV - v): volt
        dg_e/dt = -g_e / tau_ampa : siemens / meter**2
        dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
        dg_nmda/dt = -g_nmda / tau_nmda_decay + alpha * x * one_minus_g_nmda: siemens / meter**2
        dx/dt = - x / tau_nmda_rise :  siemens / meter**2 (event-driven)
        sigmoid_v = 1/(1 + exp(-0.062 * beta* v/mvolt) * (MG_C/mmole / 3.57)): 1
        one_minus_g_nmda = 1- g_nmda/siemens * meter**2 : 1
        I_nmda = g_nmda * sigmoid_v * (v-E_nmda): amp / meter**2
    """)

    def test_sigmoid_scaled(self):
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

            "record_N": 2,

            "g_L": 0.00004,

            "panel": self._testMethodName,
            "t_range": [[0, 500]],
            "voltage_range": [-70, -30]
        }

        experiment = Experiment(nmda_based_simulation)
        developing_network_with_nmda.sim_and_plot(experiment, eq="""        
        dv/dt = 1/(100*ms)*(-41 * mV - v): volt
        dg_e/dt = -g_e / tau_ampa : siemens / meter**2
        dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
        dg_nmda/dt = -g_nmda / tau_nmda_decay + alpha * x * one_minus_g_nmda: siemens / meter**2
        dx/dt = - x / tau_nmda_rise :  siemens / meter**2
        sigmoid_v = 1/(1 + exp(-0.062 * (v/mvolt +34)) * (MG_C/mmole / 3.57)): 1
        one_minus_g_nmda = 1- g_nmda/siemens * meter**2 : 1
        I_nmda = g_nmda * sigmoid_v * (v-E_nmda): amp / meter**2
    """)


    def test_g_nmda_tot(self):
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

            "record_N": 2,

            "g_L": 0.00004,

            "panel": self._testMethodName,
            "t_range": [[0, 500]],
            "voltage_range": [-70, -30]
        }

        experiment = Experiment(nmda_based_simulation)
        developing_network_with_nmda.sim_and_plot(experiment, eq="""        
        dv/dt = 1/(100*ms)*(-41 * mV - v): volt
        dg_e/dt = -g_e / tau_ampa : siemens / meter**2
        dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
        dg_nmda/dt = -g_nmda / tau_nmda_decay + alpha * x * one_minus_g_nmda: siemens / meter**2
        dx/dt = - x / tau_nmda_rise :  siemens / meter**2
        sigmoid_v = 1/(1 + exp(-0.062 * (v/mvolt +34)) * (MG_C/mmole / 3.57)): 1
        one_minus_g_nmda = 1- g_nmda/siemens * meter**2 : 1
        I_nmda = g_nmda * sigmoid_v * (v-E_nmda): amp / meter**2
    """)

    def test_v_under_g_nmda_input(self):
        nmda_based_simulation = {

            "sim_time": 1000,
            "sim_clock": 0.1 * ms,
            "g": 1,
            "g_ampa": 2.518667367869784e-06,
            "g_gaba": 2.518667367869784e-06,
            "nu_ext_over_nu_thr": 0.025,

            "N_E": 10,
            "epsilon": 0.1,
            "C_ext": 100,

            "record_N": 2,

            "g_L": 0.00004,

            "panel": self._testMethodName,
            "t_range": [[0, 500]],
            "voltage_range": [-70, -30]
        }

        experiment = Experiment(nmda_based_simulation)
        developing_network_with_nmda.sim_and_plot(experiment, eq="""        
        dv/dt = 1/C * (-g_L * (v-E_leak) - g_e * (v-E_ampa) - g_i * (v-E_gaba) - g_nmda * sigmoid_v * (v-E_nmda)): volt (unless refractory)
        dg_e/dt = -g_e / tau_ampa : siemens / meter**2
        dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
        dg_nmda/dt = -g_nmda / tau_nmda_decay + alpha * x * one_minus_g_nmda: siemens / meter**2
        dx/dt = - x / tau_nmda_rise :  siemens / meter**2
        sigmoid_v = 1/(1 + exp(-0.062 * (v/mvolt +34)) * (MG_C/mmole / 3.57)): 1
        one_minus_g_nmda = 1- g_nmda/siemens * meter**2 : 1
        I_nmda = g_nmda * sigmoid_v * (v-E_nmda): amp / meter**2
    """)

    def test_compare_translated_vs_wang_nmda_activation(self):
        nmda_based_simulation = {

            "sim_time": 1000,
            "sim_clock": 0.1 * ms,
            "g": 1,
            "g_ampa": 2.518667367869784e-06,
            "g_gaba": 2.518667367869784e-06,
            "nu_ext_over_nu_thr": 0.025,

            "N_E": 10,
            "epsilon": 0.1,
            "C_ext": 100,

            "record_N": 2,

            "g_L": 0.00004,

            "panel": "Run with NMDA with higher activation",
            "t_range": [[0, 500]],
            "voltage_range": [-70, -30]
        }

        experiment_translated = Experiment(nmda_based_simulation)
        _, _, _, _, internal_state_translated = developing_network_with_nmda.sim_and_plot(experiment_translated, eq="""        
        dv/dt = 1/C * (-g_L * (v-E_leak) - g_e * (v-E_ampa) - g_i * (v-E_gaba) - g_nmda * sigmoid_v * (v-E_nmda)): volt (unless refractory)
        dg_e/dt = -g_e / tau_ampa : siemens / meter**2
        dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
        dg_nmda/dt = -g_nmda / tau_nmda_decay + alpha * x * one_minus_g_nmda: siemens / meter**2
        dx/dt = - x / tau_nmda_rise :  siemens / meter**2
        sigmoid_v = 1/(1 + exp(-0.062 * (v/mvolt +34)) * (MG_C/mmole / 3.57)): 1
        one_minus_g_nmda = 1- g_nmda/siemens * meter**2 : 1
        I_nmda = g_nmda * sigmoid_v * (v-E_nmda): amp / meter**2
    """)

        experiment_wang = experiment_translated.with_property(PlotParams.KEY_PANEL, "Run with NMDA model from Wang et al. 2002")

        _, _, _, _, internal_state_wang = developing_network_with_nmda.sim_and_plot(experiment_wang, eq="""        
                dv/dt = 1/C * (-g_L * (v-E_leak) - g_e * (v-E_ampa) - g_i * (v-E_gaba) - g_nmda * sigmoid_v * (v-E_nmda)): volt (unless refractory)
                dg_e/dt = -g_e / tau_ampa : siemens / meter**2
                dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
                dg_nmda/dt = -g_nmda / tau_nmda_decay + alpha * x * one_minus_g_nmda: siemens / meter**2
                dx/dt = - x / tau_nmda_rise :  siemens / meter**2
                sigmoid_v = 1/(1 + exp(-0.062 * v/mvolt) * (MG_C/mmole / 3.57)): 1
                one_minus_g_nmda = 1- g_nmda/siemens * meter**2 : 1
                I_nmda = g_nmda * sigmoid_v * (v-E_nmda): amp / meter**2
            """)

