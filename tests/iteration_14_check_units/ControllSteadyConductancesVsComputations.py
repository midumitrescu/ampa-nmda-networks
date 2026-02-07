import math
import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import mV, mmole, second, ms, Hz, nS, Mohm
from brian2.units.allunits import pampere, nsiemens
from joblib import Parallel, delayed
import pandas as pd

from BinarySeach import binary_search_for_target_value
from Plotting import show_plots_non_blocking, NeuronModelParams
from iteration_14_check_units.simulate_V_Clamp import current_clamp_model, simulate_current_injection, \
    run_current_injection_simulation, no_presynaptic_input
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams, \
    SynapticParams, CurrentClampParams
from iteration_7_one_compartment_step_input.models_and_configs import \
    single_compartment_without_nmda_deactivation_and_logged_variables
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import SimulationResults
from iteration_7_one_compartment_step_input.one_compartment_with_up_only import simulate_with_up_state_and_nmda, \
    sim_and_plot_up_with_state_and_nmda
from iteration_8_compute_mean_steady_state.grid_computations import \
    sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state, \
    parallelize_simulate_with_up_state_and_nmda
from iteration_8_compute_mean_steady_state.models_and_configs import palmer_experiment, \
    palmer_experiment_0_1_Hz_with_NMDA_block, wang_recurrent_config
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_and_plot_up_down, \
    plot_voltage_trace_comparisons, sim_steady_state

plt.rcParams['text.usetex'] = False

import copy

def with_properties(params: dict, values: dict):
    new_params = copy.deepcopy(params)
    for key, value in values.items():
        new_params[key] = value
    return Experiment(new_params)

class MyTestCase(unittest.TestCase):

    def test_r_in_computation_no_state(self):
        experiment = Experiment(wang_recurrent_config)
        up_state = experiment.network_params.up_state.params

        # compare increasing N to steady state for computing g_e

        g_e_simulations = Parallel(n_jobs=-1)(
            delayed(sim_steady_state)(experiment.with_properties({
                "up_state": with_properties(up_state, {"N": current_n})
            })) for current_n in range(1, 2000)
        )

        sim_steady_state(wang_recurrent_config)


if __name__ == '__main__':
    unittest.main()
