import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import mV, second, ms, Hz, have_same_dimensions, is_dimensionless

from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import rate_LIF_whitenoise, \
    integration_limits, SiegertGradientDescent, create_anneal_decay_schedule, plot_grad_descent
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, State
from iteration_8_compute_mean_steady_state.models_and_configs import palmer_experiment_0_1_Hz_with_NMDA_block, \
    palmer_experiment
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control

tau_m = 20 * ms
mu = -55 * mV
v_reset = -65 * mV
theta = -50 * mV
sigma = 5 * mV
tau_ref = 2 * ms

r_target = 0.3 * Hz

learning_rate=1e-3 * (mV * second)**2

def prepare_experiment_with_n_and_nu_nmda(N_nmda: int, nu_nmda: float, up_state_base, base) -> Experiment:
    up_state_base_local = up_state_base.copy()
    up_state_base_local[State.KEY_N_NMDA] = N_nmda
    up_state_base_local[State.KEY_NU_NMDA] = nu_nmda
    exp = base.with_property("up_state", up_state_base_local)
    return exp

class GradientDescentInNuNMDATestCases(unittest.TestCase):

    def test_(self):
        experiment = palmer_experiment
        up_state_base = palmer_experiment.network_params.up_state.params

        def objective(params):
            N_nmda, nu_nmda = params

            experiment_increment = prepare_experiment_with_n_and_nu_nmda(N_nmda=N_nmda, nu_nmda=nu_nmda, up_state_base=up_state_base, base=experiment)


            mean_soma_v = experiment_increment.effective_time_constant_up_state.E_0_with_nmda()
            variance_soma_v = experiment_increment.effective_time_constant_up_state.std_voltage_with_nmda()**2

            # Constraint 1: Must lie on the valid line
            line_error = sigma_full - (a * mu_full + b)

            # Constraint 2: Must give target rate
            rate = solver_full.firing_rate(mu_full, sigma_full)
            rate_error = rate - r_target_full

            return [line_error / mV, rate_error / Hz]

        # Use fsolve to find N, ν that satisfy both constraints
        from scipy.optimize import fsolve
        solution = fsolve(objective, [100, 10 * Hz])
