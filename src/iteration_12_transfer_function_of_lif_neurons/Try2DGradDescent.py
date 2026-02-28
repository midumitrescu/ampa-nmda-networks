import unittest

import numpy as np
from brian2 import mV, second, ms, Hz, have_same_dimensions
from scipy import special
from scipy.integrate import quad

from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import integration_limits, rate_LIF_whitenoise
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, State
from iteration_8_compute_mean_steady_state.models_and_configs import palmer_experiment
from utils import ExtendedDict

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

    nmda_block = ExtendedDict({
        "slope": -0.2514201066,
        "intercept": -12.5676230349
    })
    control = ExtendedDict({
        "slope": -0.2884429083,
        "intercept": -14.4092171897
    })

    '''
    NMDA Block (0.05 Hz): σ = -0.2514201066·μ + -12.5676230349
    Control (0.3 Hz): σ = -0.2884429083·μ + -14.4092171897
    '''
    def test_(self):
        experiment = palmer_experiment
        up_state_base = palmer_experiment.network_params.up_state.params
        r_target = 0.3 * Hz
        mu_target = -60 * mV

        include_mu = True

        def objective(params):
            N_nmda = 1
            nu_nmda=params[0]
            print(f"Check {N_nmda}, {nu_nmda}" )

            experiment_increment = prepare_experiment_with_n_and_nu_nmda(N_nmda=N_nmda, nu_nmda=nu_nmda, up_state_base=up_state_base, base=experiment)


            mean_soma_v = experiment_increment.effective_time_constant_up_state.E_0_with_nmda()
            sigma_soma_v = experiment_increment.effective_time_constant_up_state.std_voltage_with_nmda()

            # Constraint 1: Must lie on the valid line
            line_error = sigma_soma_v - (self.control.slope * mean_soma_v + self.control.intercept * mV)

            # Constraint 2: Must give target rate
            rate = rate_LIF_whitenoise(mu=mean_soma_v, sigma_v=sigma_soma_v,
                                       tau_membrane=experiment_increment.effective_time_constant_up_state.tau_eff_with_nmda(),
                                       theta=experiment_increment.neuron_params.theta, V_reset=experiment_increment.neuron_params.V_r,
                                       tau_ref=experiment_increment.neuron_params.tau_rp)
            rate_error = rate - r_target

            print(
                f"{N_nmda}, {nu_nmda: .5f}: mean [{mean_soma_v: .5f}], sigma [{sigma_soma_v: .5f}] producing rate [{rate: .5f}] ")
            if include_mu:
                mean_target = mean_soma_v - mu_target
                return [line_error / mV, rate_error / Hz, mean_target / mV]
            else:
                return [line_error / mV, rate_error / Hz]




        from scipy.optimize import least_squares

        solution = least_squares(objective, x0=[80.0], bounds=([0.0], [200.0]))

        print(solution)

    def test_soma_v(self):
        experiment = palmer_experiment
        up_state_base = palmer_experiment.network_params.up_state.params
        experiment_increment = prepare_experiment_with_n_and_nu_nmda(N_nmda=N_nmda, nu_nmda=nu_nmda,
                                                                     up_state_base=up_state_base, base=experiment)
