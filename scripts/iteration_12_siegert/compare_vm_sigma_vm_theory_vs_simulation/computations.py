import sys

import numpy as np
from brian2 import mV, second
from loguru import logger

from iteration_12_siegert.df_utils import prepare_experiment_with_N_tot
from iteration_12_transfer_function_of_lif_neurons.siegerts_formula_in_3_d import rate_LIF_whitenoise
from utils import ExtendedDict

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment


def compute_siegert_firing_rate(mu, sigma, tau_0_membrane, experiment: Experiment):
    return rate_LIF_whitenoise(mu / mV, tau_0_membrane / second, sigma / mV, experiment.neuron_params.theta / mV,
                               experiment.neuron_params.V_r / mV, experiment.neuron_params.tau_rp / second)


def mean_and_sigma(n, up_state_base: dict, base: Experiment):
    exp = prepare_experiment_with_N_tot(n, up_state_base, base)

    mu_no_nmda = exp.effective_time_constant_up_state.E_0()
    sigma_no_nmda = exp.effective_time_constant_up_state.std_voltage()
    tau_0_no_nmda = exp.neuron_params.C / (exp.effective_time_constant_up_state.mean_total_conductance())
    siegert_no_rate_without_nmda = compute_siegert_firing_rate(mu_no_nmda, sigma_no_nmda, tau_0_no_nmda, exp)

    mu_v_with_nmda = exp.effective_time_constant_up_state.E_0_with_nmda()
    sigma_v_with_nmda = exp.effective_time_constant_up_state.std_voltage_with_nmda()
    tau_0_with_nmda = exp.neuron_params.C / (exp.effective_time_constant_up_state.mean_total_conductance_with_nmda())
    siegert_firing_rate_with_nmda = compute_siegert_firing_rate(mu_v_with_nmda, sigma_v_with_nmda, tau_0_with_nmda, exp)

    return ExtendedDict({
        "N": n,
        "mu_v_no_nmda": mu_no_nmda / mV,
        "sigma_v_no_nmda": sigma_no_nmda / mV,
        "firing_rate_no_nmda": siegert_no_rate_without_nmda,
        "mu_v_with_nmda": mu_v_with_nmda / mV,
        "sigma_v_with_nmda": sigma_v_with_nmda / mV,
        "firing_rate_with_nmda": siegert_firing_rate_with_nmda
    })


def dr_over_d_mu(mean_v, mu):
    d_rate = np.diff(mean_v.to_numpy())
    d_mu = np.diff(mu.to_numpy())
    result = d_rate / d_mu
    return result

