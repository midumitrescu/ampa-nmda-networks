from brian2 import *
import numpy as np
from scipy.optimize import root

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment

from loguru import logger

def solve(experiment: Experiment, v0 = None):
    if v0 is None:
        v0 = experiment.neuron_params.E_leak / mV

    g_L = experiment.neuron_params.g_L
    E_leak = experiment.neuron_params.E_leak

    g_ampa = experiment.synaptic_params.g_ampa
    g_gaba = experiment.synaptic_params.g_gaba
    g_x = experiment.synaptic_params.x_nmda
    g_nmda_max = experiment.synaptic_params.g_nmda

    E_ampa = experiment.synaptic_params.e_ampa
    E_gaba = experiment.synaptic_params.e_gaba
    E_nmda = experiment.synaptic_params.e_ampa

    MG_C = 1 * mmole  # extracellular magnesium concentration

    tau_ampa = experiment.synaptic_params.tau_ampa
    tau_gaba = experiment.synaptic_params.tau_gaba
    tau_nmda_rise = experiment.synaptic_params.tau_nmda_rise
    tau_nmda_decay = experiment.synaptic_params.tau_nmda_decay

    alpha = 0.5 * kHz  # saturation of NMDA channels at high presynaptic firing rates
    r_e = experiment.network_params.up_state.nu
    r_i = experiment.network_params.up_state.nu
    N_E = experiment.network_params.up_state.N_E
    N_I = experiment.network_params.up_state.N_I
    N_N = experiment.network_params.up_state.N_NMDA
    r_nmda = experiment.network_params.up_state.nu_nmda

    # Closed-form steady states
    g_e_0 = tau_ampa * g_ampa * N_E * r_e
    g_i_0 = tau_gaba * g_gaba * N_I * r_i

    x_nmda_0 = tau_nmda_rise * g_x * N_N * r_nmda
    s_nmda_0 = (alpha * x_nmda_0) / (alpha * x_nmda_0 + 1 / tau_nmda_decay)

    def sigmoid_v(v):
        return  1 / (1 + (MG_C / mmole) / 3.57 * np.exp(-0.062 * (v / mV)))

    def fixed_point_eq(v_mV):
        v = v_mV * mV

        g_nmda = g_nmda_max * sigmoid_v(v) * s_nmda_0
        logger.debug(f"Fixed point (v={v_mV.item(): .4f} mV, g_nmda={g_nmda.item() / nS: .4f} nS)")

        total_g = g_L + g_e_0 + g_i_0 + g_nmda

        v_rhs = (g_L * E_leak + g_e_0 * E_ampa + g_i_0 * E_gaba + g_nmda * E_nmda) / total_g

        logger.debug("returning {}", ((v - v_rhs) / mV, g_nmda.item() / nS))
        return (v - v_rhs) / mV


    sol = root(fixed_point_eq, v0)

    if not sol.success:
        raise RuntimeError(sol.message)

    v_inf = sol.x[0] * mV
    print("Fixed point voltage:", v_inf)

    return v_inf


