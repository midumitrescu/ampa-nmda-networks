import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from brian2 import ufarad, cm, siemens, mV, msecond

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from utils import ExtendedDict

plt.rcParams['text.usetex'] = True

def simulate_native(experiment: Experiment):
    if experiment.in_testing:
        np.random.seed(0)
        np.random.default_rng(0)

    dt = experiment.sim_clock / msecond
    T = experiment.sim_time / msecond
    steps = int(T / dt)

    C = experiment.neuron_params.C / (ufarad * (cm ** -2)) # micro Fahrad

    theta = experiment.neuron_params.theta / mV
    g_L = experiment.neuron_params.g_L / (siemens * (cm ** -2))
    E_leak = experiment.neuron_params.E_leak / mV
    V_r = experiment.neuron_params.V_r / mV

    g_ampa = experiment.synaptic_params.g_ampa / (siemens / cm ** 2)
    g_gaba = experiment.synaptic_params.g_gaba / (siemens / cm ** 2)
    g_nmda_max = experiment.synaptic_params.g_nmda / (siemens / cm ** 2)

    E_ampa = experiment.synaptic_params.e_ampa / mV
    E_gaba = experiment.synaptic_params.e_gaba / mV
    E_nmda = experiment.synaptic_params.e_ampa / mV

    MG_C = 1 # mmole, extracellular magnesium concentration

    tau_rp = experiment.neuron_params.tau_rp / msecond
    tau_ampa = experiment.synaptic_params.tau_ampa / msecond
    tau_gaba = experiment.synaptic_params.tau_gaba / msecond
    tau_nmda_decay = 100 #ms
    tau_nmda_rise = 2 # ms

    alpha = 0.5 # kHz  # saturation of NMDA channels at high presynaptic firing rates

    v = np.zeros(steps)
    g_e = np.zeros(steps)
    g_i = np.zeros(steps)
    s_nmda = np.zeros(steps)
    x_nmda = np.zeros(steps)
    I_L = np.zeros(steps)
    I_ampa = np.zeros(steps)
    I_gaba = np.zeros(steps)
    I_nmda = np.zeros(steps)
    spikes = []

    v[0] = E_leak
    g_e[0] = 0.0
    g_i[0] = 0.0
    s_nmda[0] = 0.0
    x_nmda[0] = 0.0

    # compute units of sigma e
    #sigma_e = np.sqrt(experiment.synaptic_params.tau_ampa * experiment.network_params.up_state.N_E * experiment.network_params.up_state.nu)
    #sigma_i = np.sqrt(experiment.synaptic_params.tau_gaba * experiment.network_params.up_state.N_I * experiment.network_params.up_state.nu)
    sigma_e = 0
    sigma_i = 0
    mean_ampa = experiment.effective_time_constant_up_state.mean_excitatory_conductance() / (siemens / cm ** 2)
    mean_gaba = experiment.effective_time_constant_up_state.mean_inhibitory_conductance() / (siemens / cm ** 2)

    for t in range(steps - 1):
        sigmoid_v = 1.0 / (1 + np.exp(-0.062 * (v[t] + 43)) * (MG_C / 3.57))
        g_nmda = g_nmda_max * sigmoid_v * s_nmda[t]

        I_L[t] = g_L * (v[t] - E_leak)
        I_ampa[t] = g_e[t] * (v[t] - E_ampa)
        I_gaba[t] = g_i[t] * (v[t] - E_gaba)
        I_nmda[t] = g_nmda * (v[t] - E_nmda)

        # --------------------------------
        # Conductances g_e, g_i + noise
        # --------------------------------
        g_e[t + 1] = g_e[t] + (dt * (-g_e[t] + mean_ampa) + sigma_e * np.sqrt(dt) * np.random.randn()) / tau_ampa
        g_i[t + 1] = g_i[t] + (dt * (-g_i[t] + mean_gaba) + sigma_i * np.sqrt(dt) * np.random.randn()) / tau_gaba

        # --------------------------------
        # NMDA dynamics
        # --------------------------------
        ds = -s_nmda[t] / tau_nmda_decay + alpha * x_nmda[t] * (1 - s_nmda[t])
        dx = -x_nmda[t] / tau_nmda_rise

        s_nmda[t + 1] = s_nmda[t] + dt * ds
        x_nmda[t + 1] = x_nmda[t] + dt * dx

        # --------------------------------
        # Membrane voltage update
        # --------------------------------
        #dv = (1 / C) * (-I_L - I_ampa - I_gaba - I_nmda) / 10**6
        dv = (1 / C) * (-I_L[t] - I_ampa[t] - I_gaba[t] - I_nmda[t])
        dv_step = dt * dv * 1000
        v[t + 1] = v[t] + dv_step
        #print(f"delta v = dt * dv =  {dv_step}")
        #print(f"v[{t+1}] = {v[t+1]: .5f}")
        if v[t + 1] >= theta:
            spikes.append(t * dt)
        if len(spikes) > 0:
            last_spike = spikes[-1]
            if t * dt <= last_spike + tau_rp:
                v[t+1] = V_r

    return ExtendedDict({
        "v": v,
        "g_e": g_e,
        "g_i": g_i,
        "s_nmda": s_nmda,
        "x_nmda": x_nmda,
        "experiment": experiment
    })

single_compartment_with_nmda = '''
dv/dt = 1/C * (- I_L - I_ampa - I_gaba - I_nmda): volt (unless refractory)

I_L = g_L * (v-E_leak): amp / meter ** 2

I_ampa = g_e * (v - E_ampa): amp / meter ** 2
I_gaba = g_i * (v - E_gaba): amp / meter ** 2
I_nmda = g_nmda * (v - E_nmda): amp / meter** 2

dg_e/dt = -g_e / tau_ampa : siemens / meter**2
dg_i/dt = -g_i / tau_gaba  : siemens / meter**2

g_nmda = g_nmda_max * sigmoid_v * s_nmda: siemens / meter**2
ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha * x_nmda * (1 - s_nmda) : 1
dx_nmda/dt = - x_nmda / tau_nmda_rise : 1

sigmoid_v = 1/(1 + exp(-0.062 * (v/mvolt + 43)) * (MG_C/mmole / 3.57)): 1
'''

single_compartment_with_nmda_and_logged_variables = f'''{single_compartment_with_nmda}

one_minus_s_nmda = 1 - s_nmda : 1
alpha_x_t = alpha * x_nmda: Hz
s_drive = alpha * x_nmda * (1 - s_nmda) : Hz
v_minus_e_gaba = v-E_gaba : volt
I_fast = I_ampa + I_gaba : amp / meter ** 2
'''
