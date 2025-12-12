from loguru import logger
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

from brian2 import ufarad, cm, siemens, mV, ms, uS, Hz, nS, msecond

from Plotting import show_plots_non_blocking
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import SimulationResults
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
        g_e[t + 1] = g_e[t] + dt * (-g_e[t] + mean_ampa) / tau_ampa \
                     + sigma_e * np.sqrt(dt) * np.random.randn()

        g_i[t + 1] = g_i[t] + dt * (-g_i[t] + mean_gaba) / tau_gaba \
                     + sigma_i * np.sqrt(dt) * np.random.randn()

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
        v[t + 1] = v[t] + dt * dv
        print(f"v[{t+1}] = {v[t+1]: .5f}")
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


def plot_simulation(results: dict):
    dt = results.experiment.sim_clock
    T = results.experiment.sim_time
    steps = int(T / dt)

    # ---------------------------------------------------------------------
    # Plot results
    # ---------------------------------------------------------------------
    time = np.arange(steps) * dt

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(time, results.v, label="normal scaling")
    plt.plot(time, results.v*10, label="10 scaling")
    plt.plot(time, results.v*100, label="100 scaling")
    plt.plot(time, results.v*1000, label="1000 scaling")
    plt.ylim([-70, -10])
    plt.ylabel("v (mV)")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time, results.g_e, label="g_e")
    plt.plot(time, results.g_i, label="g_i")
    plt.ylabel("conductance (S)")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time, results.s_nmda, label="s_nmda")
    plt.plot(time, results.x_nmda, label="x_nmda")
    plt.legend()

    plt.xlabel("time (s)")
    plt.tight_layout()
    show_plots_non_blocking()


def generate_title(experiment: Experiment):
    return fr"""{experiment.plot_params.panel}
    Up State: [{experiment.network_params.up_state.gen_plot_title()}, {experiment.effective_time_constant_up_state.gen_plot_title()}]
    Down State: [{experiment.network_params.down_state.gen_plot_title()}, {experiment.effective_time_constant_down_state.gen_plot_title()}]    
    Neuron: [$C={experiment.neuron_params.C * cm ** 2}$, $g_L={experiment.neuron_params.g_L * cm ** 2}$, $\theta={experiment.neuron_params.theta}$, $V_R={experiment.neuron_params.V_r}$, $E_L={experiment.neuron_params.E_leak}$, $\tau_M={experiment.neuron_params.tau}$, $\tau_{{\mathrm{{ref}}}}={experiment.neuron_params.tau_rp}$]
    Synapse: [$g_{{\mathrm{{AMPA}}}}={experiment.synaptic_params.g_ampa * (cm ** 2):.2f}$, $g_{{\mathrm{{GABA}}}}={experiment.synaptic_params.g_gaba * (cm ** 2) / nS:.2f}\,n\mathrm{{S}}$, $g={experiment.network_params.g}$, $g_{{\mathrm{{NMDA}}}}={experiment.synaptic_params.g_nmda * (cm ** 2) / nS:.2f}\,n\mathrm{{S}}$]"""


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
