import random
from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

def sim(g, nu_ext_over_nu_thr, sim_time, rate_tick_step, in_testing=True):
    """
    g -- relative inhibitory to excitatory synaptic strength
    nu_ext_over_nu_thr -- ratio of external stimulus rate to threshold rate
    sim_time -- simulation time
    ax_spikes -- matplotlib axes to plot spikes on
    ax_rates -- matplotlib axes to plot rates on
    rate_tick_step -- step size for rate axis ticks
    """
    if in_testing:
        np.random.seed(0)

    # network parameters
    N_E = 10000
    gamma = 0.25
    N_I = round(gamma * N_E)
    N = N_E + N_I
    epsilon = 0.1
    C_E = epsilon * N_E
    C_ext = C_E

    # neuron parameters
    tau = 20 * ms
    theta = -40 * mV
    V_r = -55 * mV
    E_leak = -65 * mV
    tau_rp = 2 * ms

    # synapse parameters
    J = 0.5 * mV
    D = 1.5 * ms

    # external stimulus
    nu_thr = (theta - V_r) / (J * C_E * tau)
    #nu_thr = 10 * Hz

    defaultclock.dt = 0.05 * ms

    neurons = NeuronGroup(N,
                          """
                          dv/dt = -(v-E_leak)/tau: volt (unless refractory)                          """,
                          threshold="v > theta",
                          reset="v = V_r",
                          refractory=tau_rp,
                          method="exact",
                          )

    excitatory_neurons = neurons[:N_E]
    inhibitory_neurons = neurons[N_E:]

    exc_synapses = Synapses(excitatory_neurons, target=neurons, on_pre="v += J", delay=D)
    exc_synapses.connect(p=epsilon)

    inhib_synapses = Synapses(inhibitory_neurons, target=neurons, on_pre="v += -g*J", delay=D)
    inhib_synapses.connect(p=epsilon)

    nu_ext = nu_ext_over_nu_thr * nu_thr

    external_poisson_input = PoissonInput(
        target=neurons, target_var="v", N=C_ext, rate=nu_ext, weight=J
    )

    rate_monitor = PopulationRateMonitor(neurons)

    # record from the first 50 excitatory neurons
    spike_monitor = SpikeMonitor(neurons[:50])
    v_monitor = StateMonitor(source=neurons[N_E - 25: N_E + 25], variables="v", record=True)

    run(sim_time, report='text')

    fig = plt.figure(figsize=(10, 12))

    fig.suptitle(f''' {panel}, N = {N}, $N_E = {N_E}$, $N_I = {N_I}$, $\gamma={gamma}$
    $\\nu_T = {nu_thr}$, $\\frac{{\\nu_E}}{{\\nu_T}} = {nu_ext_over_nu_thr}$ ''')

    gs = fig.add_gridspec(ncols=1, nrows=3, height_ratios=[4, 1, 2])

    ax_spikes, ax_rates, ax_voltages = gs.subplots(sharex="col")

    ax_spikes.plot(spike_monitor.t / ms, spike_monitor.i, "|")
    ax_rates.plot(rate_monitor.t / ms, rate_monitor.rate / Hz)

    for i in range(0, 5):
        ax_voltages.plot(v_monitor.t / ms, v_monitor[i].v, label=f"Exc {i}"), range(0, 5)
        ax_voltages.plot(v_monitor.t / ms, v_monitor[25 + i].v, label=f"Inh {25 + i}"), range(0, 5)

    ax_spikes.set_yticks([])

    ax_spikes.set_xlim(*params["t_range"])
    ax_rates.set_xlim(*params["t_range"])

    ax_rates.set_ylim(*params["rate_range"])
    ax_rates.set_xlabel("t [ms]")
    ax_voltages.legend(loc="best")

    ax_rates.set_yticks(
        np.arange(
            params["rate_range"][0], params["rate_range"][1] + rate_tick_step, rate_tick_step
        )
    )

    plt.subplots_adjust(hspace=0)


test_parameters = {
    "D_1": {
        "g": 1,
        "nu_ext_over_nu_thr": 0.9,
        "t_range": [2800, 3000],
        "rate_range": [0, 250],
        "rate_tick_step": 50,
    },
    "D_2": {
        "g": 2,
        "nu_ext_over_nu_thr": 0.9,
        "t_range": [2800, 3000],
        "rate_range": [0, 250],
        "rate_tick_step": 50,
    },
    "D_3": {
        "g": 3,
        "nu_ext_over_nu_thr": 0.9,
        "t_range": [2800, 3000],
        "rate_range": [0, 250],
        "rate_tick_step": 50,
    },
    "D_4": {
        "g": 4,
        "nu_ext_over_nu_thr": 0.9,
        "t_range": [2800, 3000],
        "rate_range": [0, 250],
        "rate_tick_step": 50,
    },
    "D_5": {
        "g": 5,
        "nu_ext_over_nu_thr": 0.9,
        "t_range": [2800, 3000],
        "rate_range": [0, 250],
        "rate_tick_step": 50,
    }
}

test_parameters = dict(map(
    lambda nu_e_over_nu_th: (f"Scan {nu_e_over_nu_th}", {"g": 5,
        "nu_ext_over_nu_thr": nu_e_over_nu_th,
        "t_range": [2800, 3000],
        "rate_range": [0, 250],
        "rate_tick_step": 50}), np.arange(0.5, 10, step=0.5)
))

parameters = {
    "A": {
        "g": 3,
        "nu_ext_over_nu_thr": 2,
        "t_range": [100, 200],
        "rate_range": [0, 6000],
        "rate_tick_step": 1000,
    },
    "A_1": {
        "g": 30,
        "nu_ext_over_nu_thr": 2,
        "t_range": [100, 200],
        "rate_range": [0, 6000],
        "rate_tick_step": 1000,
    },
    "A_2": {
        "g": 100,
        "nu_ext_over_nu_thr": 2,
        "t_range": [100, 200],
        "rate_range": [0, 6000],
        "rate_tick_step": 1000,
    },
    "B": {
        "g": 6,
        "nu_ext_over_nu_thr": 4,
        "t_range": [100, 200],
        "rate_range": [0, 400],
        "rate_tick_step": 100,
    },
    "C": {
        "g": 5,
        "nu_ext_over_nu_thr": 2,
        "t_range": [100, 200],
        "rate_range": [0, 200],
        "rate_tick_step": 50,
    },
    "D": {
        "g": 4.5,
        "nu_ext_over_nu_thr": 0.9,
        "t_range": [1800, 2000],
        "rate_range": [0, 250],
        "rate_tick_step": 50,
    },
}

for panel, params in test_parameters.items():
    sim(
        g=params["g"],
        nu_ext_over_nu_thr=params["nu_ext_over_nu_thr"],
        sim_time=params["t_range"][1] * ms,
        rate_tick_step=params["rate_tick_step"],
    )
    plt.show()
