from brian2 import *
from matplotlib import gridspec

from Configuration import Experiment

'''
Simulation params
g: relative inhibitory to excitatory synaptic strength. g_I / g_E
C: Each neuron receives C randomly chosen connections from other neurons in the network, from which. C = epsilon N
C E = epsilon * N E 
C I = epsilon * N I 
C_ext: It also receives C_ext connections from excitatory neurons outside the network. 

Each neuron gets C_E, C_I, C_EXT connections

'''

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

default_model = """        
    dv/dt = - 1/C * g_e * (v-E_ampa) : volt
    dg_e/dt = -g_e / tau_ampa : siemens / meter**2
    dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
"""

def sim_and_plot(experiment: Experiment, in_testing=True, eq = default_model):
    rate_monitor, spike_monitor, v_monitor, g_monitor = sim(experiment, in_testing, eq)
    plot_simulation(experiment, rate_monitor,
                    spike_monitor, v_monitor, g_monitor)

    return rate_monitor, spike_monitor, v_monitor, g_monitor


def sim(experiment: Experiment, in_testing=True, eq = default_model):
    """
    g --
    nu_ext_over_nu_thr -- ratio of external stimulus rate to threshold rate
    sim_time -- simulation time
    ax_spikes -- matplotlib axes to plot spikes on
    ax_rates -- matplotlib axes to plot rates on
    rate_tick_step -- step size for rate axis ticks
    """
    if in_testing:
        np.random.seed(0)
    defaultclock.dt = 0.05 * ms

    C = experiment.neuron_params.C

    theta = experiment.neuron_params.theta
    g_L = experiment.neuron_params.g_L
    E_leak = experiment.neuron_params.E_leak
    V_r = experiment.neuron_params.V_r
    J = experiment.synaptic_params.J


    g = experiment.network_params.g
    g_ampa = experiment.synaptic_params.g_ampa
    g_gaba = experiment.synaptic_params.g_gaba

    tau_ampa = 2 * ms
    tau_gaba = 2 * ms

    E_ampa = experiment.synaptic_params.e_ampa
    E_gaba = experiment.synaptic_params.e_gaba

    # dv/dt = - 1/C * g_L * (v-E_leak) - 1/C * g_e * (v-E_ampa) - 1/C * g_i *  (v-E_gaba) : volt (unless refractory)
    # dv/dt = - 1/C * g_L * (v-E_leak) - 1/C * g_e * (v-E_ampa) - 1/C * g_i *  (v-E_gaba) : volt (unless refractory)
    # dg_e/dt = - 1/tau_ampa * g_e: siemens / meter**2

    neurons = NeuronGroup(experiment.network_params.N,
                          eq,
                          threshold="v >= theta",
                          reset="v = V_r",
                          refractory=experiment.neuron_params.tau_rp,
                          method="euler")
    neurons.v[:] = -65 * mV

    excitatory_neurons = neurons[:experiment.network_params.N_E]
    inhibitory_neurons = neurons[experiment.network_params.N_E:]

    print(g_ampa)

    exc_synapses = Synapses(excitatory_neurons, target=neurons, on_pre="g_e += g_ampa",
                            delay=experiment.synaptic_params.D)
    exc_synapses.connect(p=experiment.network_params.epsilon)

    inhib_synapses = Synapses(inhibitory_neurons, target=neurons, on_pre="g_i += g*g_ampa",
                              delay=experiment.synaptic_params.D)
    inhib_synapses.connect(p=experiment.network_params.epsilon)

    external_poisson_input = PoissonInput(
        target=neurons, target_var="g_e", N=experiment.network_params.C_ext, rate=experiment.nu_ext,
        weight=experiment.synaptic_params.g_ampa
    )

    rate_monitor = PopulationRateMonitor(neurons)

    # record from the first 50 excitatory neurons
    spike_monitor = SpikeMonitor(neurons[experiment.network_params.N_E - 25: experiment.network_params.N_E + 25])
    v_monitor = StateMonitor(source=neurons[experiment.network_params.N_E - 25: experiment.network_params.N_E + 25],
                             variables="v", record=True)

    g_monitor = StateMonitor(source=neurons[experiment.network_params.N_E - 25: experiment.network_params.N_E + 25], variables=["g_e", "g_i"], record=True)

    run(experiment.sim_time, report='text')

    return rate_monitor, spike_monitor, v_monitor, g_monitor


def plot_v_line(experiment: Experiment, ax_voltages: Axes, v_monitor: StateMonitor, spike_monitor: SpikeMonitor,
                i: int) -> None:
    lines = ax_voltages.plot(v_monitor.t / ms, v_monitor[i].v / mV, label=f"Neuron {i} - {'Exc' if i <= 25 else 'Inh'}",
                             lw=1)
    color = lines[0].get_color()
    spike_times_current_neuron = spike_monitor.all_values()['t'][i] / ms

    v_min_plot, v_max_plot = find_v_min_and_v_max_for_plotting(experiment, v_monitor)

    ax_voltages.vlines(x=spike_times_current_neuron, ymin=v_min_plot, ymax=v_max_plot, color=color, linestyle="-.",
                       label=f"Neuron {i} Spike Time", lw=0.8)

def plot_simulation(experiment: Experiment, rate_monitor,
                    spike_monitor, v_monitor, g_monitor):

    params_t_range = experiment.plot_params.t_range

    if isinstance(params_t_range[0], list):
        for time_slot in params_t_range:
            plot_simulation_in_one_time_range(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor, time_range = time_slot)
    else:
        plot_simulation_in_one_time_range(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor, time_range=params_t_range)

def plot_simulation_in_one_time_range(experiment: Experiment, rate_monitor,
                    spike_monitor, v_monitor, g_monitor, time_range=[100, 200]):

    rate_tick_step = experiment.plot_params.rate_tick_step
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(
        f''' {experiment.plot_params.panel}, N = {experiment.network_params.N}, $N_E = {experiment.network_params.N_E}$, $N_I = {experiment.network_params.N_I}$, $\gamma={experiment.network_params.gamma}$
    $\\nu_T = {experiment.nu_thr}$, $\\frac{{\\nu_E}}{{\\nu_T}} = {experiment.nu_ext_over_nu_thr: .3f}$ , g={experiment.network_params.g}''')

    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5, 2])
    raster_and_population = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], height_ratios=[4, 1],
                                                             hspace=0)
    voltage_and_g_s_examples = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.8)

    ax_spikes, ax_rates = raster_and_population.subplots(sharex="col")
    ax_spikes.plot(spike_monitor.t / ms, spike_monitor.i, "|")
    ax_rates.plot(rate_monitor.t / ms, rate_monitor.rate / Hz)
    ax_spikes.set_yticks([])
    # ax_rates.set_ylim(*experiment.plot_params.rate_range)
    ax_rates.set_ylim([0, np.max(rate_monitor.rate / Hz)])

    ax_voltages, ax_g_s = voltage_and_g_s_examples.subplots(sharex="col")

    ax_voltages.axhline(y=experiment.neuron_params.theta / ms, linestyle="dotted", linewidth="0.3", color="k",
                        label="$\\theta$")
    v_min_plot, v_max_plot = find_v_min_and_v_max_for_plotting(experiment, v_monitor)

    ax_voltages.set_ylim([v_min_plot, v_max_plot])

    for i in [0, 2, 26, 28]:
        plot_v_line(experiment, ax_voltages, v_monitor, spike_monitor, i)

    for ax in [ax_spikes, ax_rates, ax_voltages, ax_g_s]:
        ax.set_xlim(*time_range)

    ax_voltages.legend(loc="best")
    ax_voltages.set_xlabel("t [ms]")
    ax_voltages.set_ylabel("v [mV]")
    # ax_rates.set_yticks(
    #     np.arange(
    #         experiment.plot_params.rate_range[0], experiment.plot_params.rate_range[1] + rate_tick_step, rate_tick_step
    #     )
    # )

    ax_g_s.plot(g_monitor.t / ms, g_monitor[0].g_i, label="$g_i$[0]")
    ax_g_s.plot(g_monitor.t / ms, g_monitor[0].g_e, label="$g_e$[0]")
    ax_g_s.legend(loc="best")


def find_v_min_and_v_max_for_plotting(experiment, v_monitor):
    if experiment.plot_params.voltage_range is not None:
        lim_down, lim_up = experiment.plot_params.voltage_range
    else:
        lim_down, lim_up = np.min(v_monitor.v) / mvolt, np.max(v_monitor.v) / mvolt
    return lim_down, lim_up


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
