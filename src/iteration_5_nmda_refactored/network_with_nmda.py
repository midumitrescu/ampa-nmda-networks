import brian2.devices.device
import numpy as np
from brian2 import *
from matplotlib import gridspec
from loguru import logger

from Configuration import Experiment
from Plotting import plot_non_blocking
from iteration_4_conductance_based_model.conductance_based_model import plot_psd_and_CVs

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

# NMDA model taken from https://brian2.readthedocs.io/en/2.6.0/examples/frompapers.Wang_2002.html, Wang 2000
wang_model = """        
    dv/dt = 1/C * (-g_L * (v-E_leak) - g_e * (v-E_ampa) - g_i * (v-E_gaba) - g_nmda * sigmoid_v * (v-E_nmda)): volt (unless refractory)  
    dg_e/dt = -g_e / tau_ampa : siemens / meter**2
    dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
    dg_nmda/dt = -g_nmda / tau_nmda_decay + alpha * x * (1- g_nmda/siemens * meter**2): siemens / meter**2
    dx/dt = - x / tau_nmda_rise :  siemens / meter**2
    sigmoid_v = 1/(1 + exp(-0.062 * v/mvolt) * (MG_C/mmole / 3.57)): 1
"""


def check_hidden_variables_are_present_in_model(model_to_run: str, experiment: Experiment):
    not_present = []
    for hidden_variable in experiment.recorded_hidden_variables:
        if hidden_variable not in model_to_run:
            not_present.append(hidden_variable)
    if len(not_present) > 0:
        raise ValueError(f"The following hidden variables are not present in the model: {not_present}")

def sim_and_plot(experiment: Experiment, in_testing=True):
    rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor = sim(experiment, in_testing)
    plot_simulation(experiment, rate_monitor,
                    spike_monitor, v_monitor, g_monitor, internal_states_monitor)

    return rate_monitor, spike_monitor, v_monitor, g_monitor

def sim(experiment: Experiment, in_testing=True):
    """
    g --
    nu_ext_over_nu_thr -- ratio of external stimulus rate to threshold rate
    sim_time -- simulation time
    ax_spikes -- matplotlib axes to plot spikes on
    ax_rates -- matplotlib axes to plot rates on
    rate_tick_step -- step size for rate axis ticks
    """
    start_scope()
    if in_testing:
        np.random.seed(0)
        brian2.devices.device.seed(0)

    defaultclock.dt = experiment.sim_clock

    C = experiment.neuron_params.C

    theta = experiment.neuron_params.theta
    g_L = experiment.neuron_params.g_L
    E_leak = experiment.neuron_params.E_leak
    V_r = experiment.neuron_params.V_r

    g_ampa = experiment.synaptic_params.g_ampa
    g_gaba = experiment.synaptic_params.g_gaba

    g_nmda_max = experiment.synaptic_params.g_nmda

    tau_ampa = 2 * ms
    tau_gaba = 2 * ms

    E_ampa = experiment.synaptic_params.e_ampa

    E_gaba = experiment.synaptic_params.e_gaba

    E_nmda = experiment.synaptic_params.e_ampa
    MG_C = 1 * mmole  # extracellular magnesium concentration
    tau_nmda_decay = 100 * ms
    tau_nmda_rise = 2 * ms
    alpha = 0.05 * Hz  # saturation of NMDA channels at high presynaptic firing rates

    neurons = NeuronGroup(experiment.network_params.N,
                          model=experiment.model,
                          threshold="v >= theta",
                          reset="v = V_r",
                          refractory=experiment.neuron_params.tau_rp,
                          method="euler")
    neurons.v[:] = -65 * mV

    excitatory_neurons = neurons[:experiment.network_params.N_E]
    inhibitory_neurons = neurons[experiment.network_params.N_E:]

    connections = create_connectivity_matrix(experiment)
    excitatory_connections = connections[:experiment.network_params.N_E]
    inhibitory_connections = connections[experiment.network_params.N_E:]
    exc_from, exc_to = excitatory_connections.nonzero()
    inh_from, inh_to = inhibitory_connections.nonzero()

    exc_synapses = Synapses(excitatory_neurons, target=neurons, on_pre="g_e += g_ampa", delay=experiment.synaptic_params.D)
    exc_synapses.connect(i=exc_from, j=exc_to)

    inhib_synapses = Synapses(inhibitory_neurons, target=neurons, on_pre="g_i += g_gaba", delay=experiment.synaptic_params.D)
    inhib_synapses.connect(i=inh_from, j=inh_to)

    # nmda_synapses = Synapses(excitatory_neurons, excitatory_neurons, on_pre='x += w', method="euler")
    nmda_synapses = Synapses(neurons, neurons, on_post="x_nmda = 1", method="euler")
    #TODO: pay attention here. Not only exc to exc. Also inh have NMDAs
    nmda_synapses.connect(i=exc_from, j=exc_to)

    external_poisson_input = PoissonInput(
        target=excitatory_neurons, target_var="g_e_syn", N=experiment.network_params.C_ext, rate=experiment.nu_ext,
        weight=experiment.synaptic_params.g_ampa
    )

    '''
    external_poisson_input = PoissonInput(
        target=neurons, target_var="g_e_syn", N=experiment.network_params.C_ext, rate=experiment.nu_ext,
        weight=experiment.synaptic_params.g_ampa
    )
    '''
    rate_monitor = PopulationRateMonitor(neurons)
    spike_monitor = SpikeMonitor(neurons)
    v_monitor = StateMonitor(source=neurons[
                                    experiment.network_params.N_E - experiment.network_params.neurons_to_record: experiment.network_params.N_E + experiment.network_params.neurons_to_record],
                             variables="v", record=True)

    g_monitor = StateMonitor(source=neurons[
                                    experiment.network_params.N_E - experiment.network_params.neurons_to_record: experiment.network_params.N_E + experiment.network_params.neurons_to_record],
                             variables=["g_e_syn", "g_e", "g_i", "g_nmda"], record=True)

    check_hidden_variables_are_present_in_model(experiment.model, experiment)

    internal_states_monitor = StateMonitor(source=neurons[
                                                  experiment.network_params.N_E - experiment.network_params.neurons_to_record: experiment.network_params.N_E + experiment.network_params.neurons_to_record],
                                           variables=experiment.recorded_hidden_variables, record=True)
    run(experiment.sim_time, report="text", report_period=500*ms)

    return rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor


def plot_v_line(experiment: Experiment, ax_voltages: Axes, v_monitor: StateMonitor, spike_monitor: SpikeMonitor,
                i: int) -> None:
    lines = ax_voltages.plot(v_monitor.t / ms, v_monitor[i].v / mV,
                             label=f"Neuron {i} - {'Exc' if i <= experiment.network_params.neurons_to_record else 'Inh'}",
                             lw=1)
    color = lines[0].get_color()
    spike_times_current_neuron = spike_monitor.all_values()['t'][i] / ms

    v_min_plot, v_max_plot = find_v_min_and_v_max_for_plotting(experiment, v_monitor)

    ax_voltages.vlines(x=spike_times_current_neuron, ymin=v_min_plot, ymax=v_max_plot, color=color, linestyle="-.",
                       label=f"Neuron {i} Spike Time", lw=0.8)


def plot_simulation(experiment: Experiment, rate_monitor,
                    spike_monitor, v_monitor, g_monitor, internal_states_monitor, show=True):

    params_t_range = experiment.plot_params.t_range

    if isinstance(params_t_range[0], list):
        for time_slot in params_t_range:
            plot_simulation_in_one_time_range(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor,
                                              time_range=time_slot, show=show)
            plot_internal_states(experiment, internal_states_monitor, time_range=time_slot)
    else:
        plot_internal_states(experiment, internal_states_monitor, time_range=params_t_range)
        plot_simulation_in_one_time_range(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor,
                                          time_range=params_t_range)

    if experiment.plot_params.show_psd_and_cv():
        plot_psd_and_CVs(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor)
        plot_non_blocking(show)


def plot_simulation_in_one_time_range(experiment: Experiment, rate_monitor: PopulationRateMonitor,
                                      spike_monitor, v_monitor, g_monitor, time_range=[100, 200], show=True):

    if experiment.plot_params.show_raster_and_rate():
        rate_tick_step = experiment.plot_params.rate_tick_step
        fig = plt.figure(figsize=(10, 12))
        fig.suptitle(experiment.gen_plot_title())

        outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5, 2])
        plot_raster_and_rates(experiment, outer[0], rate_monitor, spike_monitor, time_range)
        plot_voltages_and_g_s(experiment, outer[1], g_monitor, spike_monitor, time_range, v_monitor)

        plot_non_blocking(show=show)


def plot_voltages_and_g_s(experiment, grid_spec_mother, g_monitor, spike_monitor, time_range, v_monitor):
    voltage_and_g_s_examples = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec_mother, hspace=0.8)
    ax_voltages, ax_g_s = voltage_and_g_s_examples.subplots(sharex="col")
    ax_voltages.axhline(y=experiment.neuron_params.theta / ms, linestyle="dotted", linewidth="0.3", color="k",
                        label="$\\theta$")
    v_min_plot, v_max_plot = find_v_min_and_v_max_for_plotting(experiment, v_monitor)

    if not np.isnan(v_max_plot):
        ax_voltages.set_ylim([v_min_plot, v_max_plot])

    # TODO: improve this. Maybe set it as plot parameter
    for i in [0, 1]:
        plot_v_line(experiment, ax_voltages, v_monitor, spike_monitor, i)
    for ax in [ax_voltages, ax_g_s]:
        ax.set_xlim(*time_range)
    ax_voltages.legend(loc="right")
    ax_voltages.set_xlabel("t [ms]")
    ax_voltages.set_ylabel("v [mV]")
    i = 0
    ax_g_s.plot(g_monitor.t / ms, g_monitor[i].g_ext_syn, label=rf"$g_\mathrm{{ext}}$[{i}]")
    ax_g_s.plot(g_monitor.t / ms, g_monitor[i].g_e, label=rf"$g_I$[{i}]")
    ax_g_s.plot(g_monitor.t / ms, g_monitor[i].g_i, label=rf"$g_E$[{i}]")
    ax_g_s.plot(g_monitor.t / ms, g_monitor[i].g_nmda, label=rf"$g_\mathrm{{nmda}}$[{i}]")
    ax_g_s.legend(loc="best")


def plot_raster_and_rates(experiment, grid_spec_mother, rate_monitor, spike_monitor, time_range):
    raster_and_population = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec_mother, height_ratios=[4, 1],
                                                             hspace=0)
    ax_spikes, ax_rates = raster_and_population.subplots(sharex="col")
    ax_spikes.plot(spike_monitor.t / ms, spike_monitor.i, "|")
    rate_to_plot = rate_monitor.smooth_rate(
        width=experiment.plot_params.smoothened_rate_width) / Hz if experiment.plot_params.plot_smoothened_rate else rate_monitor.rate / Hz
    ax_rates.plot(rate_monitor.t / ms, rate_to_plot)
    ax_spikes.set_yticks([])
    # ax_rates.set_ylim(*experiment.plot_params.rate_range)
    # ax_rates.set_ylim([0, np.max(rate_monitor.rate[int(len(rate_monitor.t) / 2)] / Hz)])
    # ax_rates.set_ylim([0, np.max(rate_monitor.rate[int(len(rate_monitor.t) / 2)] / Hz)])
    for ax in [ax_spikes, ax_rates]:
        ax.set_xlim(*time_range)
    time_start = int(time_range[0] * ms / experiment.sim_clock)
    time_end = int(time_range[1] * ms / experiment.sim_clock)

    lims = [0, np.max(rate_to_plot[time_start:time_end]) * 1.1]
    ax_rates.set_ylim(
        lims)


def find_v_min_and_v_max_for_plotting(experiment, v_monitor):
    if experiment.plot_params.voltage_range is not None:
        return experiment.plot_params.voltage_range

    lim_down, lim_up = np.min(v_monitor.v) / mvolt, np.max(v_monitor.v) / mvolt

    return lim_down, lim_up


def plot_internal_states(experiment: Experiment, internal_states_monitor, time_range=[100, 200], show=True):

    if experiment.plot_params.show_hidden_variables():
        fig, ax = plt.subplots(len(experiment.plot_params.recorded_hidden_variables), 1, sharex=True, figsize=[10, 8])

        if 1 is len(experiment.plot_params.recorded_hidden_variables):
            ax = [ax]

        neurons_to_plot = [(experiment.network_params.neurons_to_record - 2, "excitatory"), (experiment.network_params.neurons_to_record - 1, "excitatory"),
                           (experiment.network_params.neurons_to_record, "inhibitory"), (experiment.network_params.neurons_to_record + 1, "inhibitory")]
        for neuron_i, label in neurons_to_plot:

            for hidden_var_name, hidden_var_plot_details in experiment.plot_params.create_hidden_variables_plots_grid().items():
                index = hidden_var_plot_details['index']
                ax[index].plot(internal_states_monitor.t / ms, internal_states_monitor[neuron_i].__getattr__(hidden_var_name), label=f"Neuron {neuron_i} - {label}")

        for hidden_var_name, hidden_var_plot_details in experiment.plot_params.create_hidden_variables_plots_grid().items():
            index = hidden_var_plot_details['index']
            title = hidden_var_plot_details['title']
            y_label = hidden_var_plot_details['y_label']
            ax[index].set_title(title)
            ax[index].set_ylabel(y_label)
        '''
        ax[0].plot(internal_states_monitor.t / ms, internal_states_monitor[neuron_i].sigmoid_v, label=f"{neuron_i}", alpha=0.6)
        ax[1].plot(internal_states_monitor.t / ms, internal_states_monitor[neuron_i].x_nmda, label=f"x {neuron_i}", alpha=0.6)
        ax[2].plot(internal_states_monitor.t / ms, internal_states_monitor[neuron_i].g_nmda, label=fr"$g_\mathrm{{NMDA}}$ {neuron_i}", alpha=0.6)
        ax[3].plot(internal_states_monitor.t / ms, internal_states_monitor[neuron_i].I_nmda, label=fr"$I_\mathrm{{NMDA}}$ {neuron_i}", alpha=0.6)
        ax[4].plot(internal_states_monitor.t / ms, internal_states_monitor[neuron_i].one_minus_s_nmda,
                   label=fr" how much free g_nmda exists?{neuron_i}")

        ax[0].set_title("Sigmoid")
        ax[1].set_title("X variable (NMDA upstroke)")
        ax[2].set_title(r"$g_\mathrm{NMDA}$")
        ax[3].set_title(r"NMDA current")
        ax[4].set_title("how much free g_nmda exists?")


        ax[0].set_ylabel("activation \n [unitless]")
        ax[1].set_ylabel(r"$g_\mathrm{NMDA}$ ")
        #ax[1].set_ylim([0, 0.06])
        ax[2].set_ylabel(r'$g_\mathrm{NMDA}$''\n'r'[$\frac{{nS}}{{\mathrm{{cm}}^2}}]$')
        #ax[2].set_ylim([0, 0.3])

        ax[3].set_ylabel(r"$I_\mathrm{NMDA}$""\n""[nA]")
        '''

        ax[-1].set_xlabel("t [ms]")

        ax[0].legend(loc="right")

        for current_ax in ax:
            current_ax.set_xlim(*time_range)

        fig.suptitle(f"{experiment.gen_plot_title()} \n {neurons_to_plot}")
        fig.tight_layout()

        plot_non_blocking(show)

def create_connectivity_matrix(experiment: Experiment):
    if experiment.in_testing:
        np.random.seed(0)
    # Matrix size (N x N)
    N = experiment.network_params.N
    k = int(experiment.network_params.epsilon * N)  # Number of elements to pick in each row (k <= N)

    # Create the N x N matrix, initialized with zeros
    matrix = np.zeros((N, N), dtype=int)

    for i in range(N):
        other_items = [j for j in range(N) if j != i]
        picked_indices = np.random.choice(other_items, k, replace=False)
        matrix[i, picked_indices] = 1
    return matrix
