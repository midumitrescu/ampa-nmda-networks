import brian2.devices.device
from brian2 import *
from joblib import Parallel, delayed
from matplotlib import gridspec

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from Plotting import plot_non_blocking

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True


def sim(experiment: Experiment):
    """
        g --
        nu_ext_over_nu_thr -- ratio of external stimulus rate to threshold rate
        sim_time -- simulation time
        ax_spikes -- matplotlib axes to plot spikes on
        ax_rates -- matplotlib axes to plot rates on
        rate_tick_step -- step size for rate axis ticks
        """
    start_scope()
    if experiment.in_testing:
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

    E_ampa = experiment.synaptic_params.e_ampa
    E_gaba = experiment.synaptic_params.e_gaba
    E_nmda = experiment.synaptic_params.e_ampa

    MG_C = 1 * mmole  # extracellular magnesium concentration

    tau_ampa = experiment.synaptic_params.tau_ampa
    tau_gaba = experiment.synaptic_params.tau_gaba
    tau_nmda_decay = 100 * ms
    tau_nmda_rise = 2 * ms

    alpha = 0.5 * kHz  # saturation of NMDA channels at high presynaptic firing rates

    model = experiment.model
    single_neuron = NeuronGroup(1,
                                model=model,
                                threshold="v >= theta",
                                reset="v = V_r",
                                refractory=experiment.neuron_params.tau_rp,
                                method=experiment.integration_method)
    single_neuron.v[:] = -65 * mV

    P_upstate_exc = PoissonGroup(experiment.network_params.up_state.N_E, rates=experiment.network_params.up_state.nu)
    P_upstate_inh = PoissonGroup(experiment.network_params.up_state.N_I, rates=experiment.network_params.up_state.nu)
    P_downstate_exc = PoissonGroup(experiment.network_params.down_state.N_E,
                                   rates=experiment.network_params.down_state.nu)
    P_downstate_inh = PoissonGroup(experiment.network_params.down_state.N_I,
                                   rates=experiment.network_params.down_state.nu)

    S_upstate_exc = Synapses(P_upstate_exc, single_neuron, model="on: 1", on_pre='g_e += on * g_ampa',
                             method=experiment.integration_method)
    S_upstate_inh = Synapses(P_upstate_inh, single_neuron, model="on: 1", on_pre='g_i += on * g_gaba',
                             method=experiment.integration_method)
    S_downstate_exc = Synapses(P_downstate_exc, single_neuron, model="on: 1", on_pre='g_e += on * g_ampa',
                               method=experiment.integration_method)
    S_downstate_inh = Synapses(P_downstate_inh, single_neuron, model="on: 1", on_pre='g_i += on * g_gaba',
                               method=experiment.integration_method)

    P_nmda = PoissonGroup(N=10, rates=10 * Hz)
    S_nmda = Synapses(P_nmda, single_neuron, on_pre="x_nmda = 1", method=experiment.integration_method)

    S_upstate_exc.connect(p=1)
    S_upstate_inh.connect(p=1)
    S_downstate_exc.connect(p=1)
    S_downstate_inh.connect(p=1)
    S_nmda.connect(p=1)

    S_upstate_exc.on[:] = 1
    S_upstate_inh.on[:] = 1
    S_downstate_exc.on[:] = 0
    S_downstate_inh.on[:] = 0

    @network_operation(dt=100 * ms)
    def toggle_inputs(t):
        if int(t / (0.5 * second)) % 2 == 0:
            S_upstate_exc.on = 1
            S_upstate_inh.on = 1
            S_downstate_exc.on = 0
            S_downstate_inh.on = 0
            logger.debug("at {} we have high active and low inactive", t / second)
        else:
            S_upstate_exc.on = 0
            S_upstate_inh.on = 0
            S_downstate_exc.on = 1
            S_downstate_inh.on = 1
            logger.debug("at {} we have high inactive and low inactive", t / second)

    rate_monitor = PopulationRateMonitor(single_neuron)
    spike_monitor = SpikeMonitor(single_neuron)
    v_monitor = StateMonitor(source=single_neuron,
                             variables="v", record=True)

    g_monitor = StateMonitor(source=single_neuron,
                             variables=["g_nmda", "g_e", "g_i"], record=True)

    internal_states_monitor = StateMonitor(source=single_neuron, variables=experiment.recorded_hidden_variables,
                                           record=True)
    currents_monitor = StateMonitor(source=single_neuron, variables=experiment.plot_params.recorded_currents,
                                    record=True)
    reporting = "text" if experiment.in_testing else None
    print("XXXXXXXXXXXXXXXXXXXXXXX")
    run(experiment.sim_time, report=reporting, report_period=1 * second)
    print("YYYYYYYYYYYYYYYYYYYYYYY")

    return rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor, currents_monitor


def plot(experiment: Experiment, rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor,
         currents_monitor):
    params_t_range = experiment.plot_params.t_range

    if isinstance(params_t_range[0], list):
        for time_range in params_t_range:
            plot_simulation_in_one_time_range(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor,
                                              time_range=time_range)
            plot_currents_in_one_time_range(experiment, currents_monitor, time_range=time_range)
            plot_internal_states(experiment, internal_states_monitor, time_range=time_range)
    else:
        plot_internal_states(experiment, internal_states_monitor, time_range=params_t_range)
        plot_currents_in_one_time_range(experiment, currents_monitor, time_range=params_t_range)
        plot_simulation_in_one_time_range(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor,
                                          time_range=params_t_range)


def sim_and_plot(experiment: Experiment):
    rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor, currents_monitor = sim(experiment)
    plot(experiment, rate_monitor,
         spike_monitor, v_monitor, g_monitor, internal_states_monitor, currents_monitor)

    return rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor, currents_monitor


def plot_simulation_in_one_time_range(experiment: Experiment, rate_monitor: PopulationRateMonitor,
                                      spike_monitor: SpikeMonitor,
                                      v_monitor: StateMonitor, g_monitor: StateMonitor, time_range):
    if experiment.plot_params.show_raster_and_rate():
        rate_tick_step = experiment.plot_params.rate_tick_step
        fig = plt.figure(figsize=(10, 12))
        fig.suptitle(generate_title(experiment))

        height_ratios = [1, 1]
        outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=height_ratios)
        plot_raster_and_rates(experiment, outer[0], rate_monitor, spike_monitor, time_range)
        plot_voltages_and_g_s(experiment, outer[1], g_monitor, spike_monitor, time_range, v_monitor)

        plot_non_blocking(show=True)


def plot_internal_states(experiment: Experiment, internal_states_monitor, time_range: tuple[int, int]):
    if experiment.plot_params.show_hidden_variables():
        fig, ax = plt.subplots(len(experiment.plot_params.recorded_hidden_variables), 1, sharex=True, figsize=[10, 8])

        if 1 == len(experiment.plot_params.recorded_hidden_variables):
            ax = [ax]
        neurons_to_plot = [(0, "excitatory")]
        for neuron_i, label in neurons_to_plot:

            for hidden_var_name, hidden_var_plot_details in experiment.plot_params.create_hidden_variables_plots_grid().items():
                index = hidden_var_plot_details['index']
                curve_to_plot = internal_states_monitor[neuron_i].__getattr__(hidden_var_name) / \
                                hidden_var_plot_details['scaling']
                start_index = int(time_range[0] / experiment.sim_clock * ms)
                end_index = int(time_range[1] / experiment.sim_clock * ms)
                min = np.min(curve_to_plot.data[start_index:end_index])
                max = np.max(curve_to_plot.data[start_index:end_index])

                if min is not None and min < 0:
                    min = min * 1.1
                if max is not None and max > 0:
                    max = max * 1.1

                ax[index].plot(internal_states_monitor.t / ms,
                               curve_to_plot,
                               label=f"Neuron {neuron_i} - {label}")

                if min is not None and not np.isnan(min) and max is not None and not np.isnan(max):
                    ax[index].set_ylim(bottom=min, top=max)

        for hidden_var_name, hidden_var_plot_details in experiment.plot_params.create_hidden_variables_plots_grid().items():
            index = hidden_var_plot_details['index']
            title = hidden_var_plot_details['title']
            y_label = hidden_var_plot_details['y_label']
            ax[index].set_title(title)
            ax[index].set_ylabel(y_label)

        ax[-1].set_xlabel("t [ms]")

        ax[0].legend(loc="right")

        for current_ax in ax:
            current_ax.set_xlim(*time_range)

        fig.suptitle(f"{generate_title(experiment)} \n {neurons_to_plot}")
        fig.tight_layout()

        plot_non_blocking(show)


def plot_currents_in_one_time_range(experiment: Experiment, currents_monitor: StateMonitor, time_range):
    if not experiment.plot_params.show_currents_plots():
        return

    fig, ax = plt.subplots(num=1, figsize=(10, 8))

    time_end, time_start = determine_start_and_end_recorded_indexes(experiment, time_range)

    for current_to_plot in experiment.plot_params.recorded_currents:
        current_curve = currents_monitor[0].__getattr__(current_to_plot)

        ax.plot(currents_monitor.t[time_start:time_end] / ms, current_curve[time_start:time_end],
                label=f"{current_to_plot}", alpha=0.5)

    ax.legend(loc="right")
    fig.tight_layout()
    fig.suptitle(f"Current plot {generate_title(experiment)}")
    plot_non_blocking()

def plot_raster_and_rates(experiment, grid_spec_mother, rate_monitor, spike_monitor, time_range):
    spike_monitor_results = np.vstack((spike_monitor.t / ms, spike_monitor.i))

    plot_raster_and_rates_unpickled(experiment, grid_spec_mother, rate_monitor, spike_monitor_results, time_range)


def plot_raster_and_rates_unpickled(experiment, grid_spec_mother, rate_monitor, spike_monitor, time_range):
    rate_curve = rate_monitor.smooth_rate(
        width=experiment.plot_params.smoothened_rate_width) / Hz if experiment.plot_params.plot_smoothened_rate else rate_monitor.rate / Hz

    rate_to_plot = np.vstack((rate_monitor.t / ms, rate_curve))



    raster_and_population = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec_mother, height_ratios=[4, 1],
                                                             hspace=0)
    ax_spikes, ax_rates = raster_and_population.subplots(sharex="col")
    ax_spikes.plot(spike_monitor[0], spike_monitor[1], "|")
    ax_rates.plot(rate_to_plot[0], rate_to_plot[1])
    ax_spikes.set_yticks([])

    for ax in [ax_spikes, ax_rates]:
        ax.set_xlim(*time_range)
    time_end, time_start = determine_start_and_end_recorded_indexes(experiment, time_range)

    lims = [0, np.max(rate_to_plot[time_start:time_end]) * 1.1]
    ax_rates.set_ylim(lims)


def determine_start_and_end_recorded_indexes(experiment, time_range):
    time_start = int(time_range[0] * ms / experiment.sim_clock)
    time_end = int(time_range[1] * ms / experiment.sim_clock)
    return time_end, time_start


def plot_voltages_and_g_s(experiment, grid_spec_mother, g_monitor, spike_monitor, time_range, v_monitor):
    voltage_and_g_s_examples = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec_mother, hspace=0.8)
    ax_voltages, ax_g_s = voltage_and_g_s_examples.subplots(sharex="col")
    ax_voltages.axhline(y=experiment.neuron_params.theta / ms, linestyle="dotted", linewidth="0.3", color="k",
                        label="$\\theta$")
    '''
    v_min_plot, v_max_plot = find_v_min_and_v_max_for_plotting(experiment, v_monitor)

    if not np.isnan(v_max_plot):
        ax_voltages.set_ylim([v_min_plot, v_max_plot])
    '''
    for i in [0]:
        plot_v_line(experiment, ax_voltages, v_monitor, spike_monitor, i)
    for ax in [ax_voltages, ax_g_s]:
        ax.set_xlim(*time_range)
    ax_voltages.legend(loc="right")
    ax_voltages.set_xlabel("t [ms]")
    ax_voltages.set_ylabel("v [mV]")
    i = 0

    ax_g_s.plot(g_monitor.t / ms, g_monitor[i].g_e, label=r"$g_\mathrm{Exc}$", alpha=0.5)
    ax_g_s.plot(g_monitor.t / ms, g_monitor[i].g_i, label=r"$g_\mathrm{Inh}$", alpha=0.5)

    ax_g_s.plot(g_monitor.t / ms, g_monitor[i].g_nmda, label=rf"$g_\mathrm{{nmda}}$[{i}]", alpha=0.5)
    ax_g_s.legend(loc="best")


def plot_v_line(experiment: Experiment, ax_voltages: Axes, v_monitor: StateMonitor, spike_monitor: SpikeMonitor,
                i: int) -> None:
    lines = ax_voltages.plot(v_monitor.t / ms, v_monitor[i].v / mV, lw=1)
    color = lines[0].get_color()
    spike_times_current_neuron = spike_monitor.all_values()['t'][i] / ms

    ax_voltages.vlines(x=spike_times_current_neuron, ymin=-70, ymax=-35, color=color, linestyle="-.",
                       label=f"Neuron {i} Spike Time", lw=0.8)


def generate_title(experiment: Experiment):
    return fr"""{experiment.plot_params.panel}
    Up State: [{experiment.network_params.up_state.gen_plot_title()}, {experiment.effective_time_constant_up_state.gen_plot_title()}]
    Down State: [{experiment.network_params.down_state.gen_plot_title()}, {experiment.effective_time_constant_down_state.gen_plot_title()}]    
    Neuron: [$C={experiment.neuron_params.C * cm ** 2}$, $g_L={experiment.neuron_params.g_L * cm ** 2}$, $\theta={experiment.neuron_params.theta}$, $V_R={experiment.neuron_params.V_r}$, $E_L={experiment.neuron_params.E_leak}$, $\tau_M={experiment.neuron_params.tau}$, $\tau_{{\mathrm{{ref}}}}={experiment.neuron_params.tau_rp}$]
    Synapse: [$g_{{\mathrm{{AMPA}}}}={experiment.synaptic_params.g_ampa * (cm ** 2) / uS:.2f}\,\mu\mathrm{{S}}$, $g_{{\mathrm{{GABA}}}}={experiment.synaptic_params.g_gaba * (cm ** 2) / uS:.2f}\,\mu\mathrm{{S}}$, $g={experiment.network_params.g}$]"""


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
'''

'''
Expectation is that all experiments share the same time windows
'''
def sim_and_plot_experiment_grid(experiments: list[Experiment]):

    results = run_two_experiments(experiments)

    t_range = experiments[0].plot_params.t_range
    if t_range:
        params_t_range = t_range

        if isinstance(params_t_range[0], list):
            for time_slot in params_t_range:
                plot_results_grid(results, time_slot)
        else:
            plot_results_grid(results, t_range)


def run_two_experiments(experiments):

    def sim_unpickled(experiment: Experiment):
        rate_monitor, spike_monitor, _, _, _, _ = sim(experiment)

        return experiment, spike_monitor_results, np.array(
            rate_monitor.smooth_rate(width=experiment.plot_params.smoothened_rate_width) / Hz)

    return Parallel(n_jobs=2)(
        delayed(sim_unpickled)(current_experiment) for current_experiment in experiments
    )

def plot_results_grid(results, time_range):
    fig = plt.figure(figsize=(20, 25))
    fig.suptitle("Working", size=25)

    outer = gridspec.GridSpec(1, len(results), figure=fig, hspace=0.2,
                              wspace=0.1)

    for index, result in enumerate(results):
        experiment, spike_results, rate_results = result
        plot_raster_and_rates(experiment, outer[index], rate_results, spike_results, time_range)

    fig.show()
