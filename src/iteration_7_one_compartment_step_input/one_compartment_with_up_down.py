import brian2.devices.device
from brian2 import *
from matplotlib import gridspec

from Plotting import show_plots_non_blocking
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from utils import ExtendedDict

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True


class SimulationResults:

    def __init__(self, experiment: Experiment, rate_monitor: PopulationRateMonitor, spike_monitor: SpikeMonitor,
                 v_monitor: StateMonitor, g_monitor: StateMonitor, internal_states_monitor: StateMonitor,
                 currents_monitor: StateMonitor):
        self.experiment = experiment
        self.rates = self.__extract_rates__(rate_monitor)
        self.spikes = self.__extract_spikes__(spike_monitor)
        self.voltages = self.__extract_voltages__(v_monitor)
        self.g_s = self.__extract_g_s__(g_monitor)
        self.currents = self.__extract_currents__(currents_monitor)

        self.internal_states_monitor = internal_states_monitor

    def __extract_rates__(self, rate_monitor: PopulationRateMonitor):
        rates_to_extract = rate_monitor.smooth_rate(width=self.experiment.plot_params.smoothened_rate_width) / Hz \
            if self.experiment.plot_params.plot_smoothened_rate \
            else rate_monitor.rate / Hz
        return np.vstack((rate_monitor.t / ms, rates_to_extract))

    @staticmethod
    def __extract_spikes__(spike_monitor: SpikeMonitor):
        return ExtendedDict({
            "t": np.array(spike_monitor.t / ms),
            "i": np.array(spike_monitor.i),
            "all_values": spike_monitor.all_values(),
        })

    @staticmethod
    def __extract_voltages__(v_monitor: StateMonitor):
        return ExtendedDict({
            "t": np.array(v_monitor.t / ms),
            "v": np.array(v_monitor.v/ ms),
        })

    def __extract_g_s__(self, g_monitor: StateMonitor):
        values = {
            "t": np.array(g_monitor.t / ms),
        }
        for recorded_g in self.experiment.plot_params.recorded_g_s:
            values[recorded_g] = np.array(g_monitor.__getattr__(recorded_g) / siemens * cm**2)
        return ExtendedDict(values)

    '''
    Units micro Ampere / cm^2 suggested in Neuronal Dynamics, page 83, Fig 4.1 b
    '''
    def __extract_currents__(self, currents_monitor: StateMonitor):
        if len(self.experiment.plot_params.recorded_currents) > 0:
            values = {
                "t": np.array(currents_monitor.t / ms),
            }
            for current in self.experiment.plot_params.recorded_currents:
                values[current] = np.array(currents_monitor.__getattr__(current) / uamp)
            return ExtendedDict(values)
        else:
            return ExtendedDict({})

    def rate_monitor_t(self):
        return self.rates[0]

    def rate_monitor_rates(self):
        return self.rates[1]

def sim_and_plot(experiment: Experiment) -> SimulationResults:
    simulation_results = simulate_with_up_and_down_state_and_nmda(experiment)
    plot_simulation(simulation_results)
    return simulation_results

def simulate_with_up_and_down_state_and_nmda(experiment: Experiment):
    """
        g --
        nu_ext_over_nu_thr -- ratio of external stimulus rate to threshold rate
        sim_time -- simulation time
        ax_spikes -- matplotlib axes to plot spikes on
        ax_rates -- matplotlib axes to plot rates on
        rate_tick_step -- step size for rate axis ticks
        """
    if experiment.in_testing:
        np.random.seed(0)
        brian2.devices.device.seed(0)
        seed(0)
        np.random.default_rng(0)

    start_scope()

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
    tau_nmda_rise = experiment.synaptic_params.tau_nmda_rise
    tau_nmda_decay = experiment.synaptic_params.tau_nmda_decay

    alpha = 0.5 * kHz  # saturation of NMDA channels at high presynaptic firing rates

    model = experiment.model
    single_neuron = NeuronGroup(1,
                                model=model,
                                threshold="v >= theta",
                                reset="v = V_r",
                                refractory=experiment.neuron_params.tau_rp,
                                method=experiment.integration_method)
    single_neuron.v[:] = -65 * mV

    order = [0, 1, 2, 3, 4, 5] if experiment.in_testing else [0] * 5

    P_upstate_exc = PoissonGroup(experiment.network_params.up_state.N_E, rates=experiment.network_params.up_state.nu, order=order[0])
    P_upstate_inh = PoissonGroup(experiment.network_params.up_state.N_I, rates=experiment.network_params.up_state.nu, order=order[1])
    P_upstate_nmda = PoissonGroup(N=experiment.network_params.up_state.N_NMDA, rates=experiment.network_params.up_state.nu_nmda, order=order[4])
    P_downstate_exc = PoissonGroup(experiment.network_params.down_state.N_E,
                                   rates=experiment.network_params.down_state.nu, order=order[2])
    P_downstate_inh = PoissonGroup(experiment.network_params.down_state.N_I,
                                   rates=experiment.network_params.down_state.nu, order=order[3])
    P_downstate_nmda = PoissonGroup(N=experiment.network_params.down_state.N_NMDA, rates=experiment.network_params.down_state.nu_nmda, order=order[5])

    S_upstate_exc = Synapses(P_upstate_exc, single_neuron, model="on: 1", on_pre='g_e += on * g_ampa',
                             method=experiment.integration_method)
    S_upstate_inh = Synapses(P_upstate_inh, single_neuron, model="on: 1", on_pre='g_i += on * g_gaba',
                             method=experiment.integration_method)
    S_upstate_nmda = Synapses(P_upstate_nmda, single_neuron, model="on: 1", on_pre="x_nmda += on*1",
                               method=experiment.integration_method)
    S_downstate_exc = Synapses(P_downstate_exc, single_neuron, model="on: 1", on_pre='g_e += on * g_ampa',
                               method=experiment.integration_method)
    S_downstate_inh = Synapses(P_downstate_inh, single_neuron, model="on: 1", on_pre='g_i += on * g_gaba',
                               method=experiment.integration_method)
    S_downstate_nmda = Synapses(P_downstate_nmda, single_neuron, model="on: 1", on_pre='x_nmda += on*1',
                               method=experiment.integration_method)

    S_upstate_exc.connect(p=1)
    S_upstate_inh.connect(p=1)
    S_upstate_nmda.connect(p=1)
    S_downstate_exc.connect(p=1)
    S_downstate_inh.connect(p=1)
    S_downstate_nmda.connect(p=1)

    S_upstate_exc.on[:] = 1
    S_upstate_inh.on[:] = 1
    S_upstate_nmda.on[:] = 1
    S_downstate_exc.on[:] = 0
    S_downstate_inh.on[:] = 0
    S_downstate_nmda.on[:] = 0

    @network_operation(dt=100 * ms)
    def toggle_inputs(t):
        step = int(t / (0.5 * second))
        if step % 2 == 0:
            S_upstate_exc.on = 1
            S_upstate_inh.on = 1
            S_upstate_nmda.on = 1

            S_downstate_exc.on = 0
            S_downstate_inh.on = 0
            S_downstate_nmda.on = 0
            logger.debug("at {} we have high active and low inactive", t / second)
        else:
            S_upstate_exc.on = 0
            S_upstate_inh.on = 0
            S_upstate_nmda.on = 0

            S_downstate_exc.on = 1
            S_downstate_inh.on = 1
            S_downstate_nmda.on = 1
            logger.debug("at {} we have high inactive and low inactive", t / second)

    rate_monitor = PopulationRateMonitor(single_neuron)
    spike_monitor = SpikeMonitor(single_neuron)
    v_monitor = StateMonitor(source=single_neuron,
                             variables="v", record=True)

    g_monitor = StateMonitor(source=single_neuron,
                             variables=experiment.plot_params.recorded_g_s, record=True)

    internal_states_monitor = StateMonitor(source=single_neuron, variables=experiment.recorded_hidden_variables,
                                           record=True)
    currents_monitor = StateMonitor(source=single_neuron, variables=experiment.plot_params.recorded_currents,
                                    record=True)
    reporting = "text" if experiment.in_testing else None
    run(experiment.sim_time, report=reporting, report_period=1 * second)

    return SimulationResults(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor,
                             currents_monitor)

def simulate_with_down_and_up_state_and_nmda(experiment: Experiment):
    """
        g --
        nu_ext_over_nu_thr -- ratio of external stimulus rate to threshold rate
        sim_time -- simulation time
        ax_spikes -- matplotlib axes to plot spikes on
        ax_rates -- matplotlib axes to plot rates on
        rate_tick_step -- step size for rate axis ticks
        """
    if experiment.in_testing:
        np.random.seed(0)
        brian2.devices.device.seed(0)
        seed(0)
        np.random.default_rng(0)

    start_scope()

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

    order = [0, 1, 2, 3, 4, 5] if experiment.in_testing else [0] * 5

    P_upstate_exc = PoissonGroup(experiment.network_params.up_state.N_E, rates=experiment.network_params.up_state.nu, order=order[0])
    P_upstate_inh = PoissonGroup(experiment.network_params.up_state.N_I, rates=experiment.network_params.up_state.nu, order=order[1])
    P_upstate_nmda = PoissonGroup(N=experiment.network_params.up_state.N_NMDA, rates=experiment.network_params.up_state.nu_nmda, order=order[4])
    P_downstate_exc = PoissonGroup(experiment.network_params.down_state.N_E,
                                   rates=experiment.network_params.down_state.nu, order=order[2])
    P_downstate_inh = PoissonGroup(experiment.network_params.down_state.N_I,
                                   rates=experiment.network_params.down_state.nu, order=order[3])
    P_downstate_nmda = PoissonGroup(N=experiment.network_params.down_state.N_NMDA, rates=experiment.network_params.down_state.nu_nmda, order=order[5])

    S_upstate_exc = Synapses(P_upstate_exc, single_neuron, model="on: 1", on_pre='g_e += on * g_ampa',
                             method=experiment.integration_method)
    S_upstate_inh = Synapses(P_upstate_inh, single_neuron, model="on: 1", on_pre='g_i += on * g_gaba',
                             method=experiment.integration_method)
    S_upstate_nmda = Synapses(P_upstate_nmda, single_neuron, model="on: 1", on_pre="x_nmda += on*1",
                               method=experiment.integration_method)
    S_downstate_exc = Synapses(P_downstate_exc, single_neuron, model="on: 1", on_pre='g_e += on * g_ampa',
                               method=experiment.integration_method)
    S_downstate_inh = Synapses(P_downstate_inh, single_neuron, model="on: 1", on_pre='g_i += on * g_gaba',
                               method=experiment.integration_method)
    S_downstate_nmda = Synapses(P_downstate_nmda, single_neuron, model="on: 1", on_pre='x_nmda += on*1',
                               method=experiment.integration_method)

    S_upstate_exc.connect(p=1)
    S_upstate_inh.connect(p=1)
    S_upstate_nmda.connect(p=1)
    S_downstate_exc.connect(p=1)
    S_downstate_inh.connect(p=1)
    S_downstate_nmda.connect(p=1)

    S_upstate_exc.on[:] = 1
    S_upstate_inh.on[:] = 1
    S_upstate_nmda.on[:] = 1
    S_downstate_exc.on[:] = 0
    S_downstate_inh.on[:] = 0
    S_downstate_nmda.on[:] = 0

    @network_operation(dt=100 * ms)
    def toggle_inputs(t):
        step = int(t / (0.5 * second))
        if step % 2 == 0:
            S_upstate_exc.on = 0
            S_upstate_inh.on = 0
            S_upstate_nmda.on = 0

            S_downstate_exc.on = 1
            S_downstate_inh.on = 1
            S_downstate_nmda.on =10
            logger.debug("at {} we have up inactive and low active", t / second)
        else:
            S_upstate_exc.on = 1
            S_upstate_inh.on = 1
            S_upstate_nmda.on = 1

            S_downstate_exc.on = 0
            S_downstate_inh.on = 0
            S_downstate_nmda.on = 0
            logger.debug("at {} we have up active and down inactive", t / second)

    rate_monitor = PopulationRateMonitor(single_neuron)
    spike_monitor = SpikeMonitor(single_neuron)
    v_monitor = StateMonitor(source=single_neuron,
                             variables="v", record=True)

    g_monitor = StateMonitor(source=single_neuron,
                             variables=experiment.plot_params.recorded_g_s, record=True)

    internal_states_monitor = StateMonitor(source=single_neuron, variables=experiment.recorded_hidden_variables,
                                           record=True)
    currents_monitor = StateMonitor(source=single_neuron, variables=experiment.plot_params.recorded_currents,
                                    record=True)
    reporting = "text" if experiment.in_testing else None
    run(experiment.sim_time, report=reporting, report_period=1 * second)

    return SimulationResults(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor,
                             currents_monitor)


def plot_simulation(simulation_results: SimulationResults):
    params_t_range = simulation_results.experiment.plot_params.t_range

    if isinstance(params_t_range[0], list):
        for time_range in params_t_range:
            plot_raster_and_g_s_in_one_time_range(simulation_results,
                                                  time_range=time_range)
            plot_currents_in_one_time_range(simulation_results, time_range=time_range)
            plot_internal_states_in_one_time_range(simulation_results, time_range=time_range)
    else:
        plot_raster_and_g_s_in_one_time_range(simulation_results,
                                              time_range=params_t_range)
        plot_currents_in_one_time_range(simulation_results, time_range=params_t_range)
        plot_internal_states_in_one_time_range(simulation_results, time_range=params_t_range)


def plot_raster_and_g_s_in_one_time_range(simulation_results: SimulationResults, time_range):
    if simulation_results.experiment.plot_params.show_raster_and_rate():

        fig = plt.figure(figsize=(14, 12))
        fig.suptitle(generate_title(simulation_results.experiment))

        height_ratios = [1, 1]
        outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=height_ratios)
        plot_raster_and_rates(simulation_results, time_range, outer[0])
        plot_voltages_and_g_s(simulation_results, time_range, outer[1])

        show_plots_non_blocking(show=True)


def plot_currents_in_one_time_range(simulation_results: SimulationResults, time_range):
    if not simulation_results.experiment.plot_params.show_currents_plots():
        return

    fig = plt.figure(figsize=(14, 8))
    outer = gridspec.GridSpec(1, 1, figure=fig)
    ax_voltage, ax_currents = plot_currents(simulation_results, time_range=time_range, grid_spec_mother=outer[0])

    ax_currents.set_xlabel("Time (ms)")
    ax_voltage.set_ylabel(r"[mV]")
    ax_currents.set_ylabel(r"[$\frac{\mu A}{\mathrm{cm}^2}$]")
    ax_currents.legend(loc="right")
    fig.suptitle(f"Current plot {generate_title(simulation_results.experiment)}")
    fig.tight_layout()
    show_plots_non_blocking()

def plot_currents(simulation_results: SimulationResults, time_range: tuple[int, int], grid_spec_mother: SubplotSpec):
    voltage_and_currents = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec_mother)
    ax_voltage, ax_currents = voltage_and_currents.subplots(sharex="row")

    time_end, time_start = determine_start_and_end_recorded_indexes(simulation_results.experiment, time_range)

    ax_voltage.plot(simulation_results.voltages.t[time_start:time_end], simulation_results.voltages.v[0][time_start:time_end])

    for current_to_plot in simulation_results.experiment.plot_params.recorded_currents:
        current_curve = simulation_results.currents[current_to_plot][0]
        ax_currents.plot(simulation_results.currents.t[time_start:time_end], current_curve[time_start:time_end],
                label=f"{current_to_plot} - Neuron 0", alpha=0.5)

    ax_voltage.set_title("Membrane voltage")
    ax_currents.set_title("Currents")

    return ax_voltage, ax_currents


def plot_raster_and_rates(simulation_results: SimulationResults, time_range, grid_spec_mother):

    raster_and_population = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec_mother, height_ratios=[2, 1],
                                                             hspace=0)


    ax_spikes, ax_rates = raster_and_population.subplots(sharex="col")
    ax_spikes.plot(simulation_results.spikes.t, simulation_results.spikes.i, "|")
    ax_rates.plot(simulation_results.rate_monitor_t(), simulation_results.rate_monitor_rates())
    ax_spikes.set_yticks([])

    for ax in [ax_spikes, ax_rates]:
        ax.set_xlim(*time_range)
    time_end, time_start = determine_start_and_end_recorded_indexes(simulation_results.experiment, time_range)

    lims = [0, np.max(simulation_results.rate_monitor_rates()[time_start:time_end]) * 1.1]
    ax_rates.set_ylim(lims)

    if simulation_results.experiment.plot_params.plot_smoothened_rate:
        ax_spikes.set_title(f"Raster and smoothened [$\sigma$ = {simulation_results.experiment.plot_params.smoothened_rate_width / ms :.2f} ms] population rate")
    else:
        ax_spikes.set_title("Raster and unsmoothened population rate")

    return ax_spikes, ax_rates


def determine_start_and_end_recorded_indexes(experiment, time_range):
    time_start = int(time_range[0] * ms / experiment.sim_clock)
    time_end = int(time_range[1] * ms / experiment.sim_clock)
    return time_end, time_start


def plot_voltages_and_g_s(simulation_results: SimulationResults, time_range, grid_spec_mother):
    voltage_and_g_s_examples = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec_mother, hspace=0.8)
    ax_voltages, ax_g_s = voltage_and_g_s_examples.subplots(sharex="col")
    ax_voltages.axhline(y=simulation_results.experiment.neuron_params.theta / ms, linestyle="dotted", linewidth="0.3", color="k",
                        label="$\\theta$")

    for i in [0]:
        plot_v_line(simulation_results, i, ax_voltages)
    for ax in [ax_voltages, ax_g_s]:
        ax.set_xlim(*time_range)

    ax_voltages.set_title("Membrane voltage")
    ax_voltages.legend(loc="right")
    ax_voltages.set_xlabel("t [ms]")
    ax_voltages.set_ylabel("v [mV]")
    i = 0

    ax_g_s.set_title("Conductances")
    ax_g_s.plot(simulation_results.g_s.t, simulation_results.g_s.g_e[i], label=r"$g_\mathrm{Exc}$", alpha=0.5)
    ax_g_s.plot(simulation_results.g_s.t, simulation_results.g_s.g_i[i], label=r"$g_\mathrm{Inh}$", alpha=0.5)

    ax_g_s.plot(simulation_results.g_s.t / ms, simulation_results.g_s.g_nmda[i], label=rf"$g_\mathrm{{nmda}}$[{i}]", alpha=0.5)

    ax_g_s.set_ylabel("conductance \n" + r"[$\frac{\mathrm{siemens}}{\mathrm{cm}^2}$]")
    ax_g_s.legend(loc="best")


def plot_v_line(simulation_results,
                i: int, ax_voltages: Axes) -> None:
    lines = ax_voltages.plot(simulation_results.voltages.t, simulation_results.voltages.v[i], lw=1)
    color = lines[0].get_color()
    #spike_times_current_neuron = simulation_results.spike_monitor.all_values()['t'][i] / ms Keep This in mind
    spike_times_current_neuron = simulation_results.spikes.all_values['t'][i] / ms

    ax_voltages.vlines(x=spike_times_current_neuron, ymin=-70, ymax=-35, color=color, linestyle="-.",
                       label=f"Neuron {i} Spike Time", lw=0.8)

def plot_internal_states_in_one_time_range(simulation_results, time_range: tuple[int, int]):
    if simulation_results.experiment.plot_params.show_hidden_variables():
        fig, ax = plt.subplots(len(simulation_results.experiment.plot_params.recorded_hidden_variables), 1, sharex=True, figsize=[14, 8])

        if 1 == len(simulation_results.experiment.plot_params.recorded_hidden_variables):
            ax = [ax]
        neurons_to_plot = [(0, "excitatory")]
        for neuron_i, label in neurons_to_plot:

            for hidden_var_name, hidden_var_plot_details in simulation_results.experiment.plot_params.create_hidden_variables_plots_grid().items():
                index = hidden_var_plot_details['index']
                curve_to_plot = simulation_results.internal_states_monitor[neuron_i].__getattr__(hidden_var_name) / \
                                hidden_var_plot_details['scaling']
                start_index = int(time_range[0] / simulation_results.experiment.sim_clock * ms)
                end_index = int(time_range[1] / simulation_results.experiment.sim_clock * ms)
                min = np.min(curve_to_plot.data[start_index:end_index])
                max = np.max(curve_to_plot.data[start_index:end_index])

                if min is not None and min < 0:
                    min = min * 1.1
                if max is not None and max > 0:
                    max = max * 1.1

                ax[index].plot(simulation_results.internal_states_monitor.t / ms,
                               curve_to_plot,
                               label=f"Neuron {neuron_i} - {label}")

                if min is not None and not np.isnan(min) and max is not None and not np.isnan(max):
                    ax[index].set_ylim(bottom=min, top=max)

        for hidden_var_name, hidden_var_plot_details in simulation_results.experiment.plot_params.create_hidden_variables_plots_grid().items():
            index = hidden_var_plot_details['index']
            title = hidden_var_plot_details['title']
            y_label = hidden_var_plot_details['y_label']
            ax[index].set_title(title)
            ax[index].set_ylabel(y_label)

        ax[-1].set_xlabel("t [ms]")

        ax[0].legend(loc="right")

        for current_ax in ax:
            current_ax.set_xlim(*time_range)

        fig.suptitle(f"{generate_title(simulation_results.experiment)} \n {neurons_to_plot}")
        fig.tight_layout()

        show_plots_non_blocking(show)


def generate_title(experiment: Experiment):
    return fr"""{experiment.plot_params.panel}
    Up State: [{experiment.network_params.up_state.gen_plot_title()}, {experiment.effective_time_constant_up_state.gen_plot_title()}]
    Down State: [{experiment.network_params.down_state.gen_plot_title()}, {experiment.effective_time_constant_down_state.gen_plot_title()}]    
    Neuron: [$C={experiment.neuron_params.C}$, $g_L={experiment.neuron_params.g_L}$, $\theta={experiment.neuron_params.theta}$, $V_R={experiment.neuron_params.V_r}$, $E_L={experiment.neuron_params.E_leak}$, $\tau_M={experiment.neuron_params.tau}$, $\tau_{{\mathrm{{ref}}}}={experiment.neuron_params.tau_rp}$]
    Synapse: [$g_{{\mathrm{{AMPA}}}}={experiment.synaptic_params.g_ampa:.2f}$, $g_{{\mathrm{{GABA}}}}={experiment.synaptic_params.g_gaba:.2f}$, $g={experiment.network_params.g}$, $g_{{\mathrm{{NMDA}}}}={experiment.synaptic_params.g_nmda:.2f}$]"""


single_compartment_with_nmda = '''
dv/dt = 1/C * (- I_L - I_ampa - I_gaba - I_nmda): volt (unless refractory)

I_L = g_L * (v-E_leak): amp

I_ampa = g_e * (v - E_ampa): amp
I_gaba = g_i * (v - E_gaba): amp
I_nmda = g_nmda * (v - E_nmda): amp

dg_e/dt = -g_e / tau_ampa : siemens
dg_i/dt = -g_i / tau_gaba  : siemens

g_nmda = g_nmda_max * sigmoid_v * s_nmda: siemens
ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha * x_nmda * (1 - s_nmda) : 1
dx_nmda/dt = - x_nmda / tau_nmda_rise : 1

sigmoid_v = 1/(1 + exp(-0.062 * (v/mvolt + 43)) * (MG_C/mmole / 3.57)): 1
'''

single_compartment_with_nmda_and_logged_variables = f'''{single_compartment_with_nmda}

one_minus_s_nmda = 1 - s_nmda : 1
alpha_x_t = alpha * x_nmda: Hz
s_drive = alpha * x_nmda * (1 - s_nmda) : Hz
v_minus_e_gaba = v-E_gaba : volt
I_fast = I_ampa + I_gaba : amp
'''
