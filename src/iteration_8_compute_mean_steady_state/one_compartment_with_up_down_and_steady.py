import numpy as np
from brian2 import plt, mpl, StateMonitor, mV, siemens, start_scope, defaultclock, mmole, kHz, NeuronGroup, run, \
    Network, second, stop, ms, nS
from loguru import logger
from matplotlib import gridspec
from matplotlib.gridspec import SubplotSpec
from mpl_toolkits.axes_grid1.mpl_axes import Axes

from Plotting import show_plots_non_blocking
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, State
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    SimulationResults
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    simulate_with_up_and_down_state_and_nmda, simulate_with_down_and_up_state_and_nmda

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

class SteadyStateResults:

    def __init__(self,  steady_state_results: StateMonitor):
        self.v_steady = steady_state_results[0].v[-1] / mV
        self.g_e_steady = steady_state_results[0].g_e[-1] / siemens
        self.g_i_steady = steady_state_results[0].g_i[-1] / siemens
        self.g_nmda_steady = steady_state_results[0].g_nmda[-1] / siemens
        self.x_nmda_steady = steady_state_results[0].x_nmda[-1]
        self.s_nmda_steady = steady_state_results[0].s_nmda[-1]

    def __str__(self):
        return (f"V = [{self.v_steady:.6f}], g_e = [{self.g_e_steady:.6f}], g_i = [{self.g_i_steady:.6f}], g_nmda = [{self.g_nmda_steady:.6f}], "
                f"x = [{self.x_nmda_steady:.6f}], s = [{self.s_nmda_steady:.6f}] ")

class SimulationResultsWithSteadyState(SimulationResults):

    def __init__(self, simulation_results: SimulationResults, steady_up_state_results: SteadyStateResults, steady_down_state_results: SteadyStateResults):
        self.experiment = simulation_results.experiment
        self.rates = simulation_results.rates
        self.spikes = simulation_results.spikes
        self.voltages = simulation_results.voltages
        self.g_s = simulation_results.g_s
        self.currents = simulation_results.currents
        self.internal_states_monitor = simulation_results.internal_states_monitor

        self.steady_up_results = steady_up_state_results
        self.steady_down_results = steady_down_state_results

def sim_and_plot_up_down(experiment: Experiment) -> SimulationResultsWithSteadyState:
    simulation_results = simulate_with_up_and_down_state_and_nmda_and_steady_state(experiment)
    plot_simulation(simulation_results)
    return simulation_results

def sim_and_plot_down_up(experiment: Experiment) -> SimulationResultsWithSteadyState:
    simulation_results = simulate_with_down_and_up_state_and_nmda_and_steady_state(experiment)
    plot_simulation(simulation_results)
    return simulation_results

def simulate_with_down_and_up_state_and_nmda_and_steady_state(experiment: Experiment):
    logger.debug("Simulating steady state for {}", experiment)
    steady_up_state_results = sim_steady_state(experiment, state=experiment.network_params.up_state)
    steady_down_state_results = sim_steady_state(experiment, state=experiment.network_params.down_state)
    logger.debug("Steady state simulation for {} done! Results are Up State = {}, Down State = {}", experiment, steady_up_state_results, steady_down_state_results)
    simulation_results = simulate_with_down_and_up_state_and_nmda(experiment)
    return SimulationResultsWithSteadyState(simulation_results, steady_up_state_results, steady_down_state_results)

def simulate_with_up_and_down_state_and_nmda_and_steady_state(experiment: Experiment):
    logger.debug("Simulating steady state for {}", experiment)
    steady_up_state_results = sim_steady_state(experiment, state=experiment.network_params.up_state)
    steady_down_state_results = sim_steady_state(experiment, state=experiment.network_params.down_state)
    logger.debug("Steady state simulation for {} done! Results are Up State = {}, Down State = {}", experiment, steady_up_state_results, steady_down_state_results)
    simulation_results = simulate_with_up_and_down_state_and_nmda(experiment)
    return SimulationResultsWithSteadyState(simulation_results, steady_up_state_results, steady_down_state_results)

def sim_steady_state(experiment: Experiment, state: State) -> SteadyStateResults:

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

    r_e = state.nu
    r_i = state.nu
    N_E = state.N_E
    N_I = state.N_I
    N_N = state.N_NMDA
    r_nmda = state.nu_nmda

    neuron = NeuronGroup(1,
                          model=experiment.steady_state_model,
                          method="euler")
    neuron.v[:] = experiment.neuron_params.E_leak
    v_monitor = StateMonitor(source=neuron,
                             variables=["v", "g_e", "g_i", "g_nmda", "x_nmda", "s_nmda"], record=True)

    steady_state_network = Network([neuron, v_monitor])

    run(1 * second, report="text", report_period=1 * second)
    result = SteadyStateResults(v_monitor)
    stop()
    return result

def plot_simulation(simulation_results: SimulationResultsWithSteadyState):
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


def plot_raster_and_g_s_in_one_time_range(simulation_results: SimulationResultsWithSteadyState, time_range):
    if simulation_results.experiment.plot_params.show_raster_and_rate():

        fig = plt.figure(figsize=(14, 12))
        fig.suptitle(generate_title(simulation_results.experiment))

        height_ratios = [1, 1]
        outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=height_ratios)
        plot_raster_and_rates(simulation_results, time_range, outer[0])
        plot_voltages_and_g_s(simulation_results, time_range, outer[1])

        show_plots_non_blocking(show=True)


def plot_currents_in_one_time_range(simulation_results: SimulationResultsWithSteadyState, time_range):
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

def plot_currents(simulation_results: SimulationResultsWithSteadyState, time_range: tuple[int, int], grid_spec_mother: SubplotSpec):
    voltage_and_currents = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec_mother)
    ax_voltage, ax_currents = voltage_and_currents.subplots(sharex="row")

    time_end, time_start = determine_start_and_end_recorded_indexes(simulation_results.experiment, time_range)

    ax_voltage.plot(simulation_results.voltages.t[time_start:time_end], simulation_results.voltages.v[0][time_start:time_end])
    ax_voltage.axhline(y=simulation_results.steady_up_results.v_steady, linestyle="--", linewidth=1.5,
                       label=f"Mean V={simulation_results.steady_up_results.v_steady :.3f}- UP State ")

    ax_voltage.axhline(y=simulation_results.steady_down_results.v_steady, linestyle="--", linewidth=1.5,
                       label=f"Mean V={simulation_results.steady_down_results.v_steady :.3f} - Down State")

    ax_voltage.legend()
    ax_voltage.set_ylim(top=simulation_results.experiment.neuron_params.theta / mV + 3)


    for current_to_plot in simulation_results.experiment.plot_params.recorded_currents:
        current_curve = simulation_results.currents[current_to_plot][0]
        ax_currents.plot(simulation_results.currents.t[time_start:time_end], current_curve[time_start:time_end],
                label=f"{current_to_plot} - Neuron 0", alpha=0.5)

    ax_voltage.set_title("Membrane voltage")
    ax_currents.set_title("Currents")

    return ax_voltage, ax_currents


def plot_raster_and_rates(simulation_results: SimulationResultsWithSteadyState, time_range, grid_spec_mother):

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


def plot_voltages_and_g_s(simulation_results: SimulationResultsWithSteadyState, time_range, grid_spec_mother):
    voltage_and_g_s_examples = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec_mother, hspace=0.8)
    ax_voltages, ax_g_s = voltage_and_g_s_examples.subplots(sharex="col")
    ax_voltages.axhline(y=simulation_results.experiment.neuron_params.theta / ms, linestyle="dotted", linewidth="0.3", color="k",
                        label="$\\theta$")

    for i in [0]:
        plot_v_line(simulation_results, i, ax_voltages)
    for ax in [ax_voltages, ax_g_s]:
        ax.set_xlim(*time_range)

    ax_voltages.set_title(rf"Membrane voltage. $V_\mathrm{{M, Up}}$ ={simulation_results.steady_up_results.v_steady: .3f} mV, $V_\mathrm{{M, Down}}$ ={simulation_results.steady_down_results.v_steady: .3f} mV")
    ax_voltages.legend(loc="right")
    ax_voltages.set_xlabel("t [ms]")
    ax_voltages.set_ylabel("v [mV]")
    i = 0

    ax_g_s.set_title("Conductances")
    g_e_lines = ax_g_s.plot(simulation_results.g_s.t, simulation_results.g_s.g_e[i], label=r"$g_\mathrm{Exc}$", alpha=0.5)
    g_i_lines = ax_g_s.plot(simulation_results.g_s.t, simulation_results.g_s.g_i[i], label=r"$g_\mathrm{Inh}$", alpha=0.5)
    g_nmda_lines = ax_g_s.plot(simulation_results.g_s.t / ms, simulation_results.g_s.g_nmda[i], label=rf"$g_\mathrm{{nmda}}$[{i}]", alpha=0.5)

    ax_g_s.axhline(y=simulation_results.steady_up_results.g_e_steady, linestyle="--", linewidth=2,
                        color=g_e_lines[0].get_color(),
                        label="UP $g_e$ steady", alpha=0.6)
    ax_g_s.axhline(y=simulation_results.steady_up_results.g_i_steady, linestyle="--", linewidth=2,
                   color=g_i_lines[0].get_color(),
                   label="UP $g_i$ steady", alpha=0.6)
    ax_g_s.axhline(y=simulation_results.steady_up_results.g_nmda_steady, linestyle="--", linewidth=2,
                        color=g_nmda_lines[0].get_color(),
                        label=r"UP $g_\mathrm{nmda}$", alpha=0.6)

    ax_g_s.axhline(y=simulation_results.steady_down_results.g_e_steady, linestyle="dotted", linewidth=2,
                   color=g_e_lines[0].get_color(),
                   label="Down $g_e$ steady", alpha=0.6)
    ax_g_s.axhline(y=simulation_results.steady_down_results.g_i_steady, linestyle="dotted", linewidth=2,
                   color=g_i_lines[0].get_color(),
                   label="Down $g_i$ steady", alpha=0.6)
    ax_g_s.axhline(y=simulation_results.steady_down_results.g_nmda_steady, linestyle="dotted", linewidth=2,
                   color=g_nmda_lines[0].get_color(),
                   label=r"Down $g_\mathrm{nmda}$", alpha=0.6)


    ax_g_s.set_ylabel("conductance \n" + r"[$\frac{\mathrm{siemens}}{\mathrm{cm}^2}$]")
    ax_g_s.legend(loc="right")


def plot_v_line(simulation_results,
                i: int, ax_voltages: Axes) -> None:
    lines = ax_voltages.plot(simulation_results.voltages.t, simulation_results.voltages.v[i], lw=1)
    color = lines[0].get_color()
    #spike_times_current_neuron = simulation_results.spike_monitor.all_values()['t'][i] / ms Keep This in mind
    spike_times_current_neuron = simulation_results.spikes.all_values['t'][i] / ms

    ax_voltages.vlines(x=spike_times_current_neuron, ymin=-70, ymax=-35, color=color, linestyle="-.",
                       label=f"Neuron {i} Spike Time", lw=0.8)

    ax_voltages.vlines(x=spike_times_current_neuron, ymin=-70, ymax=-35, color=color, linestyle="-.",
                       label=f"Neuron {i} Spike Time", lw=0.8)
    ax_voltages.axhline(y=simulation_results.steady_up_results.v_steady, color=color, linestyle="--", linewidth=1.5,
                   label="Mean V - UP State")

    ax_voltages.axhline(y=simulation_results.steady_down_results.v_steady, color=color, linestyle="--", linewidth=1.5,
                        label="Mean V - Down State")

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

        show_plots_non_blocking(True)


def generate_title(experiment: Experiment):
    return fr"""{experiment.plot_params.panel}
    Up State: [{experiment.network_params.up_state.gen_plot_title()}, {experiment.effective_time_constant_up_state.gen_plot_title()}]
    Down State: [{experiment.network_params.down_state.gen_plot_title()}, {experiment.effective_time_constant_down_state.gen_plot_title()}]    
    Neuron: [$C={experiment.neuron_params.C}$, $g_L={experiment.neuron_params.g_L}$, $\theta={experiment.neuron_params.theta}$, $V_R={experiment.neuron_params.V_r}$, $E_L={experiment.neuron_params.E_leak}$, $\tau_M={experiment.neuron_params.tau}$, $\tau_{{\mathrm{{ref}}}}={experiment.neuron_params.tau_rp}$]
    Synapse: [$g_{{\mathrm{{AMPA}}}}={experiment.synaptic_params.g_ampa:.2f}$, $g_{{\mathrm{{GABA}}}}={experiment.synaptic_params.g_gaba/ nS:.2f}\,n\mathrm{{S}}$, $g={experiment.network_params.g}$, $g_{{\mathrm{{NMDA}}}}={experiment.synaptic_params.g_nmda / nS:.2f}\,n\mathrm{{S}}$]"""


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
