import itertools
from typing import Callable

from brian2 import *
from joblib import Parallel, delayed
from matplotlib import gridspec

from Configuration import Experiment, NetworkParams, SynapticParams
from Plotting import show_plots_non_blocking
from iteration_5_nmda_refactored.network_with_nmda import sim, sim_with_nmda_also_to_inh
from utils import ExtendedDict


def extract_hidden_variables(experiment: Experiment, hidden_variables_monitor: StateMonitor):
    values = {
        "t": np.array(hidden_variables_monitor.t / ms),
    }
    # make an if to extract each variable's by its unit!
    for hidden_variable in experiment.recorded_hidden_variables:
        curve = np.array(hidden_variables_monitor.__getattr__(hidden_variable))
        values[hidden_variable] = curve
    return ExtendedDict(values)

def produce_comparrison_plot(experiments: list[Experiment], increasing_g_s, increasing_nu_ext_over_nu_thr, label_for_x_axis: str, label_for_y_axis: str, grid_title: str,
                             func: Callable[[Experiment], tuple[PopulationRateMonitor, SpikeMonitor, StateMonitor, StateMonitor, StateMonitor]] = sim):

    experiment = experiments[0]

    def compute_spikes_and_rates(current_experiment):

        rate_monitor, spike_monitor, _, _, hidden_state_monitor = func(current_experiment)
        spike_monitor_results = np.vstack((spike_monitor.t / ms, spike_monitor.i))
        hidden_variables_values = extract_hidden_variables(current_experiment, hidden_state_monitor)
        return (current_experiment, spike_monitor_results,
                np.array(rate_monitor.smooth_rate(width=experiment.plot_params.smoothened_rate_width) / Hz),
                hidden_variables_values)

    # Parallel computation of subplot data
    grid_simulation_results = Parallel(n_jobs=16)(
        delayed(compute_spikes_and_rates)(current_experiment)
        for current_experiment in experiments
    )

    if experiment.plot_params.t_range:
        params_t_range = experiment.plot_params.t_range

        if isinstance(params_t_range[0], list):
            for time_slot in params_t_range:
                plot(increasing_g_s, increasing_nu_ext_over_nu_thr, grid_simulation_results,
                                      time_range=time_slot, label_for_x_axis=label_for_x_axis, label_for_y_axis=label_for_y_axis, grid_title=grid_title)
        else:
            plot(increasing_g_s, increasing_nu_ext_over_nu_thr, grid_simulation_results,
                                  time_range=params_t_range, label_for_x_axis=label_for_x_axis, label_for_y_axis=label_for_y_axis, grid_title=grid_title)


def plot(y_axis_values, x_axis_values, results, time_range, label_for_x_axis: str, label_for_y_axis: str, grid_title: str):
    experiment, _, _, _ = results[0]
    if experiment.plot_params.show_raster_and_rate():
        plot_raster_and_rates(y_axis_values, x_axis_values, results, time_range, label_for_x_axis, label_for_y_axis, grid_title)
    if experiment.plot_params.show_hidden_variables():
        plot_hidden_variables(y_axis_values, x_axis_values, results, time_range, label_for_x_axis, label_for_y_axis, grid_title)

def plot_raster_and_rates(y_axis_values, x_axis_values, results, time_range, label_for_x_axis: str, label_for_y_axis: str, grid_title: str):
    fig = plt.figure(figsize=(20, 25))
    fig.suptitle(grid_title, size=25)

    outer = gridspec.GridSpec(len(y_axis_values), len(x_axis_values), figure=fig, hspace=0.2,
                              wspace=0.1)

    for result, grid_spec in zip(results, outer):
        experiment, spike_monitor, rate_monitor, _ = result
        plot_raster_and_rates_unpickled(experiment, grid_spec, rate_monitor, spike_monitor, time_range=time_range)

    for y_axis_index, y_value in zip(np.arange(0, len(y_axis_values)) * 2 * len(x_axis_values), y_axis_values):
        fig.get_axes()[y_axis_index].set_ylabel(f"{label_for_y_axis}={y_value: .3f}", size=20)

    for x_axis_index, x_value in enumerate(x_axis_values):
        # Each cell has 2 axes (spikes and rates), ordered row by row, top to bottom
        # To get the ax_rate in the bottom row, go to row = rows - 1, col = col_idx
        column_index = (len(y_axis_values) - 1) * len(
            x_axis_values) * 2 + x_axis_index * 2 + 1  # +1 because ax_rates is second in the pair
        fig.get_axes()[column_index].set_xlabel(
            f'{label_for_x_axis}'f"={x_value: .3f}", size=25)

    # Add a tiny Axes for drawing arrows manually
    ax_arrow = fig.add_axes([0, 0, 1, 1], zorder=-1)
    ax_arrow.axis("off")  # Hide this overlay axes

    # Vertical arrow (for increasing g)
    ax_arrow.annotate(
        '', xy=(0.05, 0.725), xytext=(0.05, 0.2),
        arrowprops=dict(arrowstyle="->", linewidth=2),
        xycoords='figure fraction'
    )
    fig.text(0.015, 0.46, f'increasing {label_for_y_axis}', va='center', ha='center',
             rotation='vertical', fontsize=20)

    # Shorter + lower horizontal arrow (75% length, lower y)
    arrow_start = 0.2
    arrow_end = arrow_start + 0.75 * (0.9 - arrow_start)  # 75% of original length
    arrow_y = 0.05  # move lower

    # Arrow itself
    ax_arrow.annotate(
        '', xy=(arrow_end, arrow_y), xytext=(arrow_start, arrow_y),
        arrowprops=dict(arrowstyle="->", linewidth=2),
        xycoords='figure fraction'
    )

    # Text slightly below it
    fig.text((arrow_start + arrow_end) / 2, arrow_y - 0.01,
             f'increasing {label_for_x_axis}',
             ha='center', va='top', fontsize=20)

    plt.tight_layout()
    plt.show(block=False)

def plot_hidden_variables(y_axis_values, x_axis_values, results, time_range, label_for_x_axis: str, label_for_y_axis: str, grid_title: str):

    fig = plt.figure(figsize=(20, 25))
    fig.suptitle(grid_title, size=25)

    outer = gridspec.GridSpec(len(y_axis_values), len(x_axis_values), figure=fig, hspace=0.2,
                              wspace=0.1)

    for result, grid_spec in zip(results, outer):
        experiment, _, _, hidden_variables = result
        plot_hidden_variables_unpickled(experiment, grid_spec, hidden_variables, time_range=time_range)

    for y_axis_index, y_value in zip(np.arange(0, len(y_axis_values)) * len(experiment.recorded_hidden_variables) * len(x_axis_values), y_axis_values):
        fig.get_axes()[y_axis_index + int(0.5 * len(experiment.recorded_hidden_variables))].set_ylabel(f"{label_for_y_axis}={y_value: .3f}", size=20)

    for x_axis_index, x_value in enumerate(x_axis_values):
        # Each cell has 2 axes (spikes and rates), ordered row by row, top to bottom
        # To get the ax_rate in the bottom row, go to row = rows - 1, col = col_idx
        column_index = x_axis_index * len(x_axis_values) * len(experiment.recorded_hidden_variables)
        fig.get_axes()[column_index].set_xlabel(
            f'{label_for_x_axis}'f"={x_value: .3f}", size=25)



    # Add a tiny Axes for drawing arrows manually
    ax_arrow = fig.add_axes([0, 0, 1, 1], zorder=-1)
    ax_arrow.axis("off")  # Hide this overlay axes

    # Vertical arrow (for increasing g)
    ax_arrow.annotate(
        '', xy=(0.05, 0.725), xytext=(0.05, 0.2),
        arrowprops=dict(arrowstyle="->", linewidth=2),
        xycoords='figure fraction'
    )
    fig.text(0.015, 0.46, f'increasing {label_for_y_axis}', va='center', ha='center',
             rotation='vertical', fontsize=20)

    # Shorter + lower horizontal arrow (75% length, lower y)
    arrow_start = 0.2
    arrow_end = arrow_start + 0.75 * (0.9 - arrow_start)  # 75% of original length
    arrow_y = 0.05  # move lower

    # Arrow itself
    ax_arrow.annotate(
        '', xy=(arrow_end, arrow_y), xytext=(arrow_start, arrow_y),
        arrowprops=dict(arrowstyle="->", linewidth=2),
        xycoords='figure fraction'
    )

    # Text slightly below it
    fig.text((arrow_start + arrow_end) / 2, arrow_y - 0.01,
             f'increasing {label_for_x_axis}',
             ha='center', va='top', fontsize=20)

    outer.tight_layout(figure=fig)
    fig.tight_layout()
    plt.show(block=False)


def plot_raster_and_rates_unpickled(experiment, grid_spec_mother, rate_monitor, spike_monitor, time_range):
    raster_and_population = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec_mother, height_ratios=[4, 1],
                                                             hspace=0)
    ax_spikes, ax_rates = raster_and_population.subplots(sharex="col")
    it_steps = int(experiment.sim_time / experiment.sim_clock)
    t = np.arange(0, it_steps) * experiment.sim_clock / ms
    ax_spikes.plot(spike_monitor[0], spike_monitor[1], "|")
    ax_spikes.set_yticks([])

    ax_rates.plot(t, rate_monitor[0:len(t)], lw=0.5)
    for ax in [ax_spikes, ax_rates]:
        ax.set_xlim(*time_range)
    time_start = int(time_range[0] * ms / experiment.sim_clock)
    time_end = int(time_range[1] * ms / experiment.sim_clock)
    upper_limit = np.max(rate_monitor[time_start:time_end]) * 1.1 if len(rate_monitor) > 0 else 1
    lims = [0, upper_limit]
    if lims[1] > 0:
        ax_rates.set_ylim(lims)

    ax_rates.set_xlabel("time (ms)")

def plot_hidden_variables_unpickled(experiment, grid_spec_mother, hidden_variables, time_range):
    if experiment.plot_params.show_hidden_variables():
        hidden_variables_grid_spec = gridspec.GridSpecFromSubplotSpec(len(experiment.plot_params.recorded_hidden_variables), 1, subplot_spec=grid_spec_mother, hspace=0.6)
        ax = hidden_variables_grid_spec.subplots(sharex="col")

        if 1 is len(experiment.plot_params.recorded_hidden_variables):
            ax = [ax]

        neurons_to_plot = [(experiment.network_params.neurons_to_record - 2, "excitatory"),
                           (experiment.network_params.neurons_to_record - 1, "excitatory"),
                           (experiment.network_params.neurons_to_record, "inhibitory"),
                           (experiment.network_params.neurons_to_record + 1, "inhibitory")]
        for neuron_i, label in neurons_to_plot:

            for hidden_var_name, hidden_var_plot_details in experiment.plot_params.create_hidden_variables_plots_grid().items():
                index = hidden_var_plot_details['index']
                ax[index].plot(hidden_variables.t,
                               hidden_variables[hidden_var_name][neuron_i],
                               label=f"Neuron {neuron_i} - {label}", alpha=0.3)

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
        #fig.suptitle(f"{experiment.gen_plot_title()} \n {neurons_to_plot}")
        #fig.tight_layout()

def compare_g_nmda_vs_nu_ext_over_nu_thr(experiment, g_nmdas, nu_ext_over_nu_thrs, nmda_to_inh_neurons = False):
    g_nmdas = np.flip(g_nmdas)

    experiments = [experiment.with_property(SynapticParams.KEY_G_NMDA, g_nmda).with_property(NetworkParams.KEY_NU_E_OVER_NU_THR, n)
                   for g_nmda, n in itertools.product(g_nmdas, nu_ext_over_nu_thrs)]
    title = fr""" {experiment.plot_params.panel}
    Network: [N={experiment.network_params.N}, $N_E={experiment.network_params.N_E}$, $N_I={experiment.network_params.N_I}$, $\gamma={experiment.network_params.gamma}$, $\epsilon={experiment.network_params.epsilon}$]
    Input: [$\nu_T={experiment.nu_thr}$]
    Neuron: [$C={experiment.neuron_params.C * cm ** 2}$, $g_L={experiment.neuron_params.g_L * cm ** 2}$, $\theta={experiment.neuron_params.theta}$, $V_R={experiment.neuron_params.V_r}$, $E_L={experiment.neuron_params.E_leak}$, $\tau_M={experiment.neuron_params.tau}$, $\tau_{{\mathrm{{ref}}}}={experiment.neuron_params.tau_rp}$]
    Synapse: [$g_{{\mathrm{{GABA}}}}=g \cdot g_{{\mathrm{{AMPA}}}}$ $g={experiment.network_params.g}$]"""

    produce_comparrison_plot(experiments, g_nmdas * 10**6, nu_ext_over_nu_thrs,
                             label_for_x_axis=r'$\frac{\nu_\mathrm{Ext}}{\nu_\mathrm{Thr}}$', label_for_y_axis=r"$g_\mathrm{NMDA}$",
                             grid_title=title, func=sim_with_nmda_also_to_inh if nmda_to_inh_neurons else sim)

    show_plots_non_blocking()