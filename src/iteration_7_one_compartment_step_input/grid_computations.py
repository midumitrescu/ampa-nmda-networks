import numpy as np
from brian2 import ms, Hz
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.mpl_axes import Axes

from Plotting import plot_non_blocking
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import sim, plot_raster_and_rates_unpickled

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
        rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor, currents_monitor = sim(experiment)
        spike_monitor_results = np.vstack((spike_monitor.t / ms, spike_monitor.i))
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
        plot_raster_and_rates_unpickled(experiment, outer[index], rate_results, spike_results, time_range)
        plot_simulation_in_one_time_range_unpickled(experiment, outer[index], rate_results, spike_results, None, time_range)

    plot_non_blocking()

def plot_simulation_in_one_time_range_unpickled(experiment: Experiment, rate_array: np.ndarray,
                                      spikes_array: np.ndarray,
                                      voltages_array: np.ndarray, g_s_array: dict, time_range):
    if experiment.plot_params.show_raster_and_rate():

        fig = plt.figure(figsize=(10, 12))
        fig.suptitle("")

        height_ratios = [1, 1]
        outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=height_ratios)
        plot_raster_and_rates_unpickled(experiment, outer[0], rate_array, spikes_array, time_range)
        plot_voltages_and_g_s(experiment, outer[1], g_s_array, spikes_array, time_range, voltages_array)

def plot_raster_and_rates_unpickled(experiment, grid_spec_mother, rate_monitor, spike_monitor, time_range):
    raster_and_population = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec_mother, height_ratios=[4, 1],
                                                             hspace=0)
    ax_spikes, ax_rates = raster_and_population.subplots(sharex="col")
    it_steps = int(experiment.sim_time / experiment.sim_clock)
    t = np.arange(0, it_steps) * experiment.sim_clock / ms
    #ax_spikes.plot(spike_monitor[0], spike_monitor[1], "|", lw=0.1, markersize=0.1)
    ax_spikes.plot(spike_monitor[0], spike_monitor[1], "|")
    ax_spikes.set_yticks([])

    ax_rates.plot(t, rate_monitor, lw=0.5)
    for ax in [ax_spikes, ax_rates]:
        ax.set_xlim(*time_range)
    time_start = int(time_range[0] * ms / experiment.sim_clock)
    time_end = int(time_range[1] * ms / experiment.sim_clock)
    upper_limit = np.max(rate_monitor[time_start:time_end]) * 1.1 if len(rate_monitor) > 0 else 1
    lims = [0, upper_limit]
    if lims[1] > 0:
        ax_rates.set_ylim(lims)

    ax_rates.set_xlabel("time (ms)")

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

def plot_voltages_and_g_s_unpickled(experiment, ):
    pass

def plot_v_line(experiment: Experiment, ax_voltages: Axes, v_array: np.ndarray, spike_monitor: np.ndarray,
                i: int) -> None:
    lines = ax_voltages.plot(v_array[0], v_array[1], lw=1)
    color = lines[0].get_color()
    spike_times_current_neuron = spike_monitor.all_values()['t'][i] / ms

    ax_voltages.vlines(x=spike_times_current_neuron, ymin=-70, ymax=-35, color=color, linestyle="-.",
                       label=f"Neuron {i} Spike Time", lw=0.8)

