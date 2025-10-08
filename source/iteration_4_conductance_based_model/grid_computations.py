import itertools

import numpy as np
from brian2 import *
from joblib import Parallel, delayed
from matplotlib import gridspec

from Configuration import Experiment, NetworkParams
from iteration_4_conductance_based_model.conductance_based_model import sim


def produce_comparrison_plot(experiment: Experiment, increasing_g_s, increasing_nu_ext_over_nu_thr):

    increasing_g_s = np.flip(increasing_g_s)

    def compute_spikes_and_rates(g, nu_ext_over_nu_thr):
        print(f"Computing spikes and rates for g = {g} and nu_ext_over_nu_thr = {nu_ext_over_nu_thr}")

        current_experiment = experiment.with_property(NetworkParams.KEY_G, g).with_property(NetworkParams.KEY_NU_E_OVER_NU_THR,
                                                                                   nu_ext_over_nu_thr)
        rate_monitor, spike_monitor, _, _ = sim(current_experiment)
        spike_monitor_results = np.vstack((spike_monitor.t / ms, spike_monitor.i))

        return experiment, spike_monitor_results, np.array(rate_monitor.smooth_rate(width=experiment.plot_params.smoothened_rate_width) / Hz)

    # Parallel computation of subplot data
    grid_simulation_results = Parallel(n_jobs=16)(
        delayed(compute_spikes_and_rates)(g, nu)
        for g, nu in itertools.product(increasing_g_s, increasing_nu_ext_over_nu_thr)
    )

    if experiment.plot_params.t_range:
        params_t_range = experiment.plot_params.t_range

        if isinstance(params_t_range[0], list):
            for time_slot in params_t_range:
                plot_(experiment, increasing_g_s, increasing_nu_ext_over_nu_thr, grid_simulation_results, time_range=time_slot)
        else:
            plot_(experiment, increasing_g_s, increasing_nu_ext_over_nu_thr, grid_simulation_results, time_range=params_t_range)



def plot_(config: Experiment, increasing_g_s, increasing_nu_ext_over_nu_thr, results, time_range):

    fig = plt.figure(figsize=(20, 25))
    fig.suptitle(fr"""{config.plot_params.panel}
    Network: [N={config.network_params.N}, $N_E={config.network_params.N_E}$, $N_I={config.network_params.N_I}$, $\gamma={config.network_params.gamma}$, $\epsilon={config.network_params.epsilon}$]
    Input: [$\nu_T={config.nu_thr}$ Hz]
    Neuron: [$C={config.neuron_params.C * cm**2}$, $g_L={config.neuron_params.g_L * cm**2}$, $\theta={config.neuron_params.theta}$, $V_R={config.neuron_params.V_r}$, $E_L={config.neuron_params.E_leak}$, $\tau_M={config.neuron_params.tau}$, $\tau_{{\mathrm{{ref}}}}={config.neuron_params.tau_rp}$]
    Synapse: [$g_{{\mathrm{{AMPA}}}}={config.synaptic_params.g_ampa * (cm**2) / uS:.2f}\,\mu\mathrm{{S}}$, $g_{{\mathrm{{GABA}}}}={config.synaptic_params.g_gaba * (cm**2) / uS:.2f}\,\mu\mathrm{{S}}$]""", size=25)

    outer = gridspec.GridSpec(len(increasing_g_s), len(increasing_nu_ext_over_nu_thr), figure=fig, hspace=0.2,
                              wspace=0.1)

    for result, grid_spec in zip(results, outer):
        experiment, spike_monitor, rate_monitor = result
        plot_raster_and_rates_unpickled(experiment, grid_spec, rate_monitor, spike_monitor, time_range=time_range)

    for row_index, g in zip(np.arange(0, len(increasing_g_s)) * 2 * len(increasing_nu_ext_over_nu_thr), increasing_g_s):
        fig.get_axes()[row_index].set_ylabel(f"g={g}", size=20)


    for nu_ratio_index, nu_ext_over_nu_thr in enumerate(increasing_nu_ext_over_nu_thr):
        # Each cell has 2 axes (spikes and rates), ordered row by row, top to bottom
        # To get the ax_rate in the bottom row, go to row = rows - 1, col = col_idx
        column_index = (len(increasing_g_s) - 1) * len(increasing_nu_ext_over_nu_thr) * 2 + nu_ratio_index * 2 + 1  # +1 because ax_rates is second in the pair
        fig.get_axes()[column_index].set_xlabel(r'$\frac{\nu_\mathrm{Ext}}{\nu_\mathrm{Thr}}$'f"={nu_ext_over_nu_thr: .3f}", size=25)


    # write g on axes 0, 4, 8, 12  range(0, len(g_s) * 4
    # 4 is from 2 x len(nus)

    # Add a tiny Axes for drawing arrows manually
    ax_arrow = fig.add_axes([0, 0, 1, 1], zorder=-1)
    ax_arrow.axis("off")  # Hide this overlay axes

    # Vertical arrow (for increasing g)
    ax_arrow.annotate(
        '', xy=(0.05, 0.725), xytext=(0.05, 0.2),
        arrowprops=dict(arrowstyle="->", linewidth=2),
        xycoords='figure fraction'
    )
    fig.text(0.015, 0.46, 'increasing g', va='center', ha='center',
             rotation='vertical', fontsize=15)

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
             r'increasing $\nu_{\mathrm{Ext}} / \nu_{\mathrm{Thr}}$',
             ha='center', va='top', fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_raster_and_rates_unpickled(experiment, grid_spec_mother, rate_monitor, spike_monitor, time_range):
    raster_and_population = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec_mother, height_ratios=[4, 1],
                                                             hspace=0)
    ax_spikes, ax_rates = raster_and_population.subplots(sharex="col")
    it_steps = int(experiment.sim_time / experiment.sim_clock)
    t = np.arange(0, it_steps)
    ax_spikes.plot(spike_monitor[0], spike_monitor[1], "|", lw=0.1, markersize=0.1)
    ax_spikes.set_yticks([])


    ax_rates.plot(t, rate_monitor)
    for ax in [ax_spikes, ax_rates]:
        ax.set_xlim(*time_range)
    time_start = int(time_range[0] * ms / experiment.sim_clock)
    time_end = int(time_range[1] * ms / experiment.sim_clock)
    ax_rates.set_ylim(
        [np.min(rate_monitor[time_start:time_end]) / Hz, np.max(rate_monitor[time_start:time_end]) / Hz])