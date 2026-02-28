import itertools

from brian2 import PopulationRateMonitor, SpikeMonitor, StateMonitor, ms, Hz
import matplotlib.pyplot as plt
from brian2.units.allunits import pampere

import numpy as np
from joblib import delayed, Parallel
from mpl_toolkits.axes_grid1 import make_axes_locatable

from iteration_13_compte.compte_utils import CueInfo
from iteration_13_compte.configs import AnExampleExperiment, CompteResults

def compute_binned_firing_rate(results: CompteResults, bin_size = 10):
    bins = int(results.sim_time / bin_size)

    rates = np.zeros((results.example.NE, bins))
    for i in range(results.example.NE):
        spikes = results.spikes_monitor.all_values[i] / ms
        counts, _ = np.histogram(spikes, bins=bins, range=(0, results.sim_time))
        rates[i] = 1000 * counts / bin_size  # 1000 because binsize = 10 ms => 0.01 seconds

    return rates

def compute_ordered_firing_rates_from_results(results: CompteResults, theta_array_assignment):
    rates = compute_binned_firing_rate(results)

    order = np.argsort(theta_array_assignment)
    rates_sorted = rates[order]
    return rates_sorted

def to_spike_trains_from_results(results: CompteResults, neurons: list[int], sim_time):
    neurons = np.array(neurons)  # e.g. [3, 7, 12, 20]
    dt = results.dt

    T = int(np.ceil(sim_time / dt))
    N = len(neurons)

    raster = np.zeros((N, T), dtype=np.uint8)
    # Spike data
    spikes_neuron_index = np.array(results.spikes_monitor.i)
    t = np.array(results.spikes_monitor.t)
    # Convert spike times to indices
    time_idx = (t / dt).astype(np.int32)

    # Map global neuron indices → local indices
    neuron_map = {nid: k for k, nid in enumerate(neurons)}

    # Mask spikes we care about
    mask = np.isin(spikes_neuron_index, neurons)

    # Local neuron indices
    local_i = np.fromiter(
        (neuron_map[n] for n in spikes_neuron_index[mask]),
        dtype=np.int32,
        count=np.sum(mask)
    )

    # Assign spikes
    raster[local_i, time_idx[mask]] = 1
    return raster

def raster_to_rates(raster: np.ndarray, dt: float):
    dt_ms = dt
    W_ms = 50
    W = int(W_ms / dt_ms)

    kernel = np.ones(W, dtype=np.float32) / (W * dt_ms * 1e-3)

    from scipy.signal import fftconvolve

    rate = fftconvolve(
        raster,
        kernel[None, :],
        mode='same',
        axes=1
    )
    return rate

def plot_compte_results(results: CompteResults, theta_array_assignment, cues: list[CueInfo]):

    rates_sorted = compute_ordered_firing_rates_from_results(results=results, theta_array_assignment=theta_array_assignment)

    fig, axs = plt.subplots(
        4, 1,
        figsize=(20, 16),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 2, 1, 1]},
    )

    im = axs[0].imshow(
        rates_sorted,
        aspect='auto',
        origin='lower',
        cmap='jet',
        extent=[0, results.sim_time, 0, results.example.NE]
    )

    angle_ticks = np.array([0, 90, 180, 270, 359])
    theta = np.asarray(theta_array_assignment)

    ytick_indices = [
        np.argmin(np.abs(theta - angle))
        for angle in angle_ticks
    ]
    axs[0].set_yticks(ytick_indices)
    axs[0].set_yticklabels([f"{a}°" for a in angle_ticks])
    axs[0].set_ylabel("Preferred angle (°)")

    #[axs[0].axvspan(cue.delay / ms, (cue.delay + cue.duration) /ms, alpha=0.3) for cue in cues]
    import matplotlib.transforms as mtransforms

    # transform: x in data coordinates, y in axes coordinates
    trans = mtransforms.blended_transform_factory(
        axs[0].transData, axs[0].transAxes
    )

    for _, cue in enumerate(cues):
        start = cue.delay / ms
        end = (cue.delay + cue.duration) / ms

        axs[0].plot(
            [start, end],  # x range in data coords
            [-0.015, -0.015],  # y position just below axis (axes coords)
            transform=trans,
            linewidth=10,
            solid_capstyle="butt",
            label=f"Cue ({start:.0f}–{end:.0f} ms)",
            clip_on=False,
        )

    axs[0].legend(loc="upper right")

    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(im, cax=cax, label="Firing rate (Hz)")

    axs[0].set_title('Bump attractor dynamics')

    neurons = [0, 200]
    neuron_labels = ["non-cue", "cue"]

    # --- 2) Raster plot ---
    axs[1].plot(
        results.spikes_monitor.t,
        results.spikes_monitor.i,
        ".",
        markersize=1,
        color="blue"
    )

    axs[1].set_yticks(ytick_indices)
    axs[1].set_yticklabels([f"{a}°" for a in angle_ticks])
    axs[1].set_ylabel("Preferred angle (°)")

    raster = to_spike_trains_from_results(results, neurons=neurons, sim_time=results.sim_time)
    rates = raster_to_rates(raster, results.dt)

    axs[2].plot(
        results.population_rate_monitor.t,
        results.population_rate_monitor.population_rate,
        label="Population",
        color="black",
        linewidth=2
    )
    for neuron_index, rate in zip(neurons, rates):
        axs[2].plot(
            results.population_rate_monitor.t,
            rate,
            label=f"Neuron {neuron_index}"
        )

    axs[2].set_ylabel('Rate (Hz)')
    axs[2].legend(loc="upper right")

    for neuron_index, neuron_label in zip(neurons, neuron_labels):
        axs[3].plot(
            results.currents_monitor.t,
            results.currents_monitor.I_AMPA[neuron_index],
            label=f"I AMPA, {neuron_index} - {neuron_label}",
            alpha=0.6
        )
        axs[3].plot(
            results.currents_monitor.t,
            results.currents_monitor.I_NMDA[neuron_index],
            label=f"I NMDA, {neuron_index}- {neuron_label}",
            alpha=0.6
        )
        axs[3].plot(
            results.currents_monitor.t,
            results.currents_monitor.I_GABA[neuron_index],
            label=f"I GABA, {neuron_index}- {neuron_label}",
            alpha=0.6
        )
        if "I_AMPA_cue" in results.currents_monitor.recorded:
            axs[3].plot(
                results.currents_monitor.t,
                results.currents_monitor.I_AMPA_cue[neuron_index],
                label=f"I Cue, {neuron_index}- {neuron_label}",
                alpha=0.6
            )

    axs[3].legend()
    fig.suptitle(f"Simulation {results.example.label} {f", seed {results.example.seed}" if results.example.in_testing else ''}")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.show()

def to_delayed(examples: list[AnExampleExperiment], func):
    return [delayed(func)(ex) for ex in examples]


def to_examples_experiments(prod: itertools.product):
    return [AnExampleExperiment(G_EE_AMPA=0, G_EE_NMDA=g_EE_NMDA, G_EI=0.292, G_IE=g_IE, G_II=1, NE=800, NI=200,
                                label=f"No Cue. Look for stable dynamics. G_E_NMDA={g_EE_NMDA}, G_IE = {g_IE}",
                                seed=seed) for g_EE_NMDA, g_IE, seed in prod]

import pandas as pd
def run_cartezian_product(prod: itertools.product, func, debug=False):
    n_jobs = 1 if debug else -1
    results = Parallel(n_jobs=n_jobs)(to_delayed(to_examples_experiments(prod), func))

    df = pd.DataFrame(results)
    print("XXXX all stable? ", (df.end_rate < 100).all())
