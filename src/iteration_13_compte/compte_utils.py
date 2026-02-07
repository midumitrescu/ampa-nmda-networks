from brian2 import PopulationRateMonitor, SpikeMonitor, StateMonitor, ms, Hz
import matplotlib.pyplot as plt
from brian2.units.allunits import pampere

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from iteration_13_compte.configs import AnExampleExperiment, CompteResults

def compute_ordered_firing_rates(example: AnExampleExperiment, spikes_monitor: SpikeMonitor, runtime,
                                 theta_array_assignment):
    bin_size = 10 * ms
    bins = int(runtime / bin_size)

    rates = np.zeros((example.NE, bins))
    for i in range(example.NE):
        spikes = spikes_monitor.spike_trains()[i] / ms
        counts, _ = np.histogram(spikes, bins=bins, range=(0, runtime / ms))
        rates[i] = counts / bin_size

    order = np.argsort(theta_array_assignment)
    rates_sorted = rates[order]
    return rates_sorted

def to_spike_trains(spikes_monitor: SpikeMonitor, neurons: list[int], sim_time):
    neurons = np.array(neurons)  # e.g. [3, 7, 12, 20]
    dt = spikes_monitor.clock.dt
    dt_ms = dt / ms

    T = int(np.ceil(sim_time / dt))
    N = len(neurons)

    raster = np.zeros((N, T), dtype=np.uint8)
    # Spike data
    spikes_neuron_index = np.array(spikes_monitor.i)
    t = np.array(spikes_monitor.t / ms)

    # Convert spike times to indices
    time_idx = (t / dt_ms).astype(np.int32)

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
    dt_ms = dt / ms
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

def plot_compte(sim_time, population_rate_monitor: PopulationRateMonitor, spikes_monitor: SpikeMonitor,
                currents_monitor: StateMonitor, theta_array_assignment, example: AnExampleExperiment):
    rates_sorted = compute_ordered_firing_rates(example=example,
                                                spikes_monitor=spikes_monitor,
                                                runtime=sim_time,
                                                theta_array_assignment=theta_array_assignment)

    fig, axs = plt.subplots(
        4, 1,
        figsize=(20, 16),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 2, 1, 1]}
    )

    im = axs[0].imshow(
        rates_sorted,
        aspect='auto',
        origin='lower',
        cmap='jet',
        extent=[0, sim_time / ms, 0, example.NE]
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

    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(im, cax=cax, label="Firing rate (Hz)")

    axs[0].set_title('Bump attractor dynamics')

    neurons = [0, 200]
    neuron_labels = ["non-cue", "cue"]

    # --- 2) Raster plot ---
    axs[1].plot(
        spikes_monitor.t / ms,
        spikes_monitor.i,
        ".",
        markersize=1,
        color="blue"
    )

    axs[1].set_yticks(ytick_indices)
    axs[1].set_yticklabels([f"{a}°" for a in angle_ticks])
    axs[1].set_ylabel("Preferred angle (°)")

    raster = to_spike_trains(spikes_monitor, neurons=neurons, sim_time=sim_time)
    rates = raster_to_rates(raster, spikes_monitor.clock.dt)

    axs[2].plot(
        population_rate_monitor.t / ms,
        population_rate_monitor.smooth_rate(width=10 * ms),
        label="Population",
        color="black",
        linewidth=2
    )
    for neuron_index, rate in zip(neurons, rates):
        axs[2].plot(
            population_rate_monitor.t / ms,
            rate,
            label=f"Neuron {neuron_index}"
        )

    axs[2].set_ylabel('Rate (Hz)')
    axs[2].legend(loc="upper right")

    for neuron_index, neuron_label in zip(neurons, neuron_labels):
        axs[3].plot(
            currents_monitor.t / ms,
            currents_monitor.I_AMPA[neuron_index] / pampere,
            label=f"I AMPA, {neuron_index} - {neuron_label}",
            alpha=0.6
        )
        axs[3].plot(
            currents_monitor.t / ms,
            currents_monitor.I_NMDA[neuron_index] / pampere,
            label=f"I NMDA, {neuron_index}- {neuron_label}",
            alpha=0.6
        )
        axs[3].plot(
            currents_monitor.t / ms,
            currents_monitor.I_GABA[neuron_index] / pampere,
            label=f"I GABA, {neuron_index}- {neuron_label}",
            alpha=0.6
        )
        if "I_AMPA_cue" in currents_monitor.needed_variables:
            axs[3].plot(
                currents_monitor.t / ms,
                currents_monitor.I_AMPA_cue[neuron_index] / pampere,
                label=f"I Cue, {neuron_index}- {neuron_label}",
                alpha=0.6
            )

    axs[3].legend()
    fig.suptitle(f"Simulation {example.label} {example.seed if example.in_testing else ''}")

    plt.tight_layout()
    fig.show()

