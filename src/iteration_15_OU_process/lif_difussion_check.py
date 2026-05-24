import unittest

import brian2
import matplotlib.pyplot as plt
import numpy as np
from brian2 import NeuronGroup, StateMonitor, Quantity
from brian2 import run, mV, SpikeMonitor, Hz
from brian2 import second, ms
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from sympy.physics.units import volts

from Plotting import show_plots_non_blocking
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients
from iteration_12_transfer_function_of_lif_neurons.config import default_diffusion_lif_config, DiffusionLIFConfig


def simulate_two_diffussion_approx(T, dt=0.1 * ms, mu=-47.61595645 * mV, sigma=1.90531046 * mV, delta_v=0.7 * mV,
                                   lif_config=default_diffusion_lif_config,
                                   target_rates=[0.05 * Hz, 0.18 * Hz], exp_labels=["MK801", "Control"]):
    V_r = lif_config.V_r
    theta = lif_config.theta
    mean = [mu, mu + delta_v]
    mean_no_mv_units = np.array([mean[0] / mV, mean[1] / mV])
    sigma_no_mv_units = sigma / mV
    N = NeuronGroup(
        2,
        """
        tau : second
        mean: volt
        dv/dt = -(v-mean)/tau + sqrt(2*tau**-1)*sigma*xi : volt
        """,
        method="heun",
        threshold="v >= theta",
        reset="v = V_r",
        refractory=lif_config.tau_rp,
        dt=dt)
    N.mean = mean
    N.tau = lif_config.tau_m
    N.v[:] = lif_config.V_r
    M = StateMonitor(N, ["v"], record=True)
    spikemon = SpikeMonitor(N)
    run(T)

    v_s = M.v / mV
    t = M.t / second
    spike_times = [None] * len(target_rates)
    for neuron_id in [0, 1]:
        spike_times[neuron_id] = spikemon.all_values()['t'][neuron_id] / second
    return t, v_s, spike_times


def simulate_two_diffussion_approx_no_firing(T, dt=0.1 * ms, mu=-47.61595645 * mV, sigma=1.90531046 * mV,
                                             delta_v=0.7 * mV, lif_config=default_diffusion_lif_config,
                                             target_rates=[0.05 * Hz, 0.18 * Hz], exp_labels=["MK801", "Control"]):
    V_r = lif_config.V_r
    theta = lif_config.theta
    mean = [mu, mu + delta_v]
    mean_no_mv_units = np.array([mean[0] / mV, mean[1] / mV])
    sigma_no_mv_units = sigma / mV
    N = NeuronGroup(
        2,
        """
        tau : second
        mean: volt
        dv/dt = -(v-mean)/tau + sqrt(2*tau**-1)*sigma*xi : volt
        """,
        method="heun",
        refractory=lif_config.tau_rp,
        dt=dt)
    N.mean = mean
    N.tau = lif_config.tau_m
    N.v[:] = lif_config.V_r
    M = StateMonitor(N, ["v"], record=True)
    run(T)

    v_s = M.v / mV
    t = M.t / second
    return t, v_s


def filter_spikes_in_time_window(spike_times, start, end):
    if spike_times.size == 0:
        return spike_times

    left = np.searchsorted(spike_times, start, side="left")
    right = np.searchsorted(spike_times, end, side="right")

    return spike_times[left:right]


def smoothen_v(v, smooth_width=30, dt=0.1 * ms):
    return v
    sigma_ms = smooth_width
    kernel_size = sigma_ms / (dt / ms)

    return gaussian_filter1d(v, sigma=kernel_size)


def plot_difussion_approx(script_name, lif_config: DiffusionLIFConfig, sigma: Quantity, means: list[Quantity],
                          target_rates: list[Quantity],
                          v_s: np.ndarray, t: np.ndarray, spike_times: list[np.ndarray],
                          T: Quantity, dt: Quantity, plot_start: Quantity, plot_end: Quantity,
                          exp_label="", seed=None):
    sigma_no_mv_units = sigma / mV

    num_spikes = np.array([len(trace) for trace in spike_times])
    rates = num_spikes / T
    rates_diff = (rates - target_rates) / Hz

    sg = SiegertGradients.for_lif_config(lif_config)
    expected_rates = [sg.firing_rate(means[0] * mV, sigma), sg.firing_rate(means[1] * mV, sigma)]

    means_from_data = v_s.mean(axis=1)
    delta_means = - means_from_data + means

    sigmas_from_data = v_s.std(axis=1)
    sigmas_diff = - sigmas_from_data + sigma_no_mv_units

    plot_start_index = int(plot_start / dt)
    plot_end_index = int(plot_end / dt)

    t_plot = t[plot_start_index:plot_end_index] / second

    fig = plt.figure(figsize=(12, 9))

    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[3, 1, 1.2]
    )

    ax_v = fig.add_subplot(gs[0])
    ax_raster = fig.add_subplot(gs[1], sharex=ax_v)
    ax_table = fig.add_subplot(gs[2])

    colors = {0: "orange", 1: "black"}

    for neuron_id, label, color in zip([0, 1], ["MK801", "Control"], colors.values()):
        v = v_s[neuron_id][plot_start_index:plot_end_index]
        smooth_v = smoothen_v(v)
        ax_v.plot(
            t_plot,
            smooth_v,
            color=color,
            label=label,
            alpha=0.6,
        )

    if lif_config.theta < 0 * mV:
        ax_v.axhline(y=lif_config.theta / mV, color="k", linestyle="--", label="Threshold")

    ax_v.set_ylabel("Membrane voltage [mV]")

    ax_v.set_title(
        exp_label
    )

    ax_v.legend()

    # Raster plot

    for neuron_id in [0, 1]:
        current_spike_times = filter_spikes_in_time_window(spike_times[neuron_id], start=plot_start, end=plot_end)
        ax_raster.scatter(
            current_spike_times,
            [neuron_id] * len(current_spike_times),
            color=colors[neuron_id],
            s=30,
        )

    ax_raster.set_yticks([0, 1])
    ax_raster.set_yticklabels(["MK801", "Control"])

    ax_raster.set_xlabel("t [s]")
    ax_raster.set_ylabel("Spikes")

    grid = False
    if grid:
        ax_v.grid(alpha=0.3)
        ax_raster.grid(alpha=0.3)

    FMT = ".3f"

    # -------------------------
    # Two-level header
    # -------------------------

    header_top = [
        "",
        "", "mean \n [mV]", "",
        "", "sigma \n [mV]", "",
        "", "rate \n [Hz]", "",
        r"$r_0(\mu, \sigma)$", ""
    ]

    header_bottom = [
        "",
        "target", "actual", "diff",
        "target", "actual", "diff",
        "target", "actual", "diff",
        "target \n μ, σ", "actual \n μ, σ"
    ]

    # -------------------------
    # Data rows
    # -------------------------

    table_data = [
        header_top,
        header_bottom,

        [
            "MK801",

            # mean
            f"{means[0]:{FMT}}",
            f"{means_from_data[0]:{FMT}}",
            f"{delta_means[0]:{FMT}}",

            # sigma
            f"{sigma / mV:{FMT}}",
            f"{sigmas_from_data[0]:{FMT}}",
            f"{sigmas_diff[0]:{FMT}}",

            # rate
            f"{target_rates[0] / Hz:{FMT}}",
            f"{rates[0] / Hz:{FMT}}",
            f"{rates_diff[0]:{FMT}}",

            # computed rate
            f"{sg.firing_rate(means[0] * mV, sigma) / Hz:{FMT}}",
            f"{sg.firing_rate(means_from_data[0] * mV, sigmas_from_data[0]) / Hz:{FMT}}"
        ],

        [
            "Control",

            # mean
            f"{means[1]:{FMT}}",
            f"{means_from_data[1]:{FMT}}",
            f"{delta_means[1]:{FMT}}",

            # sigma
            f"{sigma / mV:{FMT}}",
            f"{sigmas_from_data[1]:{FMT}}",
            f"{sigmas_diff[1]:{FMT}}",

            # rate
            f"{target_rates[1] / Hz:{FMT}}",
            f"{rates[1] / Hz:{FMT}}",
            f"{rates_diff[1]:{FMT}}",

            # computed rate
            f"{sg.firing_rate(means[1] * mV, sigma) / Hz:{FMT}}",
            f"{sg.firing_rate(means_from_data[1] * mV, sigmas_from_data[1]) / Hz:{FMT}}"
        ],
    ]

    # -------------------------
    # Draw table
    # -------------------------

    ax_table.axis("off")

    table = ax_table.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.tight_layout()
    show_plots_non_blocking(save_name=f"{script_name}_seed_{seed}")


def simulate_diffusion_process(seed: None, testing: bool = True, mu=-47.61595645 * mV, sigma=1.90531046 * mV,
                               delta_v=0.7 * mV, target_rates=[0.05 * Hz, 0.18 * Hz],
                               lif_config: DiffusionLIFConfig = default_diffusion_lif_config, dt=0.1 * ms):
    if seed is not None:
        brian2.devices.device.seed(seed)

    if testing:
        T = 100 * second

    else:
        T = 10_000 * second

    return simulate_two_diffussion_approx(T, mu=mu, sigma=sigma, delta_v=delta_v, dt=dt, target_rates=target_rates,
                                          lif_config=lif_config)


def simulate_and_plot(T: Quantity, dt=0.1 * ms, testing: bool = True,
                      mu=-47.61595645 * mV, sigma=1.90531046 * mV,
                      delta_v=0.7 * mV, target_rates=[0.05 * Hz, 0.18 * Hz],
                      lif_config: DiffusionLIFConfig = default_diffusion_lif_config,
                      seed = None, exp_label: str = ""):
    if testing:
        plot_start = 0 * second
        plot_end = T
    else:
        plot_start = 3 * second
        plot_end = 5 * second

    t, v_s, spike_times = simulate_diffusion_process(seed=seed, testing=testing, lif_config=lif_config)
    plot_difussion_approx(lif_config=lif_config, sigma=sigma, means=[mu / mV, (mu + delta_v) / mV],
                          target_rates=target_rates,
                          v_s=v_s, t=t, spike_times=spike_times,
                          T=T, dt=dt, plot_start=plot_start, plot_end=plot_end,
                          exp_label=exp_label, seed=seed)


class MyTestCase(unittest.TestCase):

    def test_run_lif_difussion_approximation(self, testing=True, seed=None, lif_config=default_diffusion_lif_config,
                                             exp_label=""):

        if seed is not None:
            brian2.devices.device.seed(seed)

        if testing:
            T = 100 * second
            plot_start = 0 * second
            plot_end = T
        else:
            T = 10_000 * second

            plot_start = 3 * second
            plot_end = 5 * second

        dt = 0.1 * ms

        delta_v = 0.7 * mV
        target_rates = [0.05 * Hz, 0.18 * Hz]
        mu = -47.61595645 * mV
        sigma = 1.90531046 * mV
        t, v_s, spike_times = simulate_diffusion_process(seed=seed, testing=testing, lif_config=lif_config)
        plot_difussion_approx(lif_config=lif_config, sigma=sigma, means=[mu / mV, (mu + delta_v) / mV],
                              target_rates=target_rates,
                              v_s=v_s, t=t, spike_times=spike_times,
                              T=T, dt=dt, plot_start=plot_start, plot_end=plot_end,
                              exp_label=exp_label, seed=seed, script_name=f"{self._testMethodName}_{seed}")

    def test_diffusion_approx(self):

        Parallel(n_jobs=-3, prefer="processes")(
            delayed(self.test_run_lif_difussion_approximation)(
                testing=True,
                seed=seed,
                exp_label=f"Verification for numerical solution seed {seed}",
            )
            for seed in range(1, 20)
        )

    def test_run_no_firing(self):

        not_firing_config = default_diffusion_lif_config.with_property(DiffusionLIFConfig.KEY_NEURON_THRESHOLD, 100)
        for seed in range(1, 20):
            self.test_run_lif_difussion_approximation(testing=False, seed=seed,
                                                      exp_label=f"Check OU process μ, σ.  {seed}",
                                                      lif_config=not_firing_config)


if __name__ == '__main__':
    unittest.main()
