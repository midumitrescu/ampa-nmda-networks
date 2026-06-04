from abc import abstractmethod

import brian2
import numpy as np
from brian2 import NeuronGroup, StateMonitor, Quantity
from brian2 import run, mV, SpikeMonitor, Hz
from brian2 import second, ms
from matplotlib import pyplot as plt

from Plotting import show_plots_non_blocking
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients
from iteration_12_transfer_function_of_lif_neurons.config import default_diffusion_lif_config, DiffusionLIFConfig
from iteration_15_OU_process.plot_utils import smoothen_v, filter_spikes_in_time_window, exp_label_to_folder_name


class DiffusionSimulation:

    def __init__(self, seed: None, mu=-47.61595645 * mV, sigma=1.90531046 * mV,
                 delta_v=0.7 * mV, target_rates=[0.05 * Hz, 0.18 * Hz],
                 lif_config: DiffusionLIFConfig = default_diffusion_lif_config, dt=0.1 * ms, sim_method="heun",
                 testing: bool = False):
        self.seed = seed
        self.testing = testing

        if testing:
            T = 10 * second
        else:
            T = 60*60 * second

        self.T = T
        self.dt = dt

        self.mu = mu
        self.sigma = sigma
        self.delta_v = delta_v
        self.target_rates = target_rates
        self.lif_config = lif_config

        self.means = [self.mu, self.mu + self.delta_v]

        self.sim_method = sim_method

    @abstractmethod
    def _set_seed__(self):
        pass

    @abstractmethod
    def simulate_two_diffusion_approx(self):
        pass

    def simulate_and_plot(self, exp_label=""):

        if self.testing:
            plot_start = 0 * ms
            plot_end = self.T
        else:
            plot_start = 1 * second
            plot_end = 15 * second

        t, dv, spike_times = self.simulate_two_diffusion_approx()
        self.plot_diffusion_approx(t=t, dv=dv, spike_times=spike_times, plot_start=plot_start, plot_end=plot_end, exp_label=exp_label)

    def plot_diffusion_approx(self, v_s: np.ndarray, t: np.ndarray, spike_times: list[np.ndarray],
                              plot_start: Quantity, plot_end: Quantity,
                              exp_label=""):

        T = self.T
        dt = self.dt
        seed = self.seed

        sigma_no_mv_units = self.sigma / mV

        num_spikes = np.array([len(trace) for trace in spike_times])
        rates = num_spikes / T
        rates_diff = (rates - self.target_rates) / Hz

        sg = SiegertGradients.for_lif_config(self.lif_config)

        means_from_data = v_s.mean(axis=1) * mV
        delta_means = means_from_data - self.means

        sigmas_from_data = v_s.std(axis=1) * mV
        sigmas_diff = sigmas_from_data - self.sigma

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

        if self.lif_config.theta < 0 * mV:
            ax_v.axhline(y=self.lif_config.theta / mV, color="k", linestyle="--", label="Threshold")

        ax_v.set_ylabel("Membrane voltage [mV]")

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
                f"{self.means[0] / mV:{FMT}}",
                f"{means_from_data[0] / mV:{FMT}}",
                f"{delta_means[0] / mV:{FMT}}",

                # sigma
                f"{self.sigma / mV:{FMT}}",
                f"{sigmas_from_data[0] / mV:{FMT}}",
                f"{sigmas_diff[0] / mV:{FMT}}",

                # rate
                f"{self.target_rates[0] / Hz:{FMT}}",
                f"{rates[0] / Hz:{FMT}}",
                f"{rates_diff[0]:{FMT}}",

                # computed rate
                f"{SiegertGradients.default().firing_rate(self.means[0], self.sigma) / Hz:{FMT}}",
                f"{SiegertGradients.default().firing_rate(means_from_data[0], sigmas_from_data[0]) / Hz:{FMT}}"
            ],

            [
                "Control",

                # mean
                f"{self.means[1] / mV:{FMT}}",
                f"{means_from_data[1] / mV:{FMT}}",
                f"{delta_means[1] / mV:{FMT}}",

                # sigma
                f"{self.sigma / mV:{FMT}}",
                f"{sigmas_from_data[1] / mV:{FMT}}",
                f"{sigmas_diff[1] / mV:{FMT}}",

                # rate
                f"{self.target_rates[1] / Hz:{FMT}}",
                f"{rates[1] / Hz:{FMT}}",
                f"{rates_diff[1]:{FMT}}",

                # computed rate
                f"{SiegertGradients.default().firing_rate(self.means[1], self.sigma) / Hz:{FMT}}",
                f"{SiegertGradients.default().firing_rate(means_from_data[1], sigmas_from_data[1]) / Hz:{FMT}}"
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

        fig_title = exp_label if seed is None else f"{exp_label} (seed={seed})"
        fig.suptitle(fig_title)

        plt.tight_layout()
        show_plots_non_blocking(save_name=f"{exp_label_to_folder_name(exp_label)}_seed_{seed}")


class Brian2DiffusionSimulation(DiffusionSimulation):

    def __init__(self, mu=-47.61595645 * mV, sigma=1.90531046 * mV, delta_v=0.7 * mV, target_rates=[0.05 * Hz, 0.18 * Hz],
                 lif_config: DiffusionLIFConfig = default_diffusion_lif_config, dt=0.1 * ms, sim_method="heun",
                 seed: int = None, testing: bool = False):
        super().__init__(mu=mu, sigma=sigma, delta_v=delta_v, target_rates=target_rates, lif_config=lif_config,
                         dt=dt, sim_method=sim_method, seed=seed, testing=testing)

    def _set_seed__(self):
        if self.seed is not None:
            brian2.devices.device.seed(self.seed)

    def simulate_two_diffusion_approx(self):
        self._set_seed__()

        sigma = self.sigma
        tau = self.lif_config.tau_m
        V_r = self.lif_config.V_r
        theta = self.lif_config.theta

        N = NeuronGroup(
            2,
            """
            mean: volt
            dv/dt = -(v-mean)/tau + sqrt(2*tau**-1)*sigma*xi : volt
            """,
            method=self.sim_method,
            threshold="v >= theta",
            reset="v = V_r",
            refractory=self.lif_config.tau_rp,
            dt=self.dt)
        N.mean = self.means
        N.v[:] = self.lif_config.V_r

        M = StateMonitor(N, ["v"], record=True)
        spikemon = SpikeMonitor(N)

        run(self.T)

        v_s = M.v / mV
        t = M.t / second
        spike_times = [None] * len(self.target_rates)
        for neuron_id in [0, 1]:
            spike_times[neuron_id] = spikemon.all_values()['t'][neuron_id] / second
        return t, v_s, spike_times


class NativeDiffusionSimulation(DiffusionSimulation):

    def __init__(self, mu=-47.61595645 * mV, sigma=1.90531046 * mV, delta_v=0.7 * mV, target_rates=[0.05 * Hz, 0.18 * Hz],
                 lif_config: DiffusionLIFConfig = default_diffusion_lif_config, dt=0.1 * ms, sim_method="euler-maruyama",
                 seed: int = None, testing: bool = False):
        super().__init__(mu=mu, sigma=sigma, delta_v=delta_v, target_rates=target_rates, lif_config=lif_config,
                         dt=dt, sim_method=sim_method, seed=seed, testing=testing)

    def _set_seed__(self):
        np.random.seed(self.seed)

    def simulate_two_diffusion_approx(self):
        self._set_seed__()

        """
        Units:
            time    -> ms
            voltage -> mV
        """
        v_r = self.lif_config.V_r / mV
        sigma = self.sigma / mV
        theta = self.lif_config.theta / mV
        tau_rp = self.lif_config.tau_rp / ms


        n_steps = int(self.T / self.dt)

        t = np.arange(n_steps) * self.dt

        means = np.array([self.mu / mV, (self.mu + self.delta_v) / mV])

        v = np.zeros((2, n_steps))
        v[:, 0] = v_r

        spike_times = [[], []]

        refractory_until = np.zeros(2) * ms

        # Euler-Maruyama coefficients
        dt_over_tau = self.dt / self.lif_config.tau_m

        sigma_times_sqrt_dt_over_tau_m = sigma * np.sqrt(2.0 * self.dt / self.lif_config.tau_m)

        for k in range(n_steps - 1):

            current_t = t[k]

            for i in range(2):

                # refractory period
                if current_t < refractory_until[i]:
                    v[i, k + 1] = v_r
                    continue

                eta = np.random.randn()

                dv = (means[i] - v[i, k]) * dt_over_tau + sigma_times_sqrt_dt_over_tau_m * eta
                v_new = v[i, k] + dv

                # spike
                if v_new >= theta:
                    spike_times[i].append(current_t)
                    v[i, k + 1] = v_r

                    refractory_until[i] = current_t + self.lif_config.tau_rp
                else:
                    v[i, k + 1] = v_new

        return t, v, [np.array(spike_times[0]), np.array(spike_times[1])]

