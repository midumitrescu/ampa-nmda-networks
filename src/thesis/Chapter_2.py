import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import mV, Hz

from Plotting import show_plots_non_blocking, prepare_bigger_fonts
from iteration_12_transfer_function_of_lif_neurons.LookForAllSolutionsMuSigma import \
    compute_mu_to_sigma_curve, plot_line_computation_vs_fit
from iteration_12_transfer_function_of_lif_neurons.SiegerGradientDescentTestCases import \
    compute_LIF_curves_for_mus_sigmas
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients, \
    newton_fsolve_find_mu_for_fixed_sigma, I_mu_sigma, integration_limits
from iteration_12_transfer_function_of_lif_neurons.config import default_diffusion_lif_config

rate_palmer_control = 0.18 * Hz
rate_palmer_mk_801 = 0.05 * Hz


class Chapter2Figures(unittest.TestCase):

    def test_figure_1_plot_lif_firing_with_zoom_in(self):
        mus = np.linspace(-65, -35, 1001) * mV
        sigmas = np.array([0, 2., 4., 6.]) * mV
        lif_config = default_diffusion_lif_config
        rates = compute_LIF_curves_for_mus_sigmas(mus, sigmas, lif_config=lif_config)

        sg = SiegertGradients.for_lif_config(lif_config)
        example_sigma = 4.0 * mV
        mu_rate_control = newton_fsolve_find_mu_for_fixed_sigma(siegert_gradient=sg, sigma_v=example_sigma,
                                                                r_target=rate_palmer_control)
        mu_rate_mk_801 = newton_fsolve_find_mu_for_fixed_sigma(siegert_gradient=sg, sigma_v=example_sigma,
                                                               r_target=rate_palmer_mk_801)

        fig = plt.figure(figsize=(16, 8))
        prepare_bigger_fonts()

        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

        ax_leg = fig.add_subplot(gs[:, 1])
        ax_leg.axis("off")  # hide legend axis frame

        for index, ax in enumerate([ax1, ax2]):

            ax.axhline(y=0.05, color='orange', linestyle='--',
                       label='rate from Palmer 2014, Figure 2e \n MK801, 0.05 Hz')
            ax.axhline(y=0.18, color='black', linestyle='-.',
                       label='rate from Palmer 2014, Figure 2e \n Control, 0.18 Hz')
            ax.axvline(x=lif_config.theta / mV, color='dimgray', linestyle='-.', label=r'Threshold $\theta$')

            for sigma, lif_values in zip(sigmas, rates):
                ax.plot(mus / mV, lif_values / Hz, label=fr'$\sigma_v={sigma / mV}$ mV', alpha=0.6, lw=3)

            ax.set_xlabel("Driving force $V_m$ ($\mu_v$) [mV]")
            ax.set_ylabel("Firing rate [Hz]")

            ax.text(
                0.02, 1.17, f"({chr(ord("A") + index)})",
                transform=ax.transAxes,
                fontsize=20,
                fontweight=1000,
                va="top",
                ha="left"
            )

        ax2.plot(mu_rate_control / mV, sg.firing_rate(mu_v=mu_rate_control, sigma_v=example_sigma) / Hz,
                 marker='x', color='C0', markeredgewidth=1, markersize=12, linestyle="", alpha=0.7,
                 label=fr"$\mu_{{\mathrm{{sol}}$ for $rate_{{\mathrm{{LIF}} = {rate_palmer_control / Hz : .2f}$")

        ax2.plot(mu_rate_mk_801 / mV, sg.firing_rate(mu_v=mu_rate_mk_801, sigma_v=example_sigma) / Hz,
                 marker='x', color='C0', alpha=1, markeredgewidth=0.8, markersize=12,
                 label=fr"$\mu_{{\mathrm{{sol}}$ for $rate_{{\mathrm{{LIF}} = {rate_palmer_mk_801 / Hz : .2f}$")

        ax2.set_ylim(0, 0.2)

        ax2.axvline(x=mu_rate_control / mV, ymin=0.1, ymax=0.95, color='C0', linestyle='--', alpha=0.8)
        ax2.axvline(x=mu_rate_mk_801 / mV, ymin=0.1, ymax=0.3, color='C0', linestyle='--', alpha=0.8)

        ax2.annotate(
            text="",
            xy=(mu_rate_mk_801 / mV, 0.03),
            xytext=(mu_rate_control / mV, 0.03),
            arrowprops=dict(
                arrowstyle="<->",
                color="black",
                lw=1.5
            )
        )
        ax2.text(
            x=(mu_rate_control / mV + mu_rate_mk_801 / mV) / 2,
            y=0.007,
            s=r"$\Delta \mu_v$",
            fontsize=14,
            ha="center",
            va="bottom"
        )

        handles, labels = ax1.get_legend_handles_labels()

        order = np.hstack((np.arange(0, len(sigmas)) + 3, [0, 1, 2]))

        ax_leg.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            loc="center left",
            frameon=False,
            labelspacing=1.6,
            fontsize=18
        )

        ax1.set_title("Plot of $r_0(\mu_v, \sigma_v)$, including experimental rates to be fitted")
        ax2.set_title("Detail of $r_0(\mu_v, \sigma_v)$ in the range of the rates to be fitted")
        fig.suptitle("$r_0(\mu_v, \sigma_v)$ as predicted by Siegert's first passage time formula")

        fig.tight_layout()
        show_plots_non_blocking(caller_test_case=self)

    def test_plot_sigma_as_function_of_mu_v(self):

        nmda_block_mu_to_sigma = compute_mu_to_sigma_curve(
            default_diffusion_lif_config.with_label("MK-801"), r_target=0.05 * Hz
        )
        control_mu_to_sigma = compute_mu_to_sigma_curve(
            default_diffusion_lif_config.with_label("Control"), r_target=0.18 * Hz
        )

        large_rate_mu_to_sigma = compute_mu_to_sigma_curve(
            default_diffusion_lif_config.with_label("Example high rate"), r_target=25 * Hz
        )
        sg = SiegertGradients.for_lif_config(default_diffusion_lif_config)
        # find sigma closest to some value:
        example_sigma = 4.0

        mu_rate_control = newton_fsolve_find_mu_for_fixed_sigma(siegert_gradient=sg, sigma_v=example_sigma * mV,
                                                                r_target=rate_palmer_control)
        mu_rate_mk_801 = newton_fsolve_find_mu_for_fixed_sigma(siegert_gradient=sg, sigma_v=example_sigma * mV,
                                                               r_target=rate_palmer_mk_801)

        print(f"Delta mu: {mu_rate_control - mu_rate_mk_801}")

        results = [nmda_block_mu_to_sigma, control_mu_to_sigma]

        prepare_bigger_fonts()

        fig = plt.figure(figsize=(16, 8))
        prepare_bigger_fonts()

        gs = fig.add_gridspec(1, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)

        """Plot μ vs σ curves and linear fit. Used by script runners with caller_test_case=self for figure naming."""
        for result, color in zip(results, ["orange", "black", "blue"]):
            ax1.plot(result.mus, result.sigmas, color=color, label=f"{result.exp_label}, r = {result.r_target / Hz} Hz",
                     lw=2)

        ax1.plot(mu_rate_control / mV, example_sigma,
                 marker='x', color='C0', markeredgewidth=1, markersize=12, linestyle="", alpha=0.7,
                 label=r"$\mu_\mathrm{sol}$ for $rate_\mathrm{LIF} = " + f"{rate_palmer_control / Hz : .2f}$")

        ax1.plot(mu_rate_mk_801 / mV, example_sigma,
                 marker='x', color='C0', markeredgewidth=1, markersize=12, linestyle="", alpha=0.7,
                 label=r"$\mu_\mathrm{sol}$ for $rate_\mathrm{LIF} =" + f"{rate_palmer_mk_801 / Hz : .2f}$")

        ax1.axvline(x=mu_rate_control / mV, ymin=0.1, ymax=0.8, color='C0', linestyle='--', alpha=0.8)
        ax1.axvline(x=mu_rate_mk_801 / mV, ymin=0.1, ymax=0.8, color='C0', linestyle='--', alpha=0.8)

        ax1.axhline(y=example_sigma, xmin=0, xmax=0.4, color='C0', linestyle='--', alpha=0.8,
                    label=r"$\sigma_v = $" + f"{example_sigma: .2f} mV")

        ax1.annotate(
            text="",
            xy=(mu_rate_mk_801 / mV, 0.7),
            xytext=(mu_rate_control / mV, 0.7),
            arrowprops=dict(
                arrowstyle="<->",
                color="black",
                lw=1.5
            )
        )
        ax1.text(
            x=(mu_rate_control / mV + mu_rate_mk_801 / mV) / 2,
            y=0.2,
            s=r"$\Delta \mu_v$",
            fontsize=14,
            ha="center",
            va="bottom"
        )

        """Plot μ vs σ curves and linear fit. Used by script runners with caller_test_case=self for figure naming."""
        for result, color in zip([nmda_block_mu_to_sigma, control_mu_to_sigma, large_rate_mu_to_sigma],
                                 ["orange", "black", "blue"]):
            ax2.plot(result.mus, result.sigmas, color=color, label=f"{result.exp_label}, r = {result.r_target / Hz} Hz",
                     lw=2)

        for index, ax in enumerate([ax1, ax2]):
            ax.axvline(x=default_diffusion_lif_config.theta / mV, color='dimgray', linestyle='-.',
                       label=r'Threshold $\theta$')
            ax.set_xlabel(r"$\mu_v$ [mV]")
            ax.set_ylabel(r"$\sigma_v$ [mV]")

            ax.text(
                0.02, 1.1, f"({chr(ord("A") + index)})",
                transform=ax.transAxes,
                fontsize=20,
                fontweight=1000,
                va="top",
                ha="left"
            )

            ax.legend()

        ax1.set_title("For reported spontaneous firing rates")
        ax2.set_title("For comparison low vs high firing rates")

        fig.suptitle(
            r"Predicted $\sigma_v = f(\mu_v)$ curve for $r_0(\mu_v, \sigma_v)$ = constant"
            "\n"
            r"using Siegert's first passage time formula",
            ha='center'
        )
        fig.tight_layout()
        show_plots_non_blocking(caller_test_case=self)
        plot_line_computation_vs_fit([nmda_block_mu_to_sigma, control_mu_to_sigma, large_rate_mu_to_sigma],
                                     colors=("orange", "black", "blue"),
                                     caller_test_case=self,
                                     descriptor="linear_fit")

    def test_check_integral_limits_and_linear_fit(self):
        nmda_block_mu_to_sigma = compute_mu_to_sigma_curve(
            default_diffusion_lif_config.with_label("MK-801"), r_target=0.05 * Hz
        )
        control_mu_to_sigma = compute_mu_to_sigma_curve(
            default_diffusion_lif_config.with_label("Control"), r_target=0.18 * Hz
        )

        large_rate_mu_to_sigma = compute_mu_to_sigma_curve(
            default_diffusion_lif_config.with_label("Example high rate"), r_target=25 * Hz
        )
        results = [nmda_block_mu_to_sigma, control_mu_to_sigma, large_rate_mu_to_sigma]

        siegert_gradients = SiegertGradients.for_lif_config(default_diffusion_lif_config)


        firing_rates = np.zeros(shape=(len(results), len(nmda_block_mu_to_sigma.mus)))
        integral_limits = np.zeros(shape=(len(results), 2, len(nmda_block_mu_to_sigma.mus)))
        integral_values = np.zeros(shape=(len(results), len(nmda_block_mu_to_sigma.mus)))

        for index, result in enumerate([nmda_block_mu_to_sigma, control_mu_to_sigma, large_rate_mu_to_sigma]):
            for row_index, (mu, sigma) in enumerate(zip(result.mus, result.sigmas)):
                firing_rates[index, row_index] = siegert_gradients.firing_rate(mu_v=mu, sigma_v=sigma)
                integral_limits[index, :, row_index] = integration_limits(V_mean=mu, sigma_v=sigma,
                                                                          theta=default_diffusion_lif_config.theta,
                                                                          V_reset=default_diffusion_lif_config.V_r)
                integral_values[index, row_index] = I_mu_sigma(mu_v=mu, sigma_v=sigma,
                                                               theta=default_diffusion_lif_config.theta,
                                                               V_reset=default_diffusion_lif_config.V_r)

        prepare_bigger_fonts()

        fig = plt.figure(figsize=(26, 8))
        gs = fig.add_gridspec(4, 2)

        ax_integral_limits_mmda_block = fig.add_subplot(gs[0, 0])
        ax_integral_limits_zoom_nmda_block = fig.add_subplot(gs[0, 1])

        ax_integral_limits_control = fig.add_subplot(gs[1, 0])
        ax_integral_limits_zoom_control = fig.add_subplot(gs[1, 1])

        ax_integral_limits_high_rate = fig.add_subplot(gs[2, 0])
        ax_integral_limits_zoom_high_rate = fig.add_subplot(gs[2, 1])

        axs_integral_limits = [ax_integral_limits_mmda_block, ax_integral_limits_control, ax_integral_limits_high_rate]
        axs_integral_limits_zoom = [ax_integral_limits_zoom_nmda_block, ax_integral_limits_zoom_control,
                                    ax_integral_limits_zoom_high_rate]

        ax_integral_values = fig.add_subplot(gs[3, 0])
        ax_rates_computation = fig.add_subplot(gs[3, 1], sharex=ax_integral_values)

        for index, result in enumerate(results):
        #for index, result in enumerate([nmda_block_mu_to_sigma]):
            # lower limit
            axs_integral_limits[index].plot(
                result.mus,
                integral_limits[index, 0, :],
                label=f"{result.exp_label} (lower limit)"
            )

            # upper limit
            axs_integral_limits[index].plot(
                result.mus,
                integral_limits[index, 1, :],
                linestyle='--',
                label=f"{result.exp_label} (upper limit)"
            )

            delta = integral_limits[index, 1, :] - integral_limits[index, 0, :]
            axs_integral_limits_zoom[index].plot(nmda_block_mu_to_sigma.mus,
                                                 delta,
                                                 label=f"Upper limit - lower limit {result.exp_label}")

            axs_integral_limits_zoom[index].set_ylim(top = np.max(delta) / 100)

        # ax_integral_limits_zoom.set_ylim((-2, 10))
        # ax_integral_limits_zoom.set_ylim(top=10)

        ax_integral_values.plot(nmda_block_mu_to_sigma.mus, integral_values.T,
                                label=[result.exp_label for result in results])
        ax_rates_computation.plot(nmda_block_mu_to_sigma.mus, firing_rates.T,
                                  label=[result.exp_label for result in results])

        for index, result in enumerate(results):
            axs_integral_limits[index].set_title(f"Plot of integral limits for {result.exp_label}\n" +
                                                 "upper limit " + r"$\frac{\mu_v - \theta}{\sqrt{2}\sigma_v}$ and lower limit " + r"$\frac{\mu_v - V_R}{\sqrt{2}\sigma_v}$"
                                                 )
            axs_integral_limits[index].set_ylabel(r"Integral limits (unitless)")
            axs_integral_limits_zoom[index].set_title(f"upper limit -  lower limit for {result.exp_label}")
            axs_integral_limits_zoom[index].set_ylabel(r"$\Delta$ limits (unitless)")

        ax_integral_values.set_ylabel(r"Integral value (unitless)")
        ax_rates_computation.set_ylabel(r"predicted rate [Hz]")

        for ax in [ax_integral_values, ax_rates_computation]:
            ax.set_xlabel(r"$\mu_v$ [mV]")
            ax.axvline(x=default_diffusion_lif_config.theta / mV, color='dimgray', linestyle='-.',
                       label=r'Threshold $\theta$')
            ax.legend()

        ax_integral_values.set_title(
            r"$\int_{\frac{\mu_v - \theta}{\sqrt{2}\sigma_v}}^{\frac{\mu_v - V_R}{\sqrt{2}\sigma_v}} "
            r"e^{x^2}\,\mathrm{erfc}(x)\,dx$"
        )
        ax_rates_computation.set_title(
            "$r_0(\mu_v, \sigma_v)$" + " computed with actual numerical values \n returned by Newton's method")

        fig.tight_layout()
        show_plots_non_blocking(caller_test_case=self)
