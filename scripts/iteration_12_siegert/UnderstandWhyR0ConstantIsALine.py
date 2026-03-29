import itertools
import sys
import unittest

from loguru import logger
from matplotlib.rcsetup import cycler

from Plotting import prepare_bigger_fonts, show_plots_non_blocking, add_panel_info
from iteration_12_siegert.look_for_all_mu_sigma_for_fixed_rate_scripts import d_sigma_over_d_mu
from iteration_12_transfer_function_of_lif_neurons.config import default_diffusion_lif_config, DiffusionLIFConfig

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc
from brian2 import mV, Hz

from iteration_12_transfer_function_of_lif_neurons.LookForAllSolutionsMuSigma import (
    compute_mu_to_sigma_curve_for_experiment,
    plot_mus_vs_sigmas,
    compute_mu_to_sigma_curve, binary_search_sigma_at_mu_for_firing_rate,
)
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import (
    SiegertGradients, erfcx, )
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import NeuronModelParams
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control

class CheckWhyAllSolutionsAreOnALine(unittest.TestCase):
    def test_understand_why_solutions_lie_on_a_line(self):
        experiment = palmer_control
        siegert_gradients = SiegertGradients.for_experiment(experiment)
        nmda_block_mu_to_sigma = compute_mu_to_sigma_curve_for_experiment(
            experiment.with_label("NMDA Block"), r_target=0.05 * Hz
        )

        print(nmda_block_mu_to_sigma.mus[500], nmda_block_mu_to_sigma.sigmas[500])

        firing_rate = [siegert_gradients.firing_rate(mu_v=mu, sigma_v=sigma) for mu, sigma in zip(
            nmda_block_mu_to_sigma.mus, nmda_block_mu_to_sigma.sigmas
        )]

        firing_rate = np.array(firing_rate).round(8)
        plt.plot(nmda_block_mu_to_sigma.mus, firing_rate)
        plt.show()

        integral_limits_nmda_block = [siegert_gradients.integration_limits(mu_v=mu, sigma_v=sigma) for mu, sigma in zip(
            nmda_block_mu_to_sigma.mus, nmda_block_mu_to_sigma.sigmas
        )]

        plt.plot(nmda_block_mu_to_sigma.mus, np.array(integral_limits_nmda_block), label=["lower limit", "upper limit"])
        # plt.plot(np.array(integral_limits_nmda_block)[0], label=["lower limit"])
        # plt.ylim((-3, 5))
        plt.legend()
        plt.show()

    def test_sieger_value_understand_why_lower_limit_is_higher_than_upper_limit(self):
        mu = -60.04004004004004 * mV
        sigma = 2.5299242694745767 * mV

        siegert_gradient = SiegertGradients.for_experiment(palmer_control)
        lower_bound, upper_bound = siegert_gradient.integration_limits(mu_v=mu, sigma_v=sigma)
        print(f"lower bound {lower_bound: .7f}. Errfc is {erfcx(lower_bound)}")
        print(f"upper bound {upper_bound: .7f}. Errfc is {erfcx(upper_bound)}")

    def test_understand_d_mu_d_sigma_for_high_rate(self):
        experiment = palmer_control
        palmer_control.with_property(NeuronModelParams.KEY_NEURON_V_R, -55)
        mu_to_sigma_low_rate = compute_mu_to_sigma_curve_for_experiment(
            experiment.with_label("Low Rate"), r_target=0.3 * Hz
        )

        mu_to_sigma_high_rate = compute_mu_to_sigma_curve_for_experiment(
            experiment.with_label("High Rate"), r_target=10 * Hz
        )

        plot_mus_vs_sigmas(
            [mu_to_sigma_low_rate, mu_to_sigma_high_rate],
            caller_test_case=self,
        )

    def test_compare_phi_lower_limit_vs_phi_upper_limit(self):
        with plt.rc_context({
            'axes.prop_cycle': cycler(color=['#2ca02c', '#d62728', '#9467bd', '#8c564b']),
            'lines.linewidth': 2.5
        }):
            siegert_gradients = SiegertGradients.for_lif_config(default_diffusion_lif_config)

            # conditions = [default_diffusion_lif_config.with_label("MK-801"), default_diffusion_lif_config.with_label("Control")]
            conditions = [default_diffusion_lif_config.with_label("MK-801"),
                          default_diffusion_lif_config.with_label("Control"),
                          default_diffusion_lif_config.with_label("High rate")]
            target_rates = [0.05 * Hz, 0.18 * Hz, 25 * Hz]
            integral_limits_limits = [(-3, 100), (-3, 100), (-3, 25)]

            results = [compute_mu_to_sigma_curve(condition, rate) for condition, rate in zip(conditions, target_rates)]

            for result, integral_limits_limit in zip(results, integral_limits_limits):
                int_limits = np.zeros(shape=(2, len(result.mus)))
                phi = np.zeros(shape=(2, len(result.mus)))
                d_sigma_over_d_mus = np.zeros_like(result.mus)
                grad_ratios = np.zeros_like(result.mus)

                for row_index, (mu, sigma) in enumerate(zip(result.mus, result.sigmas)):
                    int_limits[:, row_index] = siegert_gradients.integration_limits(mu_v=mu, sigma_v=sigma)
                    lower_limit, upper_limit = siegert_gradients.integration_limits(mu_v=mu, sigma_v=sigma)
                    phi[0, row_index] = siegert_gradients.phi(lower_limit)
                    phi[1, row_index] = siegert_gradients.phi(upper_limit)

                    d_sigma_over_d_mus[row_index] = d_sigma_over_d_mu(mu, sigma, siegert_gradients)
                    grad_ratios[row_index] = - siegert_gradients.d_rate_d_mu(mu,
                                                                             sigma) / siegert_gradients.d_rate_d_sigma(
                        mu, sigma)

                prepare_bigger_fonts()
                fig, axs = plt.subplots(7, 1, figsize=(10, 12))

                ax_integral_limits, ax_phi_lower, ax_phi_upper, ax_phi, ax_ratio, ax_f, ax_grad_ratio = axs

                ax_integral_limits.plot(result.mus, int_limits[0, :], label=f"lower integration limit")
                ax_integral_limits.plot(result.mus, int_limits[1, :], label=f"upper integration limit")
                ax_integral_limits.set_ylim(integral_limits_limit)
                ax_integral_limits.set_title(
                    r"Plot lower limit $\frac{\mu_v - \theta}{\sqrt{2}\sigma_v}$ and upper limit $\frac{\mu_v - V_R}{\sqrt{2}\sigma_v}$ = C\\Are upper and lower limits constant on $r_0(\mu_v, \sigma_v)$ = C?")

                ax_phi_lower.plot(result.mus, phi[0, :], color="#2ca02c")
                ax_phi_lower.set_title(
                    r"$\Phi(\frac{\mu_v - \theta}{\sqrt{2}\sigma_v})$. Is $\Phi$ (lower limit) = constant on $r_0(\mu_v, \sigma_v) = C$?")

                ax_phi_upper.plot(result.mus, phi[1, :], color="#d62728")
                ax_phi_upper.set_title(
                    r"$\Phi(\frac{\mu_v - V_R}{\sqrt{2}\sigma_v})$. Is $\Phi$ (upper limit) = constant on $r_0(\mu_v, \sigma_v) = C$?")

                ax_phi.plot(result.mus, phi[0, :], label=r"$\Phi$"" lower limit")
                ax_phi.plot(result.mus, phi[1, :], label=r"$\Phi$"" upper limit")
                ax_phi.set_title(
                    r"$\Phi(\frac{\mu_v - \theta}{\sqrt{2}\sigma_v})$ and $\Phi(\frac{\mu_v - V_R}{\sqrt{2}\sigma_v})$. Why do they look linear w.r.t. each other?")

                phi_ratio = phi[1, :] / phi[0, :]
                ax_ratio.plot(result.mus, phi_ratio, label=f"ratio, {result.exp_label}")
                # ax_ratio.plot(result.mus, (result.mus - siegert_gradients.theta / mV) / (result.mus - siegert_gradients.v_reset / mV), label=r"$\frac{\mu_v - \theta}{\mu_v - V_R}$"f", {result.exp_label}")
                ax_ratio.set_title(
                    r"Ratio $\Phi(\frac{\mu_v - V_R}{\sqrt{2}\sigma_v})$/$\Phi(\frac{\mu_v - \theta}{\sqrt{2}\sigma_v})$. Is $\frac{\Phi(\mathrm{upper})}{\Phi(\mathrm{lower})} \approx 0$ ? max($\frac{\Phi(\mathrm{upper})}{\Phi(\mathrm{lower})}) =$"f"{np.max(phi_ratio): .4f}")

                ax_f.plot(result.mus, d_sigma_over_d_mus, label=f"{result.exp_label} direct")
                ax_f.plot(result.mus, grad_ratios, label=f"{result.exp_label} from sg")
                ax_f.set_title(r"$\frac{d \sigma_v}{d \mu_v}$ directly computed or from SG")

                ax_grad_ratio.plot(result.mus, grad_ratios, label="d sigma / d mu from sg")

                for ax_indexes in [0, 3, 5]:
                    axs[ax_indexes].legend()

                # fig.suptitle(r"Try wo understand why $I(\mu_v, \sigma_v) = \int_{\frac{\mu_v - \theta}{\sqrt{2}\sigma_v}}^{\frac{\mu_v - V_R}{\sqrt{2}\sigma_v}} dx \cdot e^{x^2} \cdot \mathrm{erfc}(x)$ = C for "f"{result.exp_label}, {result.r_target / Hz} Hz")
                fig.tight_layout()
                show_plots_non_blocking(caller_test_case=self, descriptor=result.exp_label)

    def test_compare_phi_lower_limit_vs_phi_upper_limit_use_different_ordering(self):
        from cycler import cycler

        with plt.rc_context({
            'axes.prop_cycle': cycler(color=['#2ca02c', '#d62728', '#9467bd', '#8c564b']),
            'lines.linewidth': 2.5
        }):
            siegert_gradients = SiegertGradients.for_lif_config(default_diffusion_lif_config)

            conditions = [
                default_diffusion_lif_config.with_label("MK-801"),
                default_diffusion_lif_config.with_label("Control"),
                default_diffusion_lif_config.with_label("High rate")
            ]

            target_rates = [0.05 * Hz, 0.18 * Hz, 25 * Hz]
            integral_limits_limits = [(-3, 100), (-3, 100), (-3, 25)]
            colors = ("orange", "black", "blue")

            results = [compute_mu_to_sigma_curve(c, r) for c, r in zip(conditions, target_rates)]

            # -------------------------
            # PRECOMPUTE EVERYTHING
            # -------------------------
            all_data = []

            for result, integral_limits_limit, color in zip(results, integral_limits_limits, colors):
                int_limits = np.zeros((2, len(result.mus)))
                phi = np.zeros((2, len(result.mus)))
                d_sigma_over_d_mus = np.zeros_like(result.mus)
                grad_ratios = np.zeros_like(result.mus)

                for i, (mu, sigma) in enumerate(zip(result.mus, result.sigmas)):
                    lower, upper = siegert_gradients.integration_limits(mu_v=mu, sigma_v=sigma)

                    int_limits[:, i] = [lower, upper]
                    phi[0, i] = siegert_gradients.phi(lower)
                    phi[1, i] = siegert_gradients.phi(upper)

                    d_sigma_over_d_mus[i] = d_sigma_over_d_mu(mu, sigma, siegert_gradients)
                    grad_ratios[i] = - siegert_gradients.d_rate_d_mu(mu, sigma) / siegert_gradients.d_rate_d_sigma(mu,
                                                                                                                   sigma)

                all_data.append({
                    "result": result,
                    "int_limits": int_limits,
                    "phi": phi,
                    "d_sigma": d_sigma_over_d_mus,
                    "grad_ratios": grad_ratios,
                    "ylim": integral_limits_limit,
                    "color": color
                })

            prepare_bigger_fonts()

            # Helper to switch layout easily
            def make_figure(n_axes, title, detail_name, horizontal=True, sharex=False, sharey=False):
                if horizontal:
                    fig, axs = plt.subplots(1, n_axes, figsize=(5 * n_axes, 4), sharex=sharex, sharey=sharey)
                else:
                    fig, axs = plt.subplots(n_axes, 1, figsize=(10, 3 * n_axes), sharex=sharex, sharey=sharey)
                fig.suptitle(title)
                return fig, axs
            for x_lim, y_lim, label in [(None, None, "whole_interval"), ((-55, -45), (-2, 10), "V_R_less_mu_less_theta")]:
                for show_lower, show_upper in [(True, True), (True, False), (False, True)]:
                    # -------------------------
                    # 1) INTEGRATION LIMITS
                    # -------------------------
                    fig, axs = make_figure(len(all_data), f"Integration limits {label}", "integration_limits")

                    for ax, data, color in zip(axs, all_data, colors):
                        r = data["result"]
                        if show_lower: ax.plot(r.mus, data["int_limits"][0, :], label="lower", color=color)
                        if show_upper: ax.plot(r.mus, data["int_limits"][1, :], label="upper", color="red")
                        #ax.set_ylim(data["ylim"])
                        ax.set_title(f"{r.exp_label}")
                        ax.legend()

                    if x_lim is not None:
                        ax.set_xlim(x_lim)

                    if y_lim is not None:
                        ax.set_ylim(y_lim)

                    fig.tight_layout()
                    show_plots_non_blocking(self, descriptor=f"integration_limits_{label}")

            # -------------------------
            # 2) PHI LOWER
            # -------------------------
            for sharey in [True, False]:
                fig, axs = make_figure(len(all_data), r"$\Phi(\frac{\mu_v - \theta}{\sqrt{2}\sigma_v})$ i.e. $\Phi$ lower int limit", "phi_lower", sharey=sharey)

                for ax, data, color in zip(axs, all_data, colors):
                    r = data["result"]
                    ax.plot(r.mus, data["phi"][0, :], color=color)
                    ax.set_title(r.exp_label)

                fig.tight_layout()
                show_plots_non_blocking(self, descriptor="phi_lower")

                # -------------------------
                # 3) PHI UPPER
                # -------------------------
                fig, axs = make_figure(len(all_data), r"$\Phi(\frac{\mu_v - V_R}{\sqrt{2}\sigma_v})$ i.e. $\Phi$ upper int limit", "phi_upper", sharey=sharey)

                for ax in axs[1:]:
                    ax.sharey(axs[0])
                for ax, data, color in zip(axs, all_data, colors):
                    r = data["result"]
                    ax.plot(r.mus, data["phi"][1, :], color=color)
                    ax.set_title(r.exp_label)

                fig.tight_layout()
                show_plots_non_blocking(self, descriptor="phi_upper")

            # -------------------------
            # 4) PHI BOTH
            # -------------------------
            fig, axs = make_figure(len(all_data), r"$\Phi$ comparison", "phi_both")

            for ax, data, color in zip(axs, all_data, colors):
                r = data["result"]
                ax.plot(r.mus, data["phi"][0, :], label="lower", color=color)
                ax.plot(r.mus, data["phi"][1, :], label="upper")
                ax.set_title(r.exp_label)
                ax.legend()

            fig.tight_layout()
            show_plots_non_blocking(self, descriptor="phi_both")

            # -------------------------
            # 5) PHI RATIO
            # -------------------------
            fig, axs = make_figure(len(all_data), r"$\Phi(\frac{\mu_v - V_R}{\sqrt{2}\sigma_v})/$"
                                                  r"$\Phi(\frac{\mu_v - \theta}{\sqrt{2}\sigma_v})$ (upper/lower)", "phi_ratio")

            for ax, data, color in zip(axs, all_data, colors):
                r = data["result"]
                ratio = data["phi"][1, :] / data["phi"][0, :]
                ax.plot(r.mus, ratio, color=color)
                ax.set_title(f"{r.exp_label}, max={np.max(ratio):.5f}")

            fig.tight_layout()
            show_plots_non_blocking(self, descriptor="phi_ratio")

            # -------------------------
            # 6) DERIVATIVES
            # -------------------------
            fig, axs = make_figure(len(all_data), "d sigma / d mu", "derivatives")

            for ax, data in zip(axs, all_data):
                r = data["result"]
                ax.plot(r.mus, data["d_sigma"], label="direct")
                ax.plot(r.mus, data["grad_ratios"], label="SG")
                ax.set_title(r.exp_label)
                ax.legend()

            fig.tight_layout()
            show_plots_non_blocking(self, descriptor="derivatives")

    '''
    Here we see the true reason why the equation is a line:
    We look at Phi(x) = e^x^2 erfc(x).
    We have 2 regimes: x > 0 and x < 0. 
    For x >> 0 (lests say, even 1) e^x^2 erfc (x) is basically zero. This is the value of the lower limit. 
    (mu - V_R) / sqrt 2 sigma. basically zero => does not contribute to integral. Actually, 
    '''

    def test_plot_integral_phi_x_ar_various_mu_v_sigma_v_values(self):
        siegert_gradients = SiegertGradients.for_lif_config(default_diffusion_lif_config)

        conditions = [default_diffusion_lif_config.with_label("MK-801"),
                      default_diffusion_lif_config.with_label("Control"),
                      default_diffusion_lif_config.with_label("Example high rate")]
        target_rates = [0.05 * Hz, 0.18 * Hz, 25 * Hz]

        results = [compute_mu_to_sigma_curve(condition, rate) for condition, rate in zip(conditions, target_rates)]

        phi = [None] * len(results)
        int_limits = [None] * len(results)
        for result_number, result in enumerate(results):

            current_integral_limits = np.zeros(shape=(2, len(result.mus)))
            current_phi = np.zeros(shape=(2, len(result.mus)))
            for row_index, (mu, sigma) in enumerate(zip(result.mus, result.sigmas)):
                current_integral_limits[:, row_index] = siegert_gradients.integration_limits(mu_v=mu, sigma_v=sigma)
                lower_x_all_graphs, upper_x_all_graphs = current_integral_limits[:, row_index]
                current_phi[0, row_index] = siegert_gradients.phi(lower_x_all_graphs)
                current_phi[1, row_index] = siegert_gradients.phi(upper_x_all_graphs)

            int_limits[result_number] = current_integral_limits
            phi[result_number] = current_phi

        int_limits_as_np = np.array(int_limits)
        plot_indexes = [100, 200, 400]
        lower_x_all_graphs = 1.1 * np.min(int_limits_as_np[:, 0, plot_indexes[0]])
        upper_x_all_graphs = 1.1 * np.max(int_limits_as_np[:, 1, plot_indexes[-1]])
        x = np.linspace(lower_x_all_graphs, upper_x_all_graphs, 1000)

        colors = ['green', 'orange', 'red', 'blue']
        for result_number, result in enumerate(results):
            prepare_bigger_fonts()
            fig, axs = plt.subplots(4, 1, figsize=(10, 14))

            ax_integral_limits = axs[0]
            ax_integrals = axs[1:]
            for ax_integral in axs[2:]:
                ax_integral.sharex(ax_integrals[0])

            lower_limits = [int_limits[result_number][0, plotted_index] for plotted_index in plot_indexes]
            upper_limits = [int_limits[result_number][1, plotted_index] for plotted_index in plot_indexes]

            ax_integral_limits.plot(result.mus, int_limits[result_number][0, :],
                                    label=f"lower limit, {result.exp_label}")
            ax_integral_limits.plot(result.mus, int_limits[result_number][1, :],
                                    label=f"upper limit, {result.exp_label}")

            for index_of_integral, (ax, limit) in enumerate(zip(ax_integrals, plot_indexes)):
                current_index = plot_indexes[index_of_integral]
                phi = erfcx(x)
                # phi = np.clip(phi, 0, 100)
                ax.plot(x, phi)
                a = lower_limits[index_of_integral]
                b = upper_limits[index_of_integral]
                current_mu = result.mus[current_index]
                current_sigma = result.sigmas[current_index]

                ax_integral_limits.axvline(current_mu, color=colors[index_of_integral], linestyle='-.',
                                           label=f"Panel {chr(ord("B") + index_of_integral)}")

                current_integral = siegert_gradients.I_mu_sigma(mu_v=current_mu * mV, sigma_v=current_sigma * mV)
                ax.set_title(
                    f"{r"$\mu_v$="}{current_mu :.3f} mV,{r"$\sigma_v=$"}{current_sigma :.3f} mV " r"$I(\mu_v, \sigma_v)=$"f"{current_integral: .4f}""\n "
                    r"lower limit $\frac{\mu_v - \theta}{\sqrt{2}\sigma_v}$="f"{lower_limits[index_of_integral]:.4f}, "r"upper limit $\frac{\mu_v - V_R}{\sqrt{2}\sigma_v}$="f"{upper_limits[index_of_integral]:.4f}")
                ax.axvline(a, color="red", linestyle='--', label="lower limit")
                ax.axvline(b, color="blue", linestyle='--', label="upper limit")

                ymax = 100
                phi_clipped = np.minimum(phi, ymax)

                # mask for region between a and b
                mask = (x >= a) & (x <= b)

                # shaded area from y=0 to phi(x)
                ax.fill_between(
                    x[mask],
                    0,
                    phi_clipped[mask],
                    color='gray',
                    alpha=0.3,
                    hatch="//"
                )
                ax.set_ylim((-1, ymax))
            for ax in axs:
                ax.legend()

            ax_integral_limits.set_ylim((-4, 20))
            ax_integral_limits.set_title("Integral limits")

            add_panel_info(axs)

            fig.suptitle(
                r"Plot $\Phi(x) = e^{x^2} \cdot \mathrm{erfc}(x)$ together with limits (A) and area bellow the curve (B-D)""\n"
                f"{result.exp_label} {result.r_target / Hz: .3f} Hz")
            fig.tight_layout()
            show_plots_non_blocking(caller_test_case=self, descriptor=result.exp_label)

    def test_plot_phi_of_x(self):

        x = np.linspace(-3, 3, 1001)
        prepare_bigger_fonts()
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        axs[0].plot(x, np.exp(x ** 2))
        axs[1].plot(x, erfc(x))
        axs[2].plot(x, erfcx(x))

        axs[0].set_ylim((0, 1000))
        axs[2].set_ylim((0, 1000))
        axs[0].set_title(r"$e^{x^2}$")
        axs[1].set_title("erfc(x)")
        axs[2].set_title(r"$\Phi(x) = e^{x^2}\cdot \mathrm{erfc}(x)$")

        axs[2].set_xlabel(r"$x$")
        for ax in axs:
            ax.tick_params(axis='both', which='both',
                           bottom=True, top=True,  # x-axis ticks
                           labelbottom=True)  # show x tick labels on all

        prepare_bigger_fonts()
        fig.tight_layout()
        show_plots_non_blocking(caller_test_case=self)

    def test_check_if_integrating_to_infinity_produces_similar_or_same_results(self):
        siegert_gradient = SiegertGradients.for_lif_config(default_diffusion_lif_config)

        diffusion_config_low_reset = DiffusionLIFConfig(params={DiffusionLIFConfig.KEY_V_R: -1E4})
        siegert_gradients_low_reset = SiegertGradients.for_lif_config(diffusion_config_low_reset)

        siegert_gradients = [siegert_gradient, siegert_gradients_low_reset]

        normal_formula_conditions = [default_diffusion_lif_config.with_label("MK-801, usual formula"),
                                     default_diffusion_lif_config.with_label("Control, usual formula")]
        low_vr_reset_conditions = [diffusion_config_low_reset.with_label("MK-801, low reset formula"),
                                   diffusion_config_low_reset.with_label("Control, low reset formula")]
        target_rates = [0.05 * Hz, 0.18 * Hz]

        results_low_reset = [compute_mu_to_sigma_curve(condition, rate) for condition, rate in
                             zip(low_vr_reset_conditions, target_rates)]
        results_normal_formula = [compute_mu_to_sigma_curve(condition, rate) for condition, rate in
                                  zip(normal_formula_conditions, target_rates)]

        prepare_bigger_fonts()
        fig, axs = plt.subplots(3, 2, figsize=(11, 14), dpi=200, sharex=True)

        results_mk801 = [results_normal_formula[0], results_low_reset[0]]
        results_control = [results_normal_formula[1], results_low_reset[1]]

        mk_801_rates = [[None] * 2 for _ in range(2)]
        control_rates = [[None] * 2 for _ in range(2)]

        # result will be
        '''
            row computed, column rate recomputed by gradient
            mk 801 computed with usual formula, rate recomputed by usual formula, mk 801 computed with usual formula, rate recomputed by V_R infty
            mk 801 computed with V_infty formula, rate recomputed by usual formula, mk 801 computed with V_infty formula, rate recomputed by V_R infty
        '''
        for index_row, mk_801_result in enumerate(results_mk801):
            for index_column, current_siegert_gradient in enumerate(siegert_gradients):
                mk_801_rates[index_row][index_column] =  np.array([current_siegert_gradient.firing_rate(mu * mV, sigma * mV) for mu, sigma in
                                           mk_801_result.mus_to_sigmas()])

        for index_row, mk_801_result in enumerate(results_control):
            for index_column, current_siegert_gradient in enumerate(siegert_gradients):
                control_rates[index_row][index_column] =  np.array([current_siegert_gradient.firing_rate(mu * mV, sigma * mV) for mu, sigma in
                                           mk_801_result.mus_to_sigmas()])

        ignore_indexes = 130

        """Plot μ vs σ curves and linear fit. Used by script runners with caller_test_case=self for figure naming."""
        # results are: first computed with usual formula, then computed with v_infty formula
        for index, result in enumerate(results_mk801):
            axs[0, 0].plot(result.mus, result.sigmas, label=f"{result.exp_label}, r = {result.r_target / Hz} Hz", lw=2)

            axs[1, 0].plot(result.mus[:-ignore_indexes], mk_801_rates[index][0][:-ignore_indexes],
                           label= f"{result.exp_label} [in]")
            axs[2, 0].plot(result.mus[:-ignore_indexes], mk_801_rates[index][1][:-ignore_indexes],
                           label=f"{result.exp_label} [in]")

        for index, result in enumerate(results_control):
            axs[0, 1].plot(result.mus, result.sigmas, label=f"{result.exp_label}, r = {result.r_target / Hz} Hz", lw=2)

            #firing_rates_normal_formula = [siegert_gradient.firing_rate(mu * mV, sigma * mV) for mu, sigma in
            #                               result.mus_to_sigmas()]
            #firing_rates_low_reset = [siegert_gradients_low_reset.firing_rate(mu * mV, sigma * mV) for mu, sigma in
            #                          result.mus_to_sigmas()]

            axs[1, 1].plot(result.mus[:-ignore_indexes], control_rates[index][0][:-ignore_indexes],
                           label=f"{result.exp_label} [in]")
            axs[2, 1].plot(result.mus[:-ignore_indexes], control_rates[index][1][:-ignore_indexes],
                           label=f"{result.exp_label} [in]")

        for ax, exp in zip([axs[0, 0], axs[0, 1]], ["MK-801", "Control"]):
            ax.set_xlabel(r"$\mu_v$ [mV]")
            ax.set_ylabel(r"$\sigma_v$ [mV]")
            ax.set_title(f"{exp} \n"
                r"$\mu$ vs $\sigma_v$ dependency for constant firing rate")

            ax.legend()

        for ax in [axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]]:
            ax.set_xlabel(r"$\mu_v$ [mV]")
            ax.set_ylabel(r"firing rate [Hz]")
            ax.legend()

        def get_middle_plotting_pos(ax_1, ax_2):
            pos1 = ax_1.get_position()
            pos2 = ax_2.get_position()

            x_center = (pos1.x0 + pos2.x1) / 2
            y_top = max(pos1.y1, pos2.y1)
            return x_center, y_top


        delta_r_over_r = np.max(np.abs(mk_801_rates[0][0][:-ignore_indexes] - mk_801_rates[0][1][:-ignore_indexes])) / ( results_mk801[0].r_target / Hz)
        axs[1, 0].set_title(r"$\Delta r / r_{\mathrm{target}}$="f"{delta_r_over_r :.3f}")
        delta_r_over_r = np.max(np.abs(mk_801_rates[1][0][:-ignore_indexes] - mk_801_rates[1][1][:-ignore_indexes])) / (results_mk801[1].r_target / Hz)
        axs[2, 0].set_title(r"$\Delta r / r_{\mathrm{target}}$="f"{delta_r_over_r :.3f}")

        delta_r_over_r = np.max(np.abs(control_rates[0][0][:-ignore_indexes] - control_rates[0][1][:-ignore_indexes])) / (results_control[0].r_target / Hz)
        axs[1, 1].set_title(r"$\Delta r / r_{\mathrm{target}}$="f"{delta_r_over_r :.3f}")
        delta_r_over_r = np.max(np.abs(control_rates[1][0][:-ignore_indexes] - control_rates[1][1][:-ignore_indexes])) / (results_control[1].r_target / Hz)
        axs[2, 1].set_title(r"$\Delta r / r_{\mathrm{target}}$="f"{delta_r_over_r :.3f}")

        x, y = get_middle_plotting_pos(axs[1, 0], axs[1, 1])

        fig.text(x, y+0.01, r"$r_0(\mu_v, \sigma_v) = \left( \tau_{\mathrm{ref}} + \tau_m \cdot \sqrt{\pi} \cdot \int_{\frac{\mu_v - \theta}{\sqrt{2}\sigma_v}}^{\frac{\mu_v - V_R}{\sqrt{2}\sigma_v}} dx \cdot e^{x^2} \mathrm{erfc}(x) \right)^{-1}$ [out]", ha='center')

        x, y = get_middle_plotting_pos(axs[2, 0], axs[2, 1])

        fig.text(x, y, r"$r_{-\infty}(\mu_v, \sigma_v) = \left( \tau_{\mathrm{ref}} + \tau_m \cdot \sqrt{\pi} \cdot \int_{\frac{\mu_v - \theta}{\sqrt{2}\sigma_v}}^{\infty} dx \cdot e^{x^2} \mathrm{erfc}(x) \right)^{-1}$ [out]", ha="center")

        fig.suptitle("Is the low reset formula producing similar results?")
        fig.subplots_adjust(hspace=0.6, wspace=0.3)

        show_plots_non_blocking(caller_test_case=self)


    def test_fit_and_plot_r_infty(self):
        siegert_gradients = SiegertGradients.for_lif_config(default_diffusion_lif_config)
        diffusion_config_low_reset = DiffusionLIFConfig(params={DiffusionLIFConfig.KEY_V_R: -1E4})
        siegert_gradients_low_reset = SiegertGradients.for_lif_config(diffusion_config_low_reset)

        low_vr_reset_conditions = [diffusion_config_low_reset.with_label("MK-801, low reset formula"),
                                   diffusion_config_low_reset.with_label("Control, low reset formula")]
        #target_rates = [0.05 * Hz, 0.18 * Hz]
        target_rates = [0.05 * Hz]

        results = [compute_mu_to_sigma_curve(condition, rate) for condition, rate in
                             zip(low_vr_reset_conditions, target_rates)]

        prepare_bigger_fonts()
        fig, (ax_mu_to_sigma, ax_mu_to_firing_rate_standard_formula, ax_mu_to_firing_rate_infty_formula) = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

        ignore_indexes = 130

        current_result = results[0]

        ax_mu_to_sigma.plot(current_result.mus, results[0].sigmas, label=f"{current_result.exp_label}, r = {current_result.r_target / Hz} Hz", lw=2)

        firing_rates_normal_formula = [siegert_gradients.firing_rate(mu * mV, sigma * mV) for mu, sigma in
                                       current_result.mus_to_sigmas()]
        firing_rates_low_reset = [siegert_gradients_low_reset.firing_rate(mu * mV, sigma * mV) for mu, sigma in
                                  current_result.mus_to_sigmas()]

        ax_mu_to_firing_rate_standard_formula.plot(current_result.mus[:-ignore_indexes], firing_rates_normal_formula[:-ignore_indexes],
                       label=f"{current_result.exp_label}")
        ax_mu_to_firing_rate_infty_formula.plot(current_result.mus[:-ignore_indexes], firing_rates_low_reset[:-ignore_indexes],
                           label=f"{current_result.exp_label}")

        ax_mu_to_sigma.set_xlabel(r"$\mu_v$ [mV]")
        ax_mu_to_sigma.set_ylabel(r"$\sigma_v$ [mV]")
        ax_mu_to_sigma.set_title(f"{current_result.exp_label} \n"
                     r"$\mu$ vs $\sigma_v$ dependency for constant firing rate")

        ax_mu_to_sigma.legend()

        for ax in [ax_mu_to_firing_rate_standard_formula, ax_mu_to_firing_rate_infty_formula]:
            ax.set_xlabel(r"$\mu_v$ [mV]")
            ax.set_ylabel(r"firing rate [Hz]")
            ax.legend()

        ax_mu_to_firing_rate_standard_formula.set_title(r"$r_0(\mu_v, \sigma_v) = \left( \tau_{\mathrm{ref}} + \tau_m \cdot \sqrt{\pi} \cdot \int_{\frac{\mu_v - \theta}{\sqrt{2}\sigma_v}}^{\frac{\mu_v - V_R}{\sqrt{2}\sigma_v}} dx \cdot e^{x^2} \mathrm{erfc}(x) \right)^{-1}$")
        ax_mu_to_firing_rate_infty_formula.set_title(r"$r_{-\infty}(\mu_v, \sigma_v) = \left( \tau_{\mathrm{ref}} + \tau_m \cdot \sqrt{\pi} \cdot \int_{\frac{\mu_v - \theta}{\sqrt{2}\sigma_v}}^{\infty} dx \cdot e^{x^2} \mathrm{erfc}(x) \right)^{-1}$")

        fig.suptitle("Is the low reset formula producing similar results?")
        fig.tight_layout()

        show_plots_non_blocking(caller_test_case=self)

    # issue. I look for 0.05 Hz and we get back 0.05018 Hz
    def test_why_is_found_firing_rate_not_equal_to_target(self):
        mu = -58 * mV

        diffusion_config_low_reset = DiffusionLIFConfig(params={DiffusionLIFConfig.KEY_V_R: -1E4})
        siegert_gradients_low_reset = SiegertGradients.for_lif_config(diffusion_config_low_reset)
        sigma = binary_search_sigma_at_mu_for_firing_rate(mu=mu, r_target=0.05 *  Hz, lif_config=diffusion_config_low_reset)

        print(siegert_gradients_low_reset.firing_rate(mu_v = mu, sigma_v = sigma))

    def test_quickcheck_formula(self):
        mk_801_rate = 0.05 * Hz
        control_rate = 0.18 * Hz
        formula = default_diffusion_lif_config.tau_m * mk_801_rate ** 2 / 2 * (
                    default_diffusion_lif_config.theta - default_diffusion_lif_config.V_r) / (np.exp(
            (1 - mk_801_rate * default_diffusion_lif_config.tau_rp) / (
                        mk_801_rate * default_diffusion_lif_config.tau_m)) - 1) ** 2
        print(formula)

    def test_slope_d_mu_d_sigma_approximation(self):
        lif_config = default_diffusion_lif_config
        siegert_gradients = SiegertGradients.for_lif_config(lif_config)
        lif_configs = [lif_config.with_label("MK-801"),
                       lif_config.with_label("Control")]
        #target_rates = [0.05 * Hz, 0.18 * Hz]
        target_rates = [0.05 * Hz]
        results = [compute_mu_to_sigma_curve(config, r_target=target_rate) for config, target_rate in
                   zip(lif_configs, target_rates)]
        nmda_block_mu_to_sigma = results[0]

        m_mk801, b_mk801, _, _, _ = nmda_block_mu_to_sigma.linear_fit()

        mu_v = nmda_block_mu_to_sigma.mus[:900]
        sigma_v = np.linspace(min(nmda_block_mu_to_sigma.sigmas), max(nmda_block_mu_to_sigma.sigmas), len(nmda_block_mu_to_sigma.sigmas))[:900]

        for mu_sigma_index in [10, 100, 500]:
            mu_v_ex, sigma_v_ex = nmda_block_mu_to_sigma.mus[mu_sigma_index], nmda_block_mu_to_sigma.sigmas[mu_sigma_index]
            lower_limit, upper_limit = siegert_gradients.integration_limits(mu_v = mu_v_ex, sigma_v = sigma_v_ex)

            r = siegert_gradients.phi(z = upper_limit) / siegert_gradients.phi(z = lower_limit)

            d_mu_d_sigma_zero_approx = sigma_v /  (mu_v - lif_config.theta / mV)
            d_mu_d_sigma_first_approx = sigma_v /  (mu_v - lif_config.theta / mV - (lif_config.theta - lif_config.V_r) / mV * r)
            d_mu_d_sigma_second_approx = sigma_v /  (mu_v - lif_config.theta / mV - (lif_config.theta - lif_config.V_r) / mV * (r + r**2))
            #plt.plot(mu_v, d_mu_d_sigma_zero_approx, label="zeroth approx")
            plt.plot(mu_v, d_mu_d_sigma_first_approx, label=f"first approx for {mu_sigma_index}")
            plt.plot(mu_v, d_mu_d_sigma_second_approx, label=f"second approx for {mu_sigma_index}")


        plt.axhline(y=m_mk801, linestyle="--", label="Slope", lw=2)

        plt.legend()
        plt.show()




if __name__ == '__main__':
    unittest.main()
