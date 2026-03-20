"""
Runnable Siegert μ–σ scripts: plots and scans.

Convention: test_scripts_* = runnable experiments/plots (discovered by IntelliJ as tests).
"""
import sys

from loguru import logger
from matplotlib.rcsetup import cycler

from Plotting import prepare_bigger_fonts, show_plots_non_blocking, add_panel_info
from iteration_12_transfer_function_of_lif_neurons.config import default_diffusion_lif_config

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc
from brian2 import mV, Hz, mvolt

from iteration_12_transfer_function_of_lif_neurons.LookForAllSolutionsMuSigma import (
    compute_mu_to_sigma_curve_for_experiment,
    compute_mu_to_sigma_fsolve_scan_mus,
    compute_mu_to_sigma_fsolve_scan_sigmas,
    plot_mus_vs_sigmas,
    plot_loss_landscape_with_curve, compute_mu_to_sigma_curve,
)
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import (
    SiegertGradientDescent,
    SiegertGradients, erfcx, )
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import NeuronModelParams
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control


def d_sigma_over_d_mu(mu_v, sigma_v, siegert_gradients: SiegertGradients):
    # Arguments of the CDF
    a, b = siegert_gradients.integration_limits(mu_v, sigma_v)

    Phi_a = siegert_gradients.phi(a)
    Phi_b = siegert_gradients.phi(b)

    # Numerator
    numerator = sigma_v * (Phi_a - Phi_b)

    # Denominator
    denominator = (mu_v - siegert_gradients.v_reset / mV) * Phi_a - (mu_v - siegert_gradients.theta / mV) * Phi_b

    return numerator / denominator


class LookForAllSolutionsScripts(unittest.TestCase):
    """Runnable scripts: μ–σ curves, loss landscape, fsolve scans. test_scripts_* = run directly in IntelliJ."""

    def test_scripts_look_for_all_solutions(self):
        experiment = palmer_control
        r_target = 0.3 * Hz
        solver = SiegertGradientDescent(
            tau_m=experiment.effective_time_constant_up_state.tau_eff(),
            theta=experiment.neuron_params.theta,
            v_reset=experiment.neuron_params.V_r,
            tau_ref=experiment.neuron_params.tau_rp,
            unit="mV",
        )
        fig, _ = plot_loss_landscape_with_curve(
            solver, r_target, (-60, -35), (0, 6),
            resolution=50, target_rate_Hz=0.3, caller_test_case=self,
        )
        fig.show()

    def test_scripts_look_for_all_solutions_using_binary_search(self):
        experiment = palmer_control
        #palmer_control.with_property(NeuronModelParams.KEY_NEURON_V_R, -55)
        nmda_block_mu_to_sigma = compute_mu_to_sigma_curve_for_experiment(
            experiment.with_label("NMDA Block"), r_target=0.05 * Hz
        )
        control_mu_to_sigma = compute_mu_to_sigma_curve_for_experiment(
            experiment.with_label("Control"), r_target=0.3 * Hz
        )
        mu_60_arg = np.abs(nmda_block_mu_to_sigma.mus - (-60)).argmin()
        mu_close_to_60 = nmda_block_mu_to_sigma.mus[mu_60_arg] * mvolt
        sigma_for_mu_60 = nmda_block_mu_to_sigma.sigmas[mu_60_arg] * mvolt
        mu_60_1_arg = np.abs(control_mu_to_sigma.mus + (60 - 0.1)).argmin()
        mu_close_to_60_1 = control_mu_to_sigma.mus[mu_60_1_arg] * mvolt
        sigma_for_mu_60_1 = control_mu_to_sigma.sigmas[mu_60_1_arg] * mvolt
        siegert_gradients = SiegertGradients.for_experiment(experiment)
        rate_60_mv = siegert_gradients.firing_rate(
            mu_v=mu_close_to_60, sigma_v=sigma_for_mu_60
        )
        rate_60_1 = siegert_gradients.firing_rate(
            mu_v=mu_close_to_60_1, sigma_v=sigma_for_mu_60_1
        )

        integral_limits_nmda_block = [siegert_gradients.integration_limits(mu_v = mu, sigma_v = sigma) for mu, sigma in zip(
            nmda_block_mu_to_sigma.mus, nmda_block_mu_to_sigma.sigmas
        )]

        print(rate_60_mv, rate_60_1)
        print(
            f"Delta in mu {(mu_close_to_60_1 - mu_close_to_60) / mV}. "
            f"Delta in sigma {(sigma_for_mu_60_1 - sigma_for_mu_60) / mV}."
        )
        plot_mus_vs_sigmas(
            [nmda_block_mu_to_sigma, control_mu_to_sigma],
            caller_test_case=self,
        )

    def test_scripts_look_for_all_solutions_using_binary_search_palmer_rates(self):
        experiment = palmer_control
        palmer_control.with_property(NeuronModelParams.KEY_NEURON_V_R, -55)
        nmda_block_mu_to_sigma = compute_mu_to_sigma_curve_for_experiment(
            experiment.with_label("NMDA Block"), r_target=0.05 * Hz
        )
        control_mu_to_sigma = compute_mu_to_sigma_curve_for_experiment(
            experiment.with_label("Control"), r_target=0.18 * Hz
        )
        mu_60_arg = np.abs(nmda_block_mu_to_sigma.mus - (-60)).argmin()
        mu_close_to_60 = nmda_block_mu_to_sigma.mus[mu_60_arg] * mvolt
        sigma_for_mu_60 = nmda_block_mu_to_sigma.sigmas[mu_60_arg] * mvolt
        rate_60_mv = SiegertGradients.for_experiment(experiment).firing_rate(
            mu_v=mu_close_to_60, sigma_v=sigma_for_mu_60
        )
        mu_59_9_arg = np.abs(control_mu_to_sigma.mus + (60 - 0.1)).argmin()
        mu_close_to_59_9 = control_mu_to_sigma.mus[mu_59_9_arg] * mvolt
        sigma_for_mu_59_9 = control_mu_to_sigma.sigmas[mu_59_9_arg] * mvolt
        rate_59_9_mv = SiegertGradients.for_experiment(experiment).firing_rate(
            mu_v=mu_close_to_59_9, sigma_v=sigma_for_mu_59_9
        )
        print(
            f"mu = {mu_close_to_60 / mV:.4f}, sigma = {sigma_for_mu_60 / mV:.4f} "
            f"produces rate {rate_60_mv / Hz:.4f}"
        )
        print(
            f"mu = {mu_close_to_59_9 / mV:.4f}, sigma = {sigma_for_mu_59_9 / mV:.4f} "
            f"produces rate {rate_59_9_mv / Hz:.4f}"
        )
        plot_mus_vs_sigmas(
            [nmda_block_mu_to_sigma, control_mu_to_sigma],
            caller_test_case=self,
        )

    def test_scripts_look_for_all_solutions_using_fsolve_scan_sigma(self):
        result = compute_mu_to_sigma_fsolve_scan_sigmas(
            palmer_control, r_target=0.3 * Hz
        )
        print(result)
        self.assertEqual(101, len(result.mus))
        self.assertEqual(101, len(result.sigmas))
        plot_mus_vs_sigmas([result], caller_test_case=self)

    def test_scripts_look_for_all_solutions_using_fsolve_scan_mu(self):
        result = compute_mu_to_sigma_fsolve_scan_mus(
            palmer_control, r_target=0.3 * Hz
        )
        print(result)
        self.assertEqual(1001, len(result.mus))
        self.assertEqual(1001, len(result.sigmas))
        plot_mus_vs_sigmas([result], caller_test_case=self)

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
        #plt.plot(np.array(integral_limits_nmda_block)[0], label=["lower limit"])
        #plt.ylim((-3, 5))
        plt.legend()
        plt.show()

    def test_sieger_value_understand_why_lower_limit_is_higher_than_upper_limit(self):
        mu = -60.04004004004004 * mV
        sigma = 2.5299242694745767 * mV

        siegert_gradient = SiegertGradients.for_experiment(palmer_control)
        lower_bound, upper_bound = siegert_gradient.integration_limits(mu_v = mu, sigma_v = sigma)
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

            #conditions = [default_diffusion_lif_config.with_label("MK-801"), default_diffusion_lif_config.with_label("Control")]
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
                    grad_ratios[row_index] = - siegert_gradients.d_rate_d_mu(mu, sigma) / siegert_gradients.d_rate_d_sigma(mu, sigma)

                prepare_bigger_fonts()
                fig, axs = plt.subplots(7, 1, figsize=(12, 12))

                ax_integral_limits, ax_phi_lower, ax_phi_upper, ax_phi, ax_ratio, ax_f, ax_grad_ratio = axs

                ax_integral_limits.plot(result.mus, int_limits[0, :], label=f"lower integration limit")
                ax_integral_limits.plot(result.mus, int_limits[1, :], label=f"upper integration limit")
                ax_integral_limits.set_ylim(integral_limits_limit)
                ax_integral_limits.set_title(r"Plot lower limit $\frac{\mu_v - \theta}{\sqrt{2}\sigma_v}$ and upper limit $\frac{\mu_v - V_R}{\sqrt{2}\sigma_v}$ = C\\Are upper and lower limits constant on $r_0(\mu_v, \sigma_v)$ = C?")

                ax_phi_lower.plot(result.mus, phi[0, :], color="#2ca02c")
                ax_phi_lower.set_title(r"$\Phi(\frac{\mu_v - \theta}{\sqrt{2}\sigma_v})$. Is $\Phi$ (lower limit) = constant on $r_0(\mu_v, \sigma_v) = C$?")

                ax_phi_upper.plot(result.mus, phi[1, :], color="#d62728")
                ax_phi_upper.set_title(
                    r"$\Phi(\frac{\mu_v - V_R}{\sqrt{2}\sigma_v})$. Is $\Phi$ (upper limit) = constant on $r_0(\mu_v, \sigma_v) = C$?")

                ax_phi.plot(result.mus, phi[0, :], label=r"$\Phi$"" lower limit")
                ax_phi.plot(result.mus, phi[1, :], label=r"$\Phi$"" upper limit")
                ax_phi.set_title(r"$\Phi(\frac{\mu_v - \theta}{\sqrt{2}\sigma_v})$ and $\Phi(\frac{\mu_v - V_R}{\sqrt{2}\sigma_v})$. Why do they look linear w.r.t. each other?")

                phi_ratio = phi[1, :] / phi[0, :]
                ax_ratio.plot(result.mus, phi_ratio, label=f"ratio, {result.exp_label}")
                #ax_ratio.plot(result.mus, (result.mus - siegert_gradients.theta / mV) / (result.mus - siegert_gradients.v_reset / mV), label=r"$\frac{\mu_v - \theta}{\mu_v - V_R}$"f", {result.exp_label}")
                ax_ratio.set_title(r"Ratio $\Phi(\frac{\mu_v - V_R}{\sqrt{2}\sigma_v})$/$\Phi(\frac{\mu_v - \theta}{\sqrt{2}\sigma_v})$. Is $\frac{\Phi(\mathrm{upper})}{\Phi(\mathrm{lower})} \approx 0$ ? max($\frac{\Phi(\mathrm{upper})}{\Phi(\mathrm{lower})}) =$"f"{np.max(phi_ratio): .4f}")

                ax_f.plot(result.mus, d_sigma_over_d_mus, label=f"{result.exp_label} direct")
                ax_f.plot(result.mus, grad_ratios, label=f"{result.exp_label} from sg")
                ax_f.set_title(r"$\frac{d \sigma_v}{d \mu_v}$ directly computed or from SG")

                ax_grad_ratio.plot(result.mus, grad_ratios, label="d sigma / d mu from sg")

                for ax_indexes in [0, 3, 5]:
                    axs[ax_indexes].legend()

                #fig.suptitle(r"Try wo understand why $I(\mu_v, \sigma_v) = \int_{\frac{\mu_v - \theta}{\sqrt{2}\sigma_v}}^{\frac{\mu_v - V_R}{\sqrt{2}\sigma_v}} dx \cdot e^{x^2} \cdot \mathrm{erfc}(x)$ = C for "f"{result.exp_label}, {result.r_target / Hz} Hz")
                fig.tight_layout()
                show_plots_non_blocking(caller_test_case=self, descriptor=result.exp_label)

    '''
    Here we see the true reason why the equation is a line:
    We look at Phi(x) = e^x^2 erfc(x).
    We have 2 regimes: x > 0 and x < 0. 
    For x >> 0 (lests say, even 1) e^x^2 erfc (x) is basically zero. This is the value of the lower limit. 
    (mu - V_R) / sqrt 2 sigma. basically zero => does not contribute to integral. Actually, 
    '''
    def test_plot_integral_phi_x_ar_various_mu_v_sigma_v_values(self):
        siegert_gradients = SiegertGradients.for_lif_config(default_diffusion_lif_config)

        conditions = [default_diffusion_lif_config.with_label("MK-801"), default_diffusion_lif_config.with_label("Control"), default_diffusion_lif_config.with_label("Example high rate")]
        target_rates = [0.05 * Hz, 0.18 * Hz, 25 * Hz]

        results = [compute_mu_to_sigma_curve(condition, rate) for condition, rate in zip(conditions, target_rates)]

        phi = [None] * len(results)
        int_limits = [None] *len(results)
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
        lower_x_all_graphs = 0.9 * np.min(int_limits_as_np[:, 0, plot_indexes[0]])
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

            ax_integral_limits.plot(result.mus, int_limits[result_number][0, :], label=f"lower limit, {result.exp_label}")
            ax_integral_limits.plot(result.mus, int_limits[result_number][1, :], label=f"upper limit, {result.exp_label}")

            for index_of_integral, (ax, limit) in enumerate(zip(ax_integrals, plot_indexes)):

                current_index = plot_indexes[index_of_integral]
                phi = erfcx(x)
                ax.plot(x, phi)
                a = lower_limits[index_of_integral]
                b = upper_limits[index_of_integral]
                current_mu = result.mus[current_index]
                current_sigma = result.sigmas[current_index]

                ax_integral_limits.axvline(current_mu, color=colors[index_of_integral], linestyle='-.',
                                           label=f"Panel {chr(ord("B") + index_of_integral)}")

                current_integral = siegert_gradients.I_mu_sigma(mu_v=current_mu * mV, sigma_v=current_sigma * mV)
                ax.set_title(f"{r"$\mu_v$="}{current_mu :.3f} mV,{r"$\sigma_v=$"}{current_sigma :.3f} mV " r"$I(\mu_v, \sigma_v)=$"f"{current_integral: .4f}""\n "
                             r"lower limit $\frac{\mu_v - \theta}{\sqrt{2}\sigma_v}$="f"{lower_limits[index_of_integral]:.4f}, "r"upper limit $\frac{\mu_v - V_R}{\sqrt{2}\sigma_v}$="f"{upper_limits[index_of_integral]:.4f}")
                ax.axvline(a, color="red", linestyle='--', label="lower limit")
                ax.axvline(b, color="blue", linestyle='--', label="upper limit")

                # mask for region between a and b
                mask = (x >= a) & (x <= b)

                # shaded area from y=0 to phi(x)
                ax.fill_between(
                    x[mask],
                    0,
                    phi[mask],
                    color='gray',
                    alpha=0.3,
                    hatch='//'
                )
                ax.set_ylim((-1, 100))
            for ax in axs:
                ax.legend()

            ax_integral_limits.set_ylim((-4, 20))
            ax_integral_limits.set_title("Integral limits")

            add_panel_info(axs)

            fig.suptitle(r"Plot $\Phi(x) = e^{x^2} \cdot \mathrm{erfc}(x)$ together with limits (A) and area bellow the curve (B-D)""\n"
                         f"{result.exp_label} {result.r_target / Hz: .3f} Hz")
            fig.tight_layout()
            show_plots_non_blocking(caller_test_case=self, descriptor=result.exp_label)

    def test_plot_phi_of_x(self):

        x = np.linspace(-3, 3, 1001)
        prepare_bigger_fonts()
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        axs[0].plot(x, np.exp(x**2))
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








