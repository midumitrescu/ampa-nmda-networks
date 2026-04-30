import sys

import numpy as np
from loguru import logger

from Plotting import show_plots_non_blocking, prepare_bigger_fonts
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients
from iteration_12_transfer_function_of_lif_neurons.config import default_diffusion_lif_config

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

import unittest

from brian2 import Hz, mV, is_dimensionless, have_same_dimensions
import matplotlib.pyplot as plt

from scipy.integrate import quad

def taylor_error(mu_sol, sigma_sol, delta_mu, siegert_gradients: SiegertGradients):

    if is_dimensionless(mu_sol):
        mu_sol = mu_sol * mV
    if is_dimensionless(sigma_sol):
        sigma_sol = sigma_sol * mV
    if is_dimensionless(delta_mu):
        delta_mu = delta_mu * mV

    assert have_same_dimensions(mu_sol, 1*mV)
    assert have_same_dimensions(sigma_sol, 1*mV)
    assert have_same_dimensions(delta_mu, 1*mV)


    def f_of_t(t, mu, sigma, D_mu):
        #print(f"f of t: t={t}, mu={mu}, sigma={sigma}, D_mu={D_mu}")
        squared = siegert_gradients.d_squared_rate_d_mu_squared(mu + t, sigma) / Hz * mV ** 2
        d_mu_minus_t = (D_mu - t) / mV
        return d_mu_minus_t * squared

    def error_integral(mu, sigma, D_mu):
        integrand = lambda t: f_of_t(t * mV, mu * mV, sigma * mV, D_mu * mV)
        return quad(integrand, 0, delta_mu / mV, epsabs=1e-14, epsrel=1e-12, limit=1000)[0] * Hz

    return error_integral(mu_sol / mV, sigma_sol / mV, delta_mu / mV)

class SolveByTaylorExpansion(unittest.TestCase):

    def test_check_taylor_error(self):
        # SolveByGraphicalSolutionScripts().test_solve_graphical_for_two_rates()
        mu_sol = -47.60912502853491
        sigma_sol = 1.9036551073068055
        rate_mk801 = 0.05 * Hz
        rate_control = 0.18 * Hz
        delta_mu = 0.7 * mV
        delta_rate = (rate_control - rate_mk801)

        d_rate_d_mu = delta_rate / delta_mu

        sg = SiegertGradients.default()

        rate_mk801 = sg.firing_rate(mu_sol * mV, sigma_sol * mV)

        rate_est = rate_mk801 + delta_mu * sg.d_rate_d_mu(mu_sol * mV, sigma_sol * mV)

        print(
            f"rate est {rate_est}. Delta {rate_control - rate_est}. Relative Error: {(rate_control - rate_est) / rate_control}")
        print(
            f"Cause of error: dr / d mu experiment {d_rate_d_mu / Hz * mV} vs dr / d mu experiment {sg.d_rate_d_mu(mu_sol * mV, sigma_sol * mV) / Hz * mV}")

        err_taylor = taylor_error(mu_sol, sigma_sol, delta_mu, siegert_gradients=sg)

        print(err_taylor)
        print(f"Error Taylor vs desired Errror: {err_taylor} vs {rate_control - rate_est}")

        lims = [(-55 * mV, default_diffusion_lif_config.theta), ((mu_sol - 0.1) * mV, (mu_sol + delta_mu / mV + 0.1) * mV)]

        second_grads = [None] * len(lims)

        integral_term = lambda mu: (mu_sol + delta_mu / mV - mu) * sg.d_squared_rate_d_mu_squared(mu * mV, sigma_sol * mV) / Hz * mV**2

        for index, (left, right) in enumerate(lims):
            mus = np.linspace(left, right, 2001)
            second_grad = np.array(
                [sg.d_squared_rate_d_mu_squared(mu, sigma_sol * mV) / Hz * mV ** 2 for mu in mus])
            second_grads[index] = {
                "mu": mus,
                "second_grad": second_grad,
            }

        mu_to_delta_mu = np.linspace(mu_sol * mV, mu_sol * mV + delta_mu, 1000)
        d2r_to_dmu2 = np.array(
            [sg.d_squared_rate_d_mu_squared(mu, sigma_sol * mV) / Hz * mV ** 2 for mu in mu_to_delta_mu])
        first_function = (mu_sol * mV + delta_mu - mu_to_delta_mu) / mV

        delta_mus = np.linspace(0, 1, num=1000)
        taylor_errors = np.array([taylor_error(mu_sol, sigma_sol, D_mu, siegert_gradients=sg) for D_mu in delta_mus])


        prepare_bigger_fonts()
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(
            2, 3,
            height_ratios=[3, 1],
            hspace=0.4,
            wspace=0.7
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        # Bottom row (spans all columns)
        ax_legend_1 = fig.add_subplot(gs[1, 0])
        ax_legend_2 = fig.add_subplot(gs[1, 1])
        ax_legend_3 = fig.add_subplot(gs[1, 2])

        for ax, grads in zip([ax1, ax2], second_grads):
            ax.axhline(y = 0, color = "k", lw=0.6)
            l1, = ax.plot(mu_to_delta_mu / mV, first_function, label=r"Linear kernel: $\mu_{\mathrm{sol}} + \Delta \mu - \mu$")
            l2, = ax.plot(grads['mu'] / mV, grads['second_grad'], label=r"$\frac{d^2 r}{d r^2}$", linestyle="--")
            l3, = ax.plot(mu_to_delta_mu / mV, d2r_to_dmu2, label=r"$\frac{d^2 r}{d r^2}$ within integration limits")
            l4, = ax.plot(mu_to_delta_mu / mV, d2r_to_dmu2 * first_function, label="Integrand: \n" r"$(\mu + \Delta \mu - t) \cdot \frac{\partial^2 r}{\partial \mu^2} (t, \sigma)$")

            l5 = ax.fill_between(
                mu_to_delta_mu / mV,
                d2r_to_dmu2 * first_function,
                0,
                edgecolor='gray',
                facecolor='gray',
                alpha=0.6,
                label="Integral value $E_1(\mu + \Delta \mu)$"
            )

            #l6 = ax.axvline(x=mu_sol, ymin=0, ymax=, ls='--', color='blue', alpha=0.6)
            y_sol = integral_term(mu_sol)
            ymin, ymax = ax.get_ylim()
            ax.axvline(
                x=mu_sol,
                ymin=(0 - ymin) / (ymax - ymin),
                ymax=(y_sol - ymin) / (ymax - ymin),
                ls=':', color='blue', alpha=0.6,
                label=r"$\mu_{\mathrm{sol}}$"
            )


        ax3.plot(delta_mus, taylor_errors, color="darkslategray")
        ax3.axvline(x=delta_mu / mV, ymax=0.5, color="darkslategray", lw=0.6, linestyle="--")
        ax3.axhline(y=err_taylor / Hz, xmax= 0.77, color="darkslategray", lw=0.6, linestyle="--")

        dot_size = 80
        l7 = ax3.scatter(delta_mu / mV, err_taylor / Hz, s=dot_size, alpha=0.6,
                         label="Taylor error at\n"r"$\mu_{\mathrm{sol}} = $" f"{mu_sol : .2f} mV, " r"$\sigma_{\mathrm{sol}} =$" f"{sigma_sol: .2f} mV" "\n\n"
                               r"$E_1(\mu_{\mathrm{sol}} +  \Delta \mu = 0.7 $"f"mV) ={err_taylor: .4f} Hz", color="darkslategray")

        ax1.set_title("\nIntegrand\n")
        ax2.set_title("Integrand \n within \n integration limits")
        ax3.set_title("\n"r"$E_1(\mu_{\mathrm{sol}} + \Delta \mu)$""\n")

        ax_legend_1.axis("off")
        ax_legend_2.axis("off")
        ax_legend_3.axis("off")
        ax_legend_1.legend(handles=[l2, l5], loc="center", frameon=False)
        ax_legend_2.legend(handles=[l1, l3, l4], loc="center", frameon=False)
        ax_legend_3.legend(handles=[l7], loc="center", frameon=False)
        ax1.set_ylabel(r"$\frac{d^2 r}{d \mu^2}$""\n[Hz/mV$^2$]", rotation=0, labelpad=35)
        [ax.set_xlabel(r"$\mu$ [mV]") for ax in [ax1, ax2]]

        ax3.set_ylabel("Error [Hz]")
        ax3.set_xlabel(r"$\Delta \mu$ [mV]")

        for ax, label in zip([ax1, ax2, ax3], ["A", "B", "C"]):
            ax.text(
                -0.25, 1.15, f"({label})",
                transform=ax.transAxes,
                fontsize=20,
                fontweight=10,
                va="top",
                ha="left"
            )

        fig.suptitle("First-order Taylor expansion error\n"r"$E_1(\mu + \Delta \mu) = \int_\mu^{\mu + \Delta \mu} dt \cdot (\mu + \Delta \mu - t) \cdot \frac{\partial^2 r}{\partial \mu^2} (t, \sigma) $")
        fig.subplots_adjust(top=0.75)
        fig.tight_layout()
        show_plots_non_blocking(caller_test_case=self)

        dmu = np.diff(mu_to_delta_mu)[0]
        print(f"Manual integral: {np.sum(d2r_to_dmu2 * first_function * dmu)}")
