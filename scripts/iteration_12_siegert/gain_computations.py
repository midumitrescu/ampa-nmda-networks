import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import mV, Hz, volt
from loguru import logger
from scipy.optimize import fsolve

from Plotting import show_plots_non_blocking, prepare_bigger_fonts
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from joblib import Parallel, delayed

def find_sigma_for_mu_producing_rate_gain(experiment: Experiment, mu, gain):
    siegert_gradient = SiegertGradients.for_experiment(experiment)
    sigma_sol = fsolve(func=lambda sigma: [siegert_gradient.d_rate_d_mu(mu_v=mu, sigma_v=sigma[0] * volt) - gain],
           x0=4 * mV)[0] * volt

    if sigma_sol < 0 * mV:
        return 0 * mV
    return sigma_sol


class ComputeAndPlotGain(unittest.TestCase):

    def test_plot_gain(self):
        gradients = SiegertGradients.for_experiment(palmer_control)
        mus = np.linspace(-65, -35, 1000) * mV

        prepare_bigger_fonts()
        plt.figure(figsize=(10, 8))
        plt.title(r'''control_mu_to_sigma.sigmas =$\frac{d\mathrm(rate)}{d\mu}$
        for Palmer figure 2 e,f, where $\frac{\Delta r}{\Delta\mu_v} = \frac{0.25 Hz}{0.1 mV} = 2.5 \frac{\mathrm{Hz}}{\mathrm{mV}}$''')

        sigma_numerical_solution = find_sigma_for_mu_producing_rate_gain(palmer_control, mu=palmer_control.neuron_params.theta - 10 * mV,
                                                                    gain=2.5 * Hz / mV)
        for sigma in np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]):
            gains = [gradients.d_rate_d_mu(mu_v=mu, sigma_v=sigma * mV) for mu in mus] * (mV / Hz)
            plt.plot(mus / mV, gains, label=r'$\sigma_v$=' + f"{sigma} mV" )

        gains = [gradients.d_rate_d_mu(mu_v=mu, sigma_v=sigma_numerical_solution) for mu in mus] * (mV / Hz)
        plt.plot(mus / mV, gains, label=r'Sol: $\sigma_v$=' + f"{sigma_numerical_solution/mV : .3f} mV")

        plt.axhline(y=2.5, linestyle='-.', label="Palmer gain")

        # Vertical line at x = -65 mV
        plt.axvline(x=palmer_control.neuron_params.theta / mV - 10, linestyle='--', label=r"10 mV bellow $\theta$")
        plt.axvline(x=palmer_control.neuron_params.theta / mV, linestyle='--', label=r"$\theta$", color='black')

        plt.xlabel("Membrane potential (mV)")
        plt.ylabel(r"Gain [$\frac{\mathrm{Hz}}{\mathrm{mV}}$]")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        show_plots_non_blocking()

    def test_find_sigma_by_f_solve(self):
        siegert_gradient = SiegertGradients.for_experiment(palmer_control)

        mu_fixed = palmer_control.neuron_params.theta - 10 * mV
        gain_target = 2.5 * Hz / mV

        sigma = find_sigma_for_mu_producing_rate_gain(palmer_control, mu=mu_fixed, gain=gain_target)

        self.assertAlmostEqual(4.676, sigma / mV, places=3)
        self.assertAlmostEqual(gain_target * mV / Hz, siegert_gradient.d_rate_d_mu(mu_v=mu_fixed, sigma_v=sigma) * mV / Hz, places=8)

    def test_compute_all_sigmas_required_for_our_gain(self):
        delta_mus = np.linspace(15, 1, 1401) * mV

        sigmas = np.zeros_like(delta_mus)

        prepare_bigger_fonts()
        plt.figure(figsize=(8, 6))
        plt.title(r'''How much diffusion is required for a gain 2.5 $\frac{\mathrm{Hz}}{\mathrm{mV}}$?''')

        for index,  delta_mu in enumerate(delta_mus):
            sigma = find_sigma_for_mu_producing_rate_gain(palmer_control,
                                                  mu=palmer_control.neuron_params.theta - delta_mu,
                                                  gain=2.5 * Hz / mV)
            sigmas[index] = sigma

        plt.xlabel(r" $\Delta V$ below threshold (mV)")
        plt.ylabel(r"$\sigma_v$ mV]")

        plt.plot(delta_mus / mV, sigmas / mV)
        plt.tight_layout()
        plt.show()

    def test_compute_all_sigmas_required_for_various_gains(self):
        delta_mus = np.linspace(15, 1, 1401) * mV
        mask = delta_mus > 5.6 * mV

        gains = np.arange(0.5, 5.2, step=0.5)

        prepare_bigger_fonts()
        plt.figure(figsize=(8, 6))
        plt.title(r'''How much diffusion is required for various gains $\frac{\mathrm{Hz}}{\mathrm{mV}}$?''')

        def compute_sigmas_for_gain(gain):
            return np.array([
                find_sigma_for_mu_producing_rate_gain(
                    palmer_control,
                    mu=(palmer_control.neuron_params.theta - delta_mu),
                    gain=gain * Hz / mV
                )
                for delta_mu in delta_mus[mask]
            ]) * volt

        sigmas_list = Parallel(n_jobs=-1)(
            delayed(compute_sigmas_for_gain)(gain)
            for gain in gains
        )
        '''
        for gain in gains:
            sigmas = np.array([find_sigma_for_mu_producing_rate_gain(palmer_control,
                                                  mu=palmer_control.neuron_params.theta - delta_mu,
                                                  gain=gain * Hz / mV) for delta_mu in delta_mus]) * volt

            plt.plot(delta_mus / mV, sigmas / mV, label=f"gain={gain}")
        '''
        for gain, sigmas in zip(gains, sigmas_list):
            plt.plot(delta_mus[mask] / mV, sigmas / mV, label=f"gain={gain}")

        plt.xlabel(r" $\Delta V$ below threshold (mV)")
        plt.ylabel(r"$\sigma_v$ [mV]")

        plt.ylim(1, 15)
        plt.legend()
        plt.tight_layout()
        plt.show()


    def test_plot_gain_computed_from_rate(self):
        gradients = SiegertGradients.for_experiment(palmer_control)
        mus = np.linspace(-65, -35, 1000) * mV

        dmu_grid = mus[1] - mus[0]
        step = int(np.round((0.1 * mV) / dmu_grid))
        delta_mu = mus[step:] - mus[:-step]
        mu_mid = mus[:-step] + delta_mu / 2

        prepare_bigger_fonts()
        plt.figure(figsize=(10, 8))
        plt.title(r'''Same as above but for $\Delta \mu$ = 0.1 mV''')

        for sigma in np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]):

            rates = np.array([gradients.firing_rate(mu_v=mu, sigma_v=sigma * mV) for mu in mus]) * Hz
            delta_rate = rates[step:] - rates[:-step]
            delta_rate_over_delta_mu = delta_rate / delta_mu
            plt.plot(mu_mid / mV, delta_rate_over_delta_mu * mV / Hz, label=r'$\sigma_v$=' + f"{sigma} mV" )

            print("\n==============================")
            print(f"Testing sigma = {sigma} mV")
            print("rates length:", len(rates))
            print("rates min/max:", np.nanmin(rates), np.nanmax(rates))
            # ---- Check for None ----
            if any(r is None for r in rates):
                print("❌ Found None in rates")

            # ---- Check for NaN / Inf ----
            print("NaNs in rates:", np.isnan(rates).sum())
            print("Infs in rates:", np.isinf(rates).sum())
            print("delta_rate length:", len(delta_rate))
            print("delta_rate min/max:", np.nanmin(delta_rate), np.nanmax(delta_rate))
            print("NaNs in delta_rate:", np.isnan(delta_rate).sum())
            print("Infs in delta_rate:", np.isinf(delta_rate).sum())
            # ---- Check if all zeros ----
            if np.all(delta_rate == 0 * Hz):
                print("⚠ delta_rate is all zero")

            # ---- Check delta_mu ----
            print("delta_mu min/max:", np.nanmin(delta_mu), np.nanmax(delta_mu))

            # ---- Check slicing validity ----
            if step >= len(rates):
                print("❌ step too large! step =", step)
                continue


        plt.axhline(y=2.5, linestyle='-.', label="Palmer gain")

        # Vertical line at x = -65 mV
        plt.axvline(x=palmer_control.neuron_params.theta / mV - 10, linestyle='--', label=r"10 mV bellow $\theta$")
        plt.axvline(x=palmer_control.neuron_params.theta / mV, linestyle='--', label=r"$\theta$", color='black')

        plt.xlabel("Membrane potential (mV)")
        plt.ylabel(r"Gain [$\frac{\mathrm{Hz}}{\mathrm{mV}}$]")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        show_plots_non_blocking()



if __name__ == '__main__':
    unittest.main()
