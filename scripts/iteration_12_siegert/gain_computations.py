"""
Siegert gain computations and Palmer fit.

Convention: test_* = sanity/unit tests (run with sanity test runner).
           test_script_* = runnable experiments/plots (run as scripts).
"""
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
from iteration_8_compute_mean_steady_state.models_and_configs import palmer_experiment_0_1_Hz_with_NMDA_block
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

# Palmer (MK801) experiment: rate 0.18 Hz → 0.05 Hz for Δμ ≈ 0.2 mV → gain = 0.65 Hz/mV
R_MK801_HZ = 0.05
R_CONTROL_HZ = 0.18
D_MU_PALMER_MV = 0.2
GAIN_PALMER_HZ_PER_MV = (R_CONTROL_HZ - R_MK801_HZ) / D_MU_PALMER_MV  # 0.65

from joblib import Parallel, delayed

def find_sigma_for_mu_producing_rate_gain(experiment: Experiment, mu, gain):
    siegert_gradient = SiegertGradients.for_experiment(experiment)
    sigma_sol = fsolve(func=lambda sigma: [siegert_gradient.d_rate_d_mu(mu_v=mu, sigma_v=sigma[0] * volt) - gain],
           x0=4 * mV)[0] * volt

    if sigma_sol < 0 * mV:
        return 0 * mV
    return sigma_sol

def find_sigma_for_mu_producing_rate(experiment: Experiment, mu, rate):
    siegert_gradient = SiegertGradients.for_experiment(experiment)
    sigma_sol = fsolve(func=lambda sigma: [siegert_gradient.firing_rate(mu_v=mu, sigma_v=sigma[0] * volt) - rate],
           x0=4 * mV)[0] * volt

    if sigma_sol < 0 * mV:
        return 0 * mV
    return sigma_sol



class GainScripts(unittest.TestCase):
    """Runnable scripts: gain plots and sigma scans. Use test_script_* prefix."""

    def test_script_plot_gain(self):
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

        plt.axhline(y=2.5, linestyle='-.', label="Palmer gain")

        rate_bulls_eye = gradients.firing_rate(mu_v=palmer_control.neuron_params.theta - 10 * mV, sigma_v=sigma_numerical_solution)
        plt.plot(mus / mV, gains, label=r'Sol: $\sigma_v$=' + f"{sigma_numerical_solution/mV : .3f} mV, r={rate_bulls_eye/Hz : .3f} Hz")

        # Vertical line at x = -65 mV
        plt.axvline(x=palmer_control.neuron_params.theta / mV - 10, linestyle='--', label=r"10 mV bellow $\theta$")
        plt.axvline(x=palmer_control.neuron_params.theta / mV, linestyle='--', label=r"$\theta$", color='black')

        plt.xlabel("Membrane potential (mV)")
        plt.ylabel(r"Gain [$\frac{\mathrm{Hz}}{\mathrm{mV}}$]")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        show_plots_non_blocking(caller_test_case=self)

    def test_script_plot_LIF_rate_for_interesting_model(self):
        gradients = SiegertGradients.for_experiment(palmer_control)
        mus = np.linspace(-65, -49.8, 1000) * mV

        prepare_bigger_fonts()
        plt.figure(figsize=(10, 8))

        sigma_numerical_solution_gain = find_sigma_for_mu_producing_rate_gain(palmer_control, mu=palmer_control.neuron_params.theta - 10 * mV,
                                                                    gain=2.5 * Hz / mV)

        sigma_numerical_solution_rate = find_sigma_for_mu_producing_rate(palmer_control,
                                                                              mu=palmer_control.neuron_params.theta - 10 * mV,
                                                                              rate=0.05 * Hz)

        for sigma in np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, sigma_numerical_solution_gain / mV]):
            rates = np.array([gradients.firing_rate(mu_v=mu, sigma_v=sigma * mV) for mu in mus]) / Hz
            plt.plot(mus / mV, rates, label=r'$\sigma_v$=' + f"{sigma :.3f} mV" )

        plt.axhline(y=0.3, linestyle='-.', label="Palmer gain")

        # Vertical line at x = -65 mV
        plt.axvline(x=palmer_control.neuron_params.theta / mV - 10, linestyle='--', label=r"10 mV bellow $\theta$")
        plt.axvline(x=palmer_control.neuron_params.theta / mV, linestyle='--', label=r"$\theta$", color='black')

        plt.ylim((0, 10))
        plt.xlabel("Membrane potential (mV)")
        plt.ylabel(r"Rate [Hz]")

        plt.title(r'''control_mu_to_sigma.sigmas =$\frac{d\mathrm(rate)}{d\mu}$
               for Palmer figure 2 e,f, where $\frac{\Delta r}{\Delta\mu_v} = \frac{0.25 Hz}{0.1 mV} = 2.5 \frac{\mathrm{Hz}}{\mathrm{mV}}$ \\''' +
                "Numerical Solutions: rate (" + r"$\mu_v$" + f"={palmer_control.neuron_params.theta - 10 * mV}, " + r"$\sigma_v$" +
                  f"={sigma_numerical_solution_rate / mV : .2f} mV) ={gradients.firing_rate(mu_v=palmer_control.neuron_params.theta - 10 * mV, sigma_v=sigma_numerical_solution_rate): .3f} Hz" + r"\\" +
                  "gain (" + r"$\mu_v$" + f"={palmer_control.neuron_params.theta - 10 * mV}, " + r"$\sigma_v$" +
                  f"={sigma_numerical_solution_rate / mV : .2f} mV) ={gradients.d_rate_d_mu(mu_v=palmer_control.neuron_params.theta - 10 * mV, sigma_v=sigma_numerical_solution_rate) * mV / Hz: .3f} " + r"$\frac{mV}{\mathrm{mV}}$ meaning \\" +
                  f"rate (" + r"$\mu_v$" + f"={palmer_control.neuron_params.theta - 9.9 * mV}, " + r"$\sigma_v$" +
                  f"={sigma_numerical_solution_rate / mV : .2f} mV) ={gradients.firing_rate(mu_v=palmer_control.neuron_params.theta - 9.9 * mV, sigma_v=sigma_numerical_solution_rate): .3f} Hz"
                )

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        show_plots_non_blocking(caller_test_case=self)

    def test_script_compute_all_sigmas_required_for_our_gain(self):
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
        show_plots_non_blocking(caller_test_case=self)

    def test_script_compute_all_sigmas_required_for_various_gains(self):
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
        show_plots_non_blocking(caller_test_case=self)

    def test_script_plot_gain_computed_from_rate(self):
        gradients = SiegertGradients.for_experiment(palmer_control)
        mus = np.linspace(-65, -35, 1000) * mV

        dmu_grid = mus[1] - mus[0]
        step = int(np.round((0.1 * mV) / dmu_grid))
        delta_mu = mus[step:] - mus[:-step]
        mu_mid = mus[:-step] + delta_mu / 2

        prepare_bigger_fonts()
        plt.figure(figsize=(10, 8))
        plt.title(r'''Same as above but for $\Delta \mu$ = 0.1 mV''')

        #for sigma in np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]):
        for sigma in np.array([0.5, 1, 2, 3, 4]):

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
        show_plots_non_blocking(caller_test_case=self)

        print("Checks ")
        print(f"rate (-60, 4 mV)  = {gradients.firing_rate(mu_v=-60 * mV, sigma_v=4 * mV)}")
        print(f"gain (-60, 4 mV)  = 2.2 from image => for 0.1 mV we have 0.22 mV change")
        print(f"rate (-59.9, 4 mV) = {gradients.firing_rate(mu_v=(-60 * mV + 0.1 *mV), sigma_v=4 * mV)}")


from scipy.optimize import least_squares


# first variable is sigma, because this is the scan variable
def error_function(sigma, experiment, mu, desired_rate, desired_gain, weight=1):

    siegert_gradient = SiegertGradients.for_experiment(experiment)
    """
    This function computes the residuals for the least squares optimization.
    The residuals are the differences between the model's predicted values
    and the desired values for both rate and gain.
    """
    # Firing rate residual
    predicted_rate = siegert_gradient.firing_rate(mu_v=mu, sigma_v=sigma[0] * volt)
    rate_residual = predicted_rate - desired_rate

    # Firing gain residual
    predicted_gain = siegert_gradient.d_rate_d_mu(mu_v=mu, sigma_v=sigma[0] * volt)
    gain_residual = predicted_gain - desired_gain

    # Return both residuals as a vector (for least squares)
    return [rate_residual / mV, weight * gain_residual * mV / Hz]

def error_function_on_two_rates(sigma, experiment, mu, desired_rate_1, desired_rate_2, dv, weight=1):

    siegert_gradient = SiegertGradients.for_experiment(experiment)
    """
    This function computes the residuals for the least squares optimization.
    The residuals are the differences between the model's predicted values
    and the desired values for both rate and gain.
    """
    # Firing rate residual
    predicted_rate = siegert_gradient.firing_rate(mu_v=mu, sigma_v=sigma[0] * volt)
    rate_residual = predicted_rate - desired_rate_1

    # Firing gain residual
    predicted_gain = siegert_gradient.d_rate_d_mu(mu_v=mu + dv, sigma_v=sigma[0] * volt)
    gain_residual = predicted_gain - desired_gain

    # Return both residuals as a vector (for least squares)
    return [rate_residual / mV, weight * gain_residual * mV / Hz]

def find_sigma_with_levenberg_marquardt(experiment, mu, desired_rate, desired_gain, weight=1):
    # Initial guess for sigma
    initial_guess = [4 * mV]

    # Use least_squares to minimize the error function using the Levenberg-Marquardt algorithm
    result = least_squares(
        error_function, initial_guess,
        args=(experiment, mu, desired_rate, desired_gain, weight),
        method='lm'  # Levenberg-Marquardt method
    )

    print(result)
    sigma_sol = result.x[0] * volt

    if sigma_sol < 0 * mV:
        return 0 * mV

    return sigma_sol


def _residual_mu_sigma(x, experiment, r_target_hz, gain_hz_per_mv):
    """Residual for least_squares: [rate_err, gain_err] for (mu_mv, sigma_mv) in x."""
    mu_mv, sigma_mv = x[0], x[1]
    sg = SiegertGradients.for_experiment(experiment)
    r = float(sg.firing_rate(mu_v=mu_mv * mV, sigma_v=sigma_mv * mV) / Hz)
    g = float(sg.d_rate_d_mu(mu_v=mu_mv * mV, sigma_v=sigma_mv * mV) * mV / Hz)
    return [r - r_target_hz, g - gain_hz_per_mv]


def find_mu_sigma_for_rate_and_gain(experiment, r_target_hz, gain_hz_per_mv, sigma_bounds=(0.1, 10)):
    """Find (mu_mV, sigma_mV) such that rate = r_target_hz and dr/dmu = gain_hz_per_mv. Returns (mu, sigma) in mV."""
    from scipy.optimize import least_squares
    # Initial guess: mu near threshold, sigma a few mV
    theta_mv = float(experiment.neuron_params.theta / mV)
    x0 = [theta_mv - 5, 2.0]
    result = least_squares(
        _residual_mu_sigma,
        x0,
        args=(experiment, r_target_hz, gain_hz_per_mv),
        bounds=([-80, sigma_bounds[0]], [theta_mv + 5, sigma_bounds[1]]),
    )
    return result.x[0], result.x[1]


class SolveForGainAndRateScripts(unittest.TestCase):
    """Runnable script: LM fit for sigma given rate and gain."""

    def test_script_lm(self):
        sigma_sol = find_sigma_with_levenberg_marquardt(palmer_experiment_0_1_Hz_with_NMDA_block, mu=-60 * mV,
                                                      desired_rate=0.05 * Hz, desired_gain=2.5 * Hz / mV, weight=0.1)
        print(sigma_sol)
        gradient = SiegertGradients.for_experiment(palmer_experiment_0_1_Hz_with_NMDA_block)
        print(f"rate mu, sigma = {gradient.firing_rate(mu_v = -60 * mV, sigma_v = sigma_sol) / Hz}" )
        print(f"gain mu, sigma = {gradient.d_rate_d_mu(mu_v = -60 * mV, sigma_v = sigma_sol) * mV / Hz}" )
        print(f"rate mu+0.1, sigma = {gradient.firing_rate(mu_v = (-60 + 0.1) * mV, sigma_v = sigma_sol) / Hz}" )
        print(f"rate mu+0.1, sigma+0.1 = {gradient.firing_rate(mu_v = (-60 + 0.1) * mV, sigma_v = sigma_sol + 0.3 * mV) / Hz}" )
        print(f"rate mu+0.1, sigma+0.1 = {gradient.firing_rate(mu_v = (-60 + 0.1) * mV, sigma_v = sigma_sol + 0.36 * mV) / Hz}" )

        '''
        rate mu, sigma = 7.9710060010441355
        gain mu, sigma = 2.495292619925241
        rate mu+0.1, sigma = 8.288204181365028
        '''


if __name__ == '__main__':
    unittest.main()
