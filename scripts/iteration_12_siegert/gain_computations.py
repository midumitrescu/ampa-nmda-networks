"""
Siegert gain computations and Palmer fit.

Convention: test_* = sanity/unit tests (in tests/, run with sanity test runner).
           test_scripts_* = runnable experiments/plots (run directly in IntelliJ).
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
try:
    from utils import ExtendedDict
except ImportError:
    from src.utils import ExtendedDict
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


# Parametrizable defaults for "rate at baseline, desired gain, step in mu" plots (ExtendedDict = attribute-style access)
def default_rate_gain_params():
    """Return rate/gain plot params as ExtendedDict so you can use params.rate_baseline_Hz, params.gain_target_Hz_per_mV, etc."""
    return ExtendedDict({
        "rate_baseline_Hz": 0.05,
        "rate_after_dmu_Hz": None,  # if set, Taylor target at mu+delta_mu; else rate_baseline + gain * delta_mu
        "gain_target_Hz_per_mV": 2.5,
        "delta_mu_mV": 0.1,
        "mu_offset_below_theta_mV": 10.0,  # mu_baseline = theta - this
    })


def solve_mu_sigma_via_fsolve(experiment, params):
    """Search for (mu, sigma) using fsolve: fix mu_baseline = theta - offset, find sigma such that rate(mu_baseline, sigma) = rate_baseline_Hz.
    Returns ExtendedDict with sigma, mu_baseline, siegert_gradient, rates and Taylor quantities for plotting."""
    rate_baseline = params.rate_baseline_Hz * Hz
    delta_mu = params.delta_mu_mV * mV
    mu_offset = params.mu_offset_below_theta_mV * mV
    mu_baseline = experiment.neuron_params.theta - mu_offset

    sg = SiegertGradients.for_experiment(experiment)
    sigma = find_sigma_for_mu_producing_rate(experiment, mu_baseline, rate_baseline)
    rate_at_baseline = float(sg.firing_rate(mu_v=mu_baseline, sigma_v=sigma) / Hz)
    gain_at_baseline = float(sg.d_rate_d_mu(mu_v=mu_baseline, sigma_v=sigma) * mV / Hz)
    mu_shifted = mu_baseline + delta_mu
    rate_actual_shifted = float(sg.firing_rate(mu_v=mu_shifted, sigma_v=sigma) / Hz)
    rate_after = params.get("rate_after_dmu_Hz")
    if rate_after is not None:
        rate_taylor_shifted = rate_after
    else:
        rate_taylor_shifted = rate_at_baseline + params.get("gain_target_Hz_per_mV") * params.get("delta_mu_mV")

    return ExtendedDict({
        "sigma": sigma,
        "mu_baseline": mu_baseline,
        "theta": theta,
        "rate_baseline_Hz": rate_at_baseline,
        "gain_at_baseline_Hz_per_mV": gain_at_baseline,
        "rate_actual_shifted_Hz": rate_actual_shifted,
        "rate_taylor_shifted_Hz": rate_taylor_shifted,
        "taylor_error_Hz": rate_actual_shifted - rate_taylor_shifted,
        "params": params,
        "siegert_gradient": sg,
    })


def compute_rate_gain_at_params(experiment, params=None):
    """Convenience wrapper: same as solve_mu_sigma_via_fsolve (find sigma via fsolve for rate_baseline at mu_baseline)."""
    return solve_mu_sigma_via_fsolve(experiment, params)


def plot_rate_and_gain_with_taylor(ax_gain, ax_rate, experiment, solution, params):
    """Plot (1) gain vs μ with solution σ and (2) actual rate vs μ with Taylor approximations and error shading.
    solution: ExtendedDict from solve_mu_sigma_via_fsolve (sigma, mu_baseline, siegert_gradient, rate_baseline_Hz, etc.).
    params: ExtendedDict with gain_target_Hz_per_mV, delta_mu_mV, etc."""
    sg = solution.siegert_gradient
    sigma_sol = solution.sigma
    mu_baseline = solution.mu_baseline
    gain_target = params.gain_target_Hz_per_mV
    mus = np.linspace(-65, -35, 500) * mV

    # Left: gain vs mu
    for sigma in np.array([0.5, 1, 2, 3, 4]):
        gains = np.array([float(sg.d_rate_d_mu(mu_v=mu, sigma_v=sigma * mV) * mV / Hz) for mu in mus])
        ax_gain.plot(mus / mV, gains, label=rf'$\sigma_v$={sigma} mV')
    gains_sol = np.array([float(sg.d_rate_d_mu(mu_v=mu, sigma_v=sigma_sol) * mV / Hz) for mu in mus])
    ax_gain.plot(mus / mV, gains_sol, 'k-', linewidth=2, label=rf'Sol $\sigma$={float(sigma_sol/mV):.3f} mV')
    ax_gain.axhline(y=gain_target, linestyle='-.', color='gray', label=f'Target gain {gain_target}')
    ax_gain.axvline(x=float(mu_baseline / mV), linestyle='--', alpha=0.7, label=rf'$\mu$ baseline')
    ax_gain.axvline(x=float(solution.theta / mV), linestyle='--', color='black', label=r'$\theta$')
    ax_gain.set_xlabel(r'$\mu$ (mV)')
    ax_gain.set_ylabel(r'Gain [Hz/mV]')
    mu_base_mV = float(mu_baseline / mV)
    sigma_sol_mV = float(sigma_sol / mV)
    r_baseline_Hz = float(sg.firing_rate(mu_v=mu_baseline, sigma_v=sigma_sol) / Hz)
    r_plus_dmu_Hz = float(sg.firing_rate(mu_v=mu_baseline + 0.1 * mV, sigma_v=sigma_sol) / Hz)
    ax_gain.set_title(
        rf'Gain vs $\mu$'
        + rf' \\ $\mu_0$ = {mu_base_mV:.3f} mV, $\sigma_{{sol}}$ = {sigma_sol_mV:.3f} mV, r = {r_baseline_Hz:.3f} Hz'
        + rf' \\ $\mu_0 + \Delta\mu$ = {mu_base_mV + 0.1:.3f} mV, $\sigma_{{sol}}$ = {sigma_sol_mV:.3f} mV, r = {r_plus_dmu_Hz:.3f} Hz'
    )
    ax_gain.legend(loc='upper right', fontsize=8)
    ax_gain.grid(True, alpha=0.3)

    # Right: rate vs mu with Taylor-error shaded area
    mu_plot = np.linspace(-60.5, -59.5, 500)
    desired_gain = (solution.rate_taylor_shifted_Hz - solution.rate_baseline_Hz) / params.delta_mu_mV
    r_actual = np.array([float(sg.firing_rate(mu_v=mu * mV, sigma_v=sigma_sol) / Hz) for mu in mu_plot])
    r_taylor = solution.rate_baseline_Hz + desired_gain * (mu_plot - mu_base_mV)
    dr_dmu_at_base = float(sg.d_rate_d_mu(mu_v=mu_baseline, sigma_v=sigma_sol) * mV / Hz)
    r_linear_taylor_at_mu0 = solution.rate_baseline_Hz + dr_dmu_at_base * (mu_plot - mu_base_mV)
    ax_rate.fill_between(mu_plot, r_taylor, r_actual, alpha=0.6, color='orange', label=r'Desired $dr/d\mu$ vs actual')
    ax_rate.plot(mu_plot, r_actual, 'b-', linewidth=2, label='Siegert (actual)')
    ax_rate.plot(mu_plot, r_taylor, 'r--', linewidth=1.5, label=rf'Taylor at $\mu_0$ for desired slope={desired_gain:.0f}')
    ax_rate.plot(mu_plot, r_linear_taylor_at_mu0, 'g-.', linewidth=1.5, label=rf'Taylor at $\mu_0$: $dr/d\mu$={dr_dmu_at_base:.3f}')

    mus_wide = np.linspace(-63, -57, 1000) * mV
    rates_wide = np.array([float(sg.firing_rate(mu_v=mu, sigma_v=sigma_sol) / Hz) for mu in mus_wide])
    ax_rate.plot(mus_wide / mV, rates_wide, 'b-', alpha=0.4, linewidth=1)

    ax_rate.axhline(y=solution.rate_baseline_Hz, linestyle='-.', color='gray', label=f'Baseline {solution.rate_baseline_Hz:.3f} Hz')
    ax_rate.axvline(x=mu_base_mV, linestyle='--', alpha=0.7)
    ax_rate.set_xlabel(r'$\mu$ (mV)')
    ax_rate.set_ylabel(r'Rate [Hz]')
    ax_rate.set_title(rf'Rate vs $\mu$ - Taylor error = {solution.taylor_error_Hz:.4f} Hz at $\mu+{params.delta_mu_mV}$ mV')
    ax_rate.legend(loc='upper right', fontsize=8)
    ax_rate.grid(True, alpha=0.3)
    ax_rate.set_ylim(0, max(2, 1.1 * max(r_actual.max(), r_taylor.max())))


class GainScripts(unittest.TestCase):
    """Runnable scripts: gain plots and sigma scans. test_scripts_* = run directly in IntelliJ."""

    def test_scripts_plot_gain(self, params=None):
        """Gain vs μ for several σ; sigma chosen so rate = rate_baseline at mu_baseline. Params: rate_baseline_Hz, gain_target_Hz_per_mV, delta_mu_mV, mu_offset_below_theta_mV."""
        if params is None:
            params = default_rate_gain_params()
        p = compute_rate_gain_at_params(palmer_control, params)
        gradients = p.siegert_gradient
        sigma_sol = p.sigma
        gain_target = params.gain_target_Hz_per_mV
        mu_off = params.mu_offset_below_theta_mV
        mus = np.linspace(-65, -35, 1000) * mV

        prepare_bigger_fonts()
        plt.figure(figsize=(10, 8))
        plt.title(rf"Gain $dr/d\mu$ vs $\mu$ — baseline rate={params.rate_baseline_Hz} Hz, target gain={gain_target} Hz/mV, $\Delta\mu$={params.delta_mu_mV} mV")

        for sigma in np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]):
            gains = [gradients.d_rate_d_mu(mu_v=mu, sigma_v=sigma * mV) for mu in mus] * (mV / Hz)
            plt.plot(mus / mV, gains, label=r'$\sigma_v$=' + f"{sigma} mV")

        gains_sol = [gradients.d_rate_d_mu(mu_v=mu, sigma_v=sigma_sol) for mu in mus] * (mV / Hz)
        plt.axhline(y=gain_target, linestyle='-.', label=f"Target gain {gain_target} Hz/mV")
        plt.plot(mus / mV, gains_sol, label=r'Sol: $\sigma_v$=' + f"{sigma_sol/mV:.3f} mV, r={p.rate_baseline_Hz:.3f} Hz")

        plt.axvline(x=p.theta / mV - mu_off, linestyle='--', label=rf"$\mu$ = $\theta$ - {mu_off} mV")
        plt.axvline(x=p.theta / mV, linestyle='--', label=r"$\theta$", color='black')
        plt.xlabel(r"Membrane potential $\mu$ (mV)")
        plt.ylabel(r"Gain [Hz/mV]")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        show_plots_non_blocking(caller_test_case=self)

    def test_scripts_plot_LIF_rate_for_interesting_model(self, params=None):
        """Rate vs μ for sigma that gives rate_baseline at mu_baseline; show rate at mu_baseline + delta_mu (actual vs Taylor)."""
        if params is None:
            params = default_rate_gain_params()
        p = compute_rate_gain_at_params(palmer_control, params)
        gradients = p.siegert_gradient
        sigma_sol = p.sigma
        mu_off = params.mu_offset_below_theta_mV
        mus = np.linspace(-65, -49.8, 1000) * mV

        prepare_bigger_fonts()
        plt.figure(figsize=(10, 8))

        for sigma in np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, sigma_sol / mV]):
            rates = np.array([gradients.firing_rate(mu_v=mu, sigma_v=sigma * mV) for mu in mus]) / Hz
            plt.plot(mus / mV, rates, label=r'$\sigma_v$=' + f"{sigma:.3f} mV")

        plt.axhline(y=p.rate_baseline_Hz, linestyle='-.', label=f"Baseline rate {p.rate_baseline_Hz:.3f} Hz")
        plt.axhline(y=p.rate_taylor_shifted_Hz, linestyle=':', label=rf"Taylor at $\mu+\Delta\mu$ = {p.rate_taylor_shifted_Hz:.3f} Hz")
        plt.axhline(y=p.rate_actual_shifted_Hz, linestyle='--', label=rf"Actual at $\mu+\Delta\mu$ = {p.rate_actual_shifted_Hz:.3f} Hz (err={p.taylor_error_Hz:.4f})")

        plt.axvline(x=p.theta / mV - mu_off, linestyle='--', label=rf"$\mu$ = $\theta$ - {mu_off} mV")
        plt.axvline(x=p.theta / mV, linestyle='--', label=r"$\theta$", color='black')
        plt.ylim((0, 10))
        plt.xlabel(r"Membrane potential $\mu$ (mV)")
        plt.ylabel(r"Rate [Hz]")
        plt.title(rf"Rate vs $\mu$ - $\sigma$ s.t. r($\theta$-{mu_off}) = {params.rate_baseline_Hz} Hz; "
                  rf"at $\mu+{params.delta_mu_mV}$ mV: Taylor={p.rate_taylor_shifted_Hz:.3f}, actual={p.rate_actual_shifted_Hz:.3f}, error={p.taylor_error_Hz:.4f} Hz")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        show_plots_non_blocking(caller_test_case=self)

    def test_scripts_plot_rate_and_gain_combined(self, params=default_rate_gain_params()):
        """Single figure: (1) solve for (μ, σ) via fsolve, (2) plot gain vs μ and rate vs μ with Taylor approximations."""
        # First part: search for (mu, sigma) using fsolve
        solution = solve_mu_sigma_via_fsolve(palmer_control, params)
        mu_base_mV = float(solution.mu_baseline / mV)
        sigma_sol_mV = float(solution.sigma / mV)
        rate_at_mu_plus_01 = float(solution.siegert_gradient.firing_rate(
            mu_v=solution.mu_baseline + 0.1 * mV, sigma_v=solution.sigma) / Hz)
        print(f"Rate at μ + 0.1 mV: r(μ_base + 0.1 mV, σ_sol) = {rate_at_mu_plus_01:.4f} Hz  [μ_base = {mu_base_mV:.3f} mV, σ_sol = {sigma_sol_mV:.3f} mV]")

        # Second part: plot algorithm result and actual vs desired rate (Taylor)
        prepare_bigger_fonts()
        fig, (ax_gain, ax_rate) = plt.subplots(1, 2, figsize=(14, 6))
        plot_rate_and_gain_with_taylor(ax_gain, ax_rate, palmer_control, solution, params)

        fig.suptitle(rf'Rate baseline = {params.rate_baseline_Hz} Hz, target gain = {params.gain_target_Hz_per_mV} Hz/mV, $\Delta\mu$ = {params.delta_mu_mV} mV', fontsize=11)
        plt.tight_layout()
        show_plots_non_blocking(caller_test_case=self)

    def test_scripts_plot_for_relaxed_conditions(self):

        self.test_scripts_plot_rate_and_gain_combined(ExtendedDict({
        "rate_baseline_Hz": 0.05,
        "rate_after_dmu_Hz": None,  # if set, Taylor target at mu+delta_mu; else rate_baseline + gain * delta_mu
        "gain_target_Hz_per_mV": 2.5,
        "delta_mu_mV": 0.1,
        "mu_offset_below_theta_mV": 10.0,  # mu_baseline = theta - this
    }))

    def test_scripts_compute_all_sigmas_required_for_our_gain(self):
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

    def test_scripts_compute_all_sigmas_required_for_various_gains(self):
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

    def test_scripts_plot_gain_computed_from_rate(self):
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
    gain_residual = predicted_gain - desired_rate_2

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

    def test_scripts_lm(self):
        sigma_sol = find_sigma_with_levenberg_marquardt(palmer_experiment_0_1_Hz_with_NMDA_block, mu=-60 * mV,
                                                      desired_rate=0.05 * Hz, desired_gain=2.5 * Hz / mV, weight=100)
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
