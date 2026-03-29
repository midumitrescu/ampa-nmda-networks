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
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients, I_mu_sigma
from iteration_12_transfer_function_of_lif_neurons.config import DiffusionLIFConfig

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

class RateGainSearchParams:

    def __init__(self, rate_nmda_block, rate_with_nmda, mu_v_nmda_block, mu_v_with_nmda, d_sigma = 0.2):
        self.rat_nmda_block_unitless = rate_nmda_block
        self.rate_with_nmda_unitless = rate_with_nmda
        self.mu_v_nmda_block_unitless = mu_v_nmda_block
        self.mu_v_with_nmda_unitless = mu_v_with_nmda

        self.rate_nmda_block = rate_nmda_block * Hz
        self.rate_with_nmda = rate_with_nmda * Hz
        self.mu_v_nmda_block = mu_v_nmda_block * mV
        self.mu_v_with_nmda = mu_v_with_nmda * mV

        self.d_rate = self.rate_with_nmda - self.rate_nmda_block
        self.d_rate_unitless = self.d_rate / Hz

        self.d_mu = self.mu_v_with_nmda - self.mu_v_nmda_block
        self.d_mu_unitless = self.d_mu / mV

        self.gain = self.d_rate / self.d_mu
        self.gain_unitless = self.gain * mV / Hz

        self.d_sigma = d_sigma * mV
        self.d_sigma_unitless = self.d_sigma / mV

def find_sigma_for_mu_producing_rate_from_lif_config(lif_config: DiffusionLIFConfig, params: RateGainSearchParams):
    siegert_gradient = SiegertGradients.for_lif_config(lif_config)
    sigma_sol = fsolve(func=lambda sigma: [siegert_gradient.firing_rate(mu_v=params.mu_v_nmda_block, sigma_v=sigma[0] * volt) - params.rate_nmda_block],
                       x0=4 * mV)[0] * volt
    if sigma_sol < 0 * mV:
        return 0 * mV
    return sigma_sol

def find_sigma_for_mu_producing_rate(experiment: Experiment, params: RateGainSearchParams):
    return find_sigma_for_mu_producing_rate_from_lif_config(DiffusionLIFConfig.from_experiment(experiment), params)

# Parametrizable defaults for "rate at baseline, desired gain, step in mu" plots (ExtendedDict = attribute-style access)
default_rate_gain_params = RateGainSearchParams(rate_nmda_block=0.05, rate_with_nmda=0.3, mu_v_nmda_block=-60, mu_v_with_nmda=-60 + 0.1)

def solve_mu_sigma_via_fsolve_for_LIF_config(lif_config: DiffusionLIFConfig, params: RateGainSearchParams):
    sieger_gradient = SiegertGradients.for_lif_config(lif_config)

    sigma = find_sigma_for_mu_producing_rate_from_lif_config(lif_config, params)
    rate_nmda_block_recomputed = sieger_gradient.firing_rate(mu_v=params.mu_v_nmda_block, sigma_v=sigma)
    print(
        f"mu_v {params.mu_v_nmda_block_unitless: .3f} mV, sigma={sigma: .3f} mV rate computed = {rate_nmda_block_recomputed}")

    d_rate_d_mu = sieger_gradient.d_rate_d_mu(mu_v=params.mu_v_nmda_block, sigma_v=sigma)
    taylor_rate_actual = params.rate_nmda_block + d_rate_d_mu * params.d_mu
    taylor_rate_wanted = params.rate_nmda_block + params.gain * params.d_mu

    rate_delta_mu = sieger_gradient.firing_rate(mu_v=params.mu_v_with_nmda, sigma_v=sigma) / Hz

    return ExtendedDict({
        "sigma": sigma,
        "mu_nmda_block": params.mu_v_nmda_block_unitless,
        "theta": lif_config.theta,
        "rate_nmda_block_recomputed": rate_nmda_block_recomputed / Hz,
        "rate_delta_mu_Hz": rate_delta_mu,
        "taylor_error_Hz": rate_nmda_block_recomputed - taylor_rate_actual,
        "gain_error_Hz": rate_nmda_block_recomputed - taylor_rate_wanted,
        "params": params,
        "siegert_gradient": sieger_gradient
    })

def solve_mu_sigma_via_fsolve(experiment: Experiment, params: RateGainSearchParams):
    """Search for (mu, sigma) using fsolve: fix mu_nmda_block = theta - offset, find sigma such that rate(mu_nmda_block, sigma) = rate_nmda_block_recomputed.
    Returns ExtendedDict with sigma, mu_nmda_block, siegert_gradient, rates and Taylor quantities for plotting."""

    sieger_gradient = SiegertGradients.for_experiment(experiment)
    sigma = find_sigma_for_mu_producing_rate(experiment, params)
    rate_nmda_block_recomputed = sieger_gradient.firing_rate(mu_v=params.mu_v_nmda_block, sigma_v=sigma)
    print(f"mu_v {params.mu_v_nmda_block_unitless: .3f} mV, sigma={sigma: .3f} mV rate computed = {rate_nmda_block_recomputed}")

    d_rate_d_mu = sieger_gradient.d_rate_d_mu(mu_v=params.mu_v_nmda_block, sigma_v=sigma)
    taylor_rate_actual = params.rate_nmda_block + d_rate_d_mu * params.d_mu
    taylor_rate_wanted = params.rate_nmda_block + params.gain * params.d_mu

    rate_delta_mu = sieger_gradient.firing_rate(mu_v=params.mu_v_with_nmda, sigma_v=sigma) / Hz

    return ExtendedDict({
        "sigma": sigma,
        "mu_nmda_block": params.mu_v_nmda_block_unitless,
        "theta": experiment.neuron_params.theta,
        "rate_nmda_block_recomputed": rate_nmda_block_recomputed / Hz,
        "rate_delta_mu_Hz": rate_delta_mu,
        "taylor_error_Hz": rate_nmda_block_recomputed - taylor_rate_actual,
        "gain_error_Hz": rate_nmda_block_recomputed - taylor_rate_wanted,
        "params": params,
        "siegert_gradient": sieger_gradient
    })

def compute_rate_gain_at_params(experiment, params=None):
    """Convenience wrapper: same as solve_mu_sigma_via_fsolve (find sigma via fsolve for rate_baseline at mu_nmda_block)."""
    return solve_mu_sigma_via_fsolve(experiment, params)


def plot_rate_and_gain_with_taylor(experiment: Experiment, params: RateGainSearchParams, solution: ExtendedDict, test: unittest.TestCase, lim=None):
    prepare_bigger_fonts()
    fig, (ax_gain, ax_rate) = plt.subplots(1, 2, figsize=(16, 8))

    siegert_gradient, sigma_solution = plot_gain(ax_gain, experiment, params, solution, lim=lim)

    plot_taylor_expansion(ax_rate, params, sigma_solution, solution)

    fig.suptitle(
        rf'$\theta - \mu_0 $ = {(experiment.neuron_params.theta - params.mu_v_nmda_block) / mV: .2f} mV \\ target $r(\mu_0)$ = {params.rate_nmda_block / Hz} Hz, target $r(\mu_0 + \Delta \mu)$ = {params.rate_with_nmda_unitless: .2f} Hz $\\$target gain = {params.gain_unitless :.3f} Hz/mV, $\Delta\mu$ = {params.d_mu_unitless :.2f} mV, $\Delta\sigma$ = {params.d_sigma_unitless :.2f} mV')
    plt.tight_layout()
    show_plots_non_blocking(caller_test_case=test)


def plot_taylor_expansion(ax_rate, params, sigma_solution, solution):
    siegert_gradient = solution.siegert_gradient
    # Right: rate vs mu with Taylor-error shaded area
    mu_taylor = np.linspace(params.mu_v_nmda_block - 0.5 * mV, params.mu_v_with_nmda + 0.5 * mV, 1000)
    mu_plot = np.linspace(params.mu_v_nmda_block - 1 * mV, params.mu_v_nmda_block + 1 * mV, 1500)
    desired_gain = params.gain_unitless
    actual_rate = siegert_gradient.firing_rate(mu_v=params.mu_v_with_nmda, sigma_v=sigma_solution) / Hz

    r_siegert = np.array([siegert_gradient.firing_rate(mu_v=mu, sigma_v=sigma_solution) / Hz for mu in mu_plot])
    r_taylor_slope_desired_gain = params.rate_nmda_block + params.gain * (mu_taylor - params.mu_v_nmda_block)
    dr_dmu_at_base = siegert_gradient.d_rate_d_mu(mu_v=params.mu_v_nmda_block, sigma_v=sigma_solution)
    r_linear_taylor_at_mu0 = params.rate_nmda_block + dr_dmu_at_base * (mu_taylor - params.mu_v_nmda_block)

    r_taylor_slope_desired_gain = r_taylor_slope_desired_gain / Hz
    r_linear_taylor_at_mu0 = r_linear_taylor_at_mu0 / Hz

    ax_rate.plot(mu_plot / mV, r_siegert, 'b-', linewidth=2, label='Siegert (actual)')
    ax_rate.plot(mu_taylor / mV, r_taylor_slope_desired_gain, 'r-.', linewidth=1.5, label=rf'$r = r_0 + \mathrm{{gain}} \cdot \Delta \mu$ for desired gain/slope={desired_gain:.3f} Hz/mV')
    ax_rate.plot(mu_taylor / mV, r_linear_taylor_at_mu0, 'g--', linewidth=1.5, label=rf'$r = r_0 + \frac{{dr}}{{d\mu}} \cdot \Delta \mu$ for $dr/d\mu$ at $\mu_0$={dr_dmu_at_base * mV / Hz:.3f} Hz/mV')

    ax_rate.plot(params.mu_v_with_nmda / mV, params.rate_with_nmda_unitless, 'ro', markersize=6, label=f'Desired rate {params.rate_with_nmda_unitless: .2f} Hz')
    ax_rate.plot(params.mu_v_with_nmda / mV, actual_rate, 'bo', markersize=6, label=f'Actual rate {actual_rate:.2f} Hz')

    ax_rate.axhline(y=solution.rate_nmda_block_recomputed, linestyle='-.', color='gray', label=f'Baseline {solution.rate_nmda_block_recomputed:.3f} Hz')
    ax_rate.axvline(x=params.mu_v_nmda_block / mV, linestyle='--', alpha=0.7)

    ax_rate.set_xlabel(r'$\mu$ (mV)')
    ax_rate.set_ylabel(r'Rate [Hz]')
    ax_rate.set_title(
        rf'Rate vs $\mu$ \\ Taylor approximation error = {solution.taylor_error_Hz / Hz:.4f} Hz at $\mu+{params.d_mu_unitless : .2f}$ mV. \\ $\Delta$ to Palmer experiment {solution.gain_error_Hz / Hz:.4f} Hz')
    ax_rate.legend(loc='upper right', bbox_to_anchor=(1.02, 0.8))
    ax_rate.grid(True, alpha=0.3)
    ax_rate.set_ylim(0, 0.75)


def plot_gain(ax_gain, experiment, params: RateGainSearchParams, solution, lim=None):
    """Plot (1) gain vs μ with solution σ and (2) actual rate vs μ with Taylor approximations and error shading.
    solution: ExtendedDict from solve_mu_sigma_via_fsolve (sigma, mu_nmda_block, siegert_gradient, rate_nmda_block_recomputed, etc.).
    params: ExtendedDict with gain_target_Hz_per_mV, delta_mu_mV, etc."""
    siegert_gradient = solution.siegert_gradient
    sigma_solution = solution.sigma
    gain_target = params.gain_unitless
    mus = np.linspace(-65, -35, 500) * mV
    # Left: gain vs mu
    for sigma in np.array([0.5, 1, 2, 3, 4]):
        gains = np.array([siegert_gradient.d_rate_d_mu(mu_v=mu, sigma_v=sigma * mV) * mV / Hz for mu in mus])
        ax_gain.plot(mus / mV, gains, label=rf'$\sigma_v$={sigma} mV')
    gains_for_sigma_solution = np.array(
        [siegert_gradient.d_rate_d_mu(mu_v=mu, sigma_v=sigma_solution) * mV / Hz for mu in mus])
    ax_gain.plot(mus / mV, gains_for_sigma_solution, 'k-', linewidth=2.6,
                 label=rf'$\sigma_{{sol}}$={sigma_solution / mV:.3f} mV for rate={params.rat_nmda_block_unitless: .2f} Hz')
    ax_gain.axhline(y=gain_target, linestyle='-.', color='red', label=f'Target gain {gain_target: .2f}')
    ax_gain.axvline(x=params.mu_v_nmda_block_unitless, linestyle='--', alpha=0.7, label=r'$\mu$ baseline')
    ax_gain.axvline(x=experiment.neuron_params.theta / mV, linestyle='--', color='black', label=r'$\theta$')
    ax_gain.set_xlabel(r'$\mu$ (mV)')
    ax_gain.set_ylabel(r'Gain [Hz/mV]')
    r_nmda_block = siegert_gradient.firing_rate(mu_v=params.mu_v_nmda_block, sigma_v=sigma_solution) / Hz
    r_with_nmda = siegert_gradient.firing_rate(mu_v=params.mu_v_with_nmda, sigma_v=sigma_solution) / Hz
    r_with_nmda_and_d_sigma = siegert_gradient.firing_rate(mu_v=params.mu_v_with_nmda, sigma_v=sigma_solution + params.d_sigma) / Hz
    ax_gain.set_title(
        r'Gain vs $\mu$'
        + rf' \\ $\mu_0$ = {params.mu_v_nmda_block_unitless:.3f} mV, $\sigma_{{sol}}$ = {sigma_solution / mV:.3f} mV, r = {r_nmda_block:.3f} Hz'
        + rf' \\ $\mu_0 + \Delta\mu$ = {params.mu_v_with_nmda_unitless:.3f} mV, $\sigma_{{sol}}$ = {sigma_solution / mV:.3f} mV, r = {r_with_nmda:.3f} Hz'
        + rf' \\ $\mu_0 + \Delta\mu$ = {params.mu_v_with_nmda_unitless:.3f} mV, $\sigma_{{sol}} + \Delta\sigma $ = {(sigma_solution + params.d_sigma) / mV:.3f} mV, r = {r_with_nmda_and_d_sigma:.3f} Hz'
    )
    ax_gain.legend(loc='upper right')
    if lim is not None:
        ax_gain.set_ylim((0, lim))
    ax_gain.grid(True, alpha=0.3)
    return siegert_gradient, sigma_solution

def plot_rate_and_gain_combined(params=default_rate_gain_params, test: unittest.TestCase=None):
    """Single figure: (1) solve for (μ, σ) via fsolve, (2) plot gain vs μ and rate vs μ with Taylor approximations."""
    # First part: search for (mu, sigma) using fsolve
    solution = solve_mu_sigma_via_fsolve(palmer_control, params)
    rate_at_mu = solution.siegert_gradient.firing_rate(mu_v=params.mu_v_nmda_block, sigma_v=solution.sigma) / Hz
    rate_at_mu_plus_01 = solution.siegert_gradient.firing_rate(mu_v=params.mu_v_with_nmda, sigma_v=solution.sigma) / Hz
    print(f"Rate at [μ_base = {params.mu_v_nmda_block_unitless:.3f} mV, σ_sol = {solution.sigma:.3f} mV] : r = {rate_at_mu:.4f} Hz")
    print(f"Rate at [μ_base+delta = {params.mu_v_with_nmda_unitless:.3f} mV, σ_sol = {solution.sigma:.3f} mV]: r = {rate_at_mu_plus_01:.4f} Hz  ")

    # Second part: plot algorithm result and actual vs desired rate (Taylor)

    plot_rate_and_gain_with_taylor(experiment=palmer_control, params=params, solution=solution, test=test)
    plot_rate_and_gain_with_taylor(experiment=palmer_control, params=params, solution=solution, test=test, lim=3)

class GainScripts(unittest.TestCase):
    """Runnable scripts: gain plots and sigma scans. test_scripts_* = run directly in IntelliJ."""

    def test_scripts_plot_gain(self, params=None):
        """Gain vs μ for several σ; sigma chosen so rate = rate_baseline at mu_nmda_block. Params: rate_nmda_block_recomputed, gain_target_Hz_per_mV, delta_mu_mV, mu_offset_below_theta_mV."""
        if params is None:
            params = default_rate_gain_params()
        p = compute_rate_gain_at_params(palmer_control, params)
        gradients = p.siegert_gradient
        sigma_sol = p.sigma
        gain_target = params.gain_target_Hz_per_mV
        mu_off = params.mu_offset_below_theta_mV
        mus = np.linspace(-65, -35, 1000) * mV

        prepare_bigger_fonts()
        plt.figure(figsize=(15, 8))
        plt.title(rf"Gain $dr/d\mu$ vs $\mu$ - baseline rate={params.rate_nmda_block_recomputed} Hz, target gain={gain_target} Hz/mV, $\Delta\mu$={params.delta_mu_mV} mV")

        for sigma in np.array([0.5, 1, 2, 3, 4]):
            gains = [gradients.d_rate_d_mu(mu_v=mu, sigma_v=sigma * mV) for mu in mus] * (mV / Hz)
            plt.plot(mus / mV, gains, label=r'$\sigma_v$=' + f"{sigma} mV")

        gains_sol = [gradients.d_rate_d_mu(mu_v=mu, sigma_v=sigma_sol) for mu in mus] * (mV / Hz)
        plt.axhline(y=gain_target, linestyle='-.', label=f"Target gain {gain_target} Hz/mV")
        plt.plot(mus / mV, gains_sol, label=r'Sol: $\sigma_v$=' + f"{sigma_sol/mV:.3f} mV, r={p.rate_nmda_block_recomputed:.3f} Hz")

        plt.axvline(x=p.theta / mV - mu_off, linestyle='--', label=rf"$\mu$ = $\theta$ - {mu_off} mV")
        plt.axvline(x=p.theta / mV, linestyle='--', label=r"$\theta$", color='black')
        plt.xlabel(r"Membrane potential $\mu$ (mV)")
        plt.ylabel(r"Gain [Hz/mV]")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        show_plots_non_blocking(caller_test_case=self)

    def test_scripts_plot_LIF_rate_for_interesting_model(self, params=None):
        """Rate vs μ for sigma that gives rate_baseline at mu_nmda_block; show rate at mu_nmda_block + delta_mu (actual vs Taylor)."""
        if params is None:
            params = default_rate_gain_params
        p = compute_rate_gain_at_params(palmer_control, params)
        gradients = p.siegert_gradient
        sigma_sol = p.sigma
        mu_off = params.d_mu
        mus = np.linspace(-65, -49.8, 1000) * mV

        prepare_bigger_fonts()
        plt.figure(figsize=(10, 8))

        for sigma in np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, sigma_sol / mV]):
            rates = np.array([gradients.firing_rate(mu_v=mu, sigma_v=sigma * mV) for mu in mus]) / Hz
            plt.plot(mus / mV, rates, label=r'$\sigma_v$=' + f"{sigma:.3f} mV")

        plt.axhline(y=p.rate_nmda_block_recomputed, linestyle='-.', label=f"Baseline rate {p.rate_nmda_block_recomputed:.3f} Hz")
        #plt.axhline(y=p.rate_taylor_shifted_Hz, linestyle=':', label=rf"Taylor at $\mu+\Delta\mu$ = {p.rate_taylor_shifted_Hz:.3f} Hz")
        plt.axhline(y=p.rate_delta_mu_Hz, linestyle='--', label=rf"Actual at $\mu+\Delta\mu$ = {p.rate_delta_mu_Hz:.3f} Hz (err={p.taylor_error_Hz:.4f})")

        plt.axvline(x=p.theta / mV - mu_off / mV, linestyle='--', label=rf"$\mu$ = $\theta$ - {mu_off / mV} mV")
        plt.axvline(x=p.theta / mV, linestyle='--', label=r"$\theta$", color='black')
        #plt.ylim((0, 10))
        plt.xlabel(r"Membrane potential $\mu$ (mV)")
        plt.ylabel(r"Rate [Hz]")
        plt.title(rf"Rate vs $\mu$ - $\sigma$ s.t. r($\theta$-{mu_off}) = TODO Hz; "
                  rf"at $\mu+{params.d_mu / mV :.3f}$ mV: Taylor=TODO, actual= :.3f, error={p.taylor_error_Hz:.4f} Hz")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        show_plots_non_blocking(caller_test_case=self)



    def test_scripts_plot_for_relaxed_conditions(self):
        # actual numbers from the palmer paper
        plot_rate_and_gain_combined(params=RateGainSearchParams(rate_nmda_block=0.05, rate_with_nmda=0.18, mu_v_nmda_block=-60, mu_v_with_nmda=-60 + 0.4, d_sigma=0.2), test=self)


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

    @unittest.skip("This returns some stupid graph. Not useful")
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

    def test_check_derivative_d_rate_d_mu(self):

        siegert_gradient = SiegertGradients.for_experiment(palmer_control)

        sigma = 3 * mV
        mus = np.linspace(-65, -35, 500) * mV
        delta = 1e-5 * mV  # 1e-3 mV as requested

        analytical = []
        numerical = []
        rel_error = []

        for mu in mus:
            # --- analytical derivative ---
            dr_analytical = siegert_gradient.d_rate_d_mu(
                mu_v=mu,
                sigma_v=sigma
            )

            # --- numerical central difference ---
            r_plus = siegert_gradient.firing_rate(mu_v=mu + delta, sigma_v=sigma)
            r_minus = siegert_gradient.firing_rate(mu_v=mu - delta, sigma_v=sigma)

            dr_numerical = (r_plus - r_minus) / (2 * delta)

            # convert to Hz/mV (pure float)
            dr_a = float(dr_analytical * mV / Hz)
            dr_n = float(dr_numerical * mV / Hz)

            analytical.append(dr_a)
            numerical.append(dr_n)

            if abs(dr_n) > 1e-12:
                rel_error.append((dr_a - dr_n) / dr_n)
            else:
                rel_error.append(0.0)

        analytical = np.array(analytical)
        numerical = np.array(numerical)
        rel_error = np.array(rel_error)

        # --- plotting ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

        ax1.plot(mus / mV, analytical, label="Analytical", linewidth=2)
        ax1.plot(mus / mV, numerical, "--", label=rf"Numerical ($\Delta$={delta})")
        ax1.set_ylabel("Gain [Hz/mV]")
        ax1.set_title(r"Derivative check for $\sigma = 3$ mV")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(mus / mV, 100 * rel_error)
        ax2.set_ylabel("Relative error [%]")
        ax2.set_xlabel(r"$\mu$ (mV)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("Max abs relative error (%):", np.max(np.abs(100 * rel_error)))

    def test_check_derivative_d_I_d_mu(self):

        experiment = palmer_control
        siegert_gradient = SiegertGradients.for_experiment(experiment)

        sigma = 3 * mV
        mus = np.linspace(-65, -35, 500) * mV
        delta = 1e-5 * mV  # 1e-3 mV as requested

        analytical = []
        numerical = []
        rel_error = []

        for mu in mus:
            # --- analytical derivative ---
            dr_analytical = siegert_gradient.grad_I(
                mu_v=mu,
                sigma_v=sigma
            )[0]

            # --- numerical central difference ---
            i_plus = I_mu_sigma(mu_v=mu + delta, sigma_v=sigma, V_reset=experiment.neuron_params.V_r, theta=experiment.neuron_params.theta)
            i_minus = I_mu_sigma(mu_v=mu - delta, sigma_v=sigma, V_reset=experiment.neuron_params.V_r, theta=experiment.neuron_params.theta)

            dr_numerical = (i_plus - i_minus) / (2 * delta)

            # convert to Hz/mV (pure float)
            dr_a = float(dr_analytical * mV / Hz)
            dr_n = float(dr_numerical * mV / Hz)

            analytical.append(dr_a)
            numerical.append(dr_n)

            if abs(dr_n) > 1e-12:
                rel_error.append((dr_a - dr_n) / dr_n)
            else:
                rel_error.append(0.0)

        analytical = np.array(analytical)
        numerical = np.array(numerical)
        rel_error = np.array(rel_error)

        # --- plotting ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

        ax1.plot(mus / mV, analytical, label="Analytical", linewidth=2)
        ax1.plot(mus / mV, numerical, "--", label=rf"Numerical ($\Delta$={delta})")
        ax1.set_ylabel(r"I($\mu, \sigma$)")
        ax1.set_title(r"Derivative check for $\sigma = 3$ mV")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(mus / mV, 100 * rel_error)
        ax2.set_ylabel("Relative error [%]")
        ax2.set_xlabel(r"$\mu$ (mV)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("Max abs relative error (%):", np.max(np.abs(100 * rel_error)))



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
