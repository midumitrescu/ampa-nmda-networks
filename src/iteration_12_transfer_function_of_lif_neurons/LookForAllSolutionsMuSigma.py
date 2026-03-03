import sys

from loguru import logger
from scipy.optimize import fsolve

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from src.Plotting import show_plots_non_blocking

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from brian2 import mV, Hz, Quantity, volt, mvolt
from joblib import Parallel, delayed
from scipy.stats import stats

from BinarySeach import binary_search_for_target_value_precission_in_result_space
from build.lib.src.Plotting import NeuronModelParams
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradientDescent, \
    rate_LIF_whitenoise, SiegertGradients, newton_fsolve_find_mu_for_fixed_sigma
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control


def find_curve_grid(solver, r_target, mu_range, sigma_range, resolution=100):
    """
    Find curve by contouring the error surface
    """
    mu_grid = np.linspace(mu_range[0], mu_range[1], resolution)
    sigma_grid = np.linspace(sigma_range[0], sigma_range[1], resolution)
    Mu, Sigma = np.meshgrid(mu_grid, sigma_grid)

    # Compute F - r_target on grid
    F_grid = np.zeros_like(Mu)
    for i in range(resolution):
        for j in range(resolution):
            F_grid[j, i] = solver.firing_rate(Mu[j, i] * mV, Sigma[j, i] * mV) / Hz

    # Find contour at r_target
    import matplotlib.pyplot as plt
    plt.contour(Mu, Sigma, F_grid, levels=[r_target / Hz], colors='r')

    # Extract contour points
    from skimage import measure
    contours = measure.find_contours(F_grid, r_target / Hz)

    return contours


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_loss_landscape_with_curve(solver, r_target, mu_range, sigma_range,
                                   resolution=50, target_rate_Hz=None, caller_test_case=None):
    """
    Plot 3D loss landscape with the solution curve F(μ,σ) = r_target

    Parameters:
    -----------
    solver : your SiegertGradientDescent instance
    r_target : Brian2 quantity, target firing rate
    mu_range : tuple (min, max) in mV
    sigma_range : tuple (min, max) in mV
    resolution : int, grid resolution
    target_rate_Hz : float, if provided, use this for contour level
    """
    # Create grid
    mu_grid = np.linspace(mu_range[0], mu_range[1], resolution)
    sigma_grid = np.linspace(sigma_range[0], sigma_range[1], resolution)
    Mu, Sigma = np.meshgrid(mu_grid, sigma_grid)

    # Compute loss on grid
    loss_grid = np.zeros_like(Mu)
    rate_grid = np.zeros_like(Mu)

    print("Computing loss landscape...")
    for i in range(resolution):
        for j in range(resolution):
            # Get firing rate
            F = solver.firing_rate(Mu[j, i] * mV, Sigma[j, i] * mV)
            rate_grid[j, i] = F / Hz

            # Compute loss
            loss_grid[j, i] = 0.5 * ((F - r_target) / Hz) ** 2

    # Target rate in Hz for plotting
    if target_rate_Hz is None:
        target_rate_Hz = r_target / Hz

    # Create 3D plot
    fig = plt.figure(figsize=(16, 8))

    # Plot 1: 3D surface with loss
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot loss surface with colormap
    surf = ax1.plot_surface(Mu, Sigma, loss_grid,
                            cmap=cm.viridis, alpha=0.8,
                            linewidth=0, antialiased=True)

    # Find and plot the solution curve (where loss is minimal)
    # This is where rate_grid ≈ target_rate_Hz
    tolerance = 0.05 * target_rate_Hz  # 5% tolerance
    mask = np.abs(rate_grid - target_rate_Hz) < tolerance

    # Extract curve points
    curve_mu = Mu[mask]
    curve_sigma = Sigma[mask]
    curve_loss = loss_grid[mask]

    # Plot the curve in red
    ax1.scatter(curve_mu, curve_sigma, curve_loss,
                c='red', s=20, alpha=0.8, label=f'F = {target_rate_Hz} Hz')

    ax1.set_xlabel(r"$\mu$"' (mV)')
    ax1.set_ylabel(r"$\sigma$"' (mV)')
    ax1.set_zlabel('Loss')
    ax1.set_title(f'3D Loss Landscape with Solution Curve\nTarget Rate = {target_rate_Hz} Hz')
    ax1.legend()

    # Add colorbar
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Loss')

    # Plot 2: 2D contour with the curve
    ax2 = fig.add_subplot(122)

    # Contour plot of loss
    contour = ax2.contourf(Mu, Sigma, loss_grid, levels=20, cmap=cm.viridis, alpha=0.8)
    plt.colorbar(contour, ax=ax2, label='Loss')

    # Plot the solution curve (red)
    ax2.contour(Mu, Sigma, rate_grid, levels=[target_rate_Hz],
                colors='red', linewidths=3, linestyles='-')

    # Optional: add gradient descent path if you have history
    # if 'history' in locals():
    #     ax2.plot(history['mu'], history['sigma'], 'w-o', linewidth=2, markersize=4, label='GD path')

    ax2.set_xlabel(r"$\mu$"' (mV)')
    ax2.set_ylabel(r"$\sigma$" ' (mV)')
    ax2.set_title(f'Loss Contour with Solution Curve\nF = {target_rate_Hz} Hz')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    show_plots_non_blocking(caller_test_case=caller_test_case, descriptor="loss_landscape")

    return fig, (ax1, ax2)

class MuToSigmaResult:

    def __init__(self, mus, sigmas, r_target, exp_label=""):
        self.mus = mus / mV
        self.sigmas = sigmas / mV
        self.r_target = r_target
        self.exp_label = exp_label

def newton_fsolve_find_sigma_for_fixed_mu(siegert_gradient: SiegertGradients, mu_v: Quantity, r_target: Quantity):
    return fsolve(func=lambda sigma: [siegert_gradient.firing_rate(mu_v=mu_v, sigma_v=sigma[0] * volt) - r_target],
                  x0=-55 * mV,
                  fprime=lambda sigma: [siegert_gradient.d_rate_d_mu(mu_v=mu_v, sigma_v=sigma[0] * volt)])[0] * volt

def compute_mu_to_sigma_fsolve_scan_mus(experiment: Experiment, r_target):
    mus = np.linspace(experiment.neuron_params.theta - 20*mV, experiment.neuron_params.theta - 0.1 * mV, 1001)
    siegert_gradient = SiegertGradients.for_experiment(experiment)

    sigmas = np.zeros_like(mus)
    for index, mu in enumerate(mus):
        sigmas[index] = newton_fsolve_find_sigma_for_fixed_mu(siegert_gradient=siegert_gradient, mu_v = mu, r_target=r_target)

    return MuToSigmaResult(mus, sigmas, r_target)

def compute_mu_to_sigma_fsolve_scan_sigmas(experiment, r_target):
    sigmas = np.linspace(0, 10, 101) * mV
    siegert_gradient = SiegertGradients.for_experiment(experiment)

    mus = np.zeros_like(sigmas)
    for index, sigma_v in enumerate(sigmas):
        mus[index] = newton_fsolve_find_mu_for_fixed_sigma(siegert_gradient=siegert_gradient, sigma_v=sigma_v,
                                                     r_target=r_target)

    return MuToSigmaResult(mus, sigmas, r_target)

def compute_mu_to_sigma_curve(experiment, r_target):

    # scan over mu, keep sigma
    # first, look for max mu i.e. the mu for zero sigma that returns r_target
    look_for_mu = lambda mu: rate_LIF_whitenoise(mu,
                                                 tau_membrane=experiment.effective_time_constant_up_state.tau_eff(),
                                                 sigma_v=0 * mV, theta=experiment.neuron_params.theta,
                                                 V_reset=experiment.neuron_params.V_r,
                                                 tau_ref=experiment.neuron_params.tau_rp)
    _, max_mu = binary_search_for_target_value_precission_in_result_space(-55 * mV, upper_value=-35 * mV,
                                                                          func=look_for_mu, target_result=r_target,
                                                                          precision=1E-10 * Hz, max_iters=100)
    # Prepare mu values
    mu_s = np.linspace(experiment.neuron_params.E_leak, max_mu-0.1*mV, num=1000)

    def compute_sigma(mu, r_target, experiment):
        """
        Find sigma for a given mu using binary search to hit r_target.
        """
        look_for_sigma = lambda s: rate_LIF_whitenoise(
            mu,
            tau_membrane=experiment.effective_time_constant_up_state.tau_eff(),
            sigma_v=s,
            theta=experiment.neuron_params.theta,
            V_reset=experiment.neuron_params.V_r,
            tau_ref=experiment.neuron_params.tau_rp
        )
        try:
            sigma, _ = binary_search_for_target_value_precission_in_result_space(
                lower_value=0 * mV,
                upper_value=20 * mV,
                func=look_for_sigma,
                target_result=r_target,
                precision=1e-10 * Hz,
                max_iters=100
            )
            return sigma
        except ValueError as e:
            print(f"mu={mu}: {e}")
            return np.nan  # fallback if binary search fails

    # Run in parallel on all mu values
    sigmas = Parallel(n_jobs=-1, backend="loky")(
        delayed(compute_sigma)(mu, r_target, experiment) for mu in mu_s
    )
    # Convert to numpy array
    logger.debug("sigma[0] = {}", sigmas[0])
    sigmas = np.array(sigmas/mV) * mV
    logger.debug("sigma[0] = {}. Attention! np.array removes units! This is why I need to re-add units!! Otherwise, bug", sigmas[0])
    return MuToSigmaResult(mus=mu_s, sigmas=sigmas, r_target=r_target, exp_label=experiment.plot_params.panel)


def plot_line_computation_vs_fit(results: list[MuToSigmaResult], caller_test_case=None, descriptor="linear_fit"):

    fig2, ax = plt.subplots(figsize=(10, 6))
    for result in results:

        slope, intercept, r_value, p_value, std_err = \
            stats.linregress(result.mus, result.sigmas)

        print(f"{result.exp_label} ({result.r_target}): σ = {slope:.10f}·μ + {intercept:.10f}")

        x = result.mus
        y = result.sigmas

        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = slope * x_fit + intercept
        # Calculate R² to show goodness of fit
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        ax.plot(x, y, alpha=0.5, label=f'{result.exp_label} data', linewidth=10)

        # Plot fitted lines across a common range
        x_plot = np.linspace(-70, -45, 100)
        y_plot = slope * x_plot + intercept

        ax.plot(x_plot, y_plot, linewidth=2,
                label=f'{result.exp_label}: $\sigma$={slope:.3f}$\mu${intercept:.2f}')

        print("=" * 60)
        print("LINEAR FIT RESULTS")
        print("=" * 60)
        print(f"{result.exp_label} ({result.r_target}):")
        print(f"  σ = {slope:.4f}·μ + {intercept:.4f}")
        print(f"  R² = {r2:.6f}")
        print(f"  Number of points: {len(x)}")
        print()

    ax.set_xlabel('$\mu$ (mV)', fontsize=12)
    ax.set_ylabel('$\sigma$ (mV)', fontsize=12)
    ax.set_title('Comparison of Linear Fits for Both Conditions', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    show_plots_non_blocking(caller_test_case=caller_test_case, descriptor=descriptor)


def plot_mus_vs_sigmas(results: list[MuToSigmaResult], caller_test_case=None):
    """Plot μ vs σ curves and linear fit. Used by script runners with caller_test_case=self for figure naming."""
    for result in results:
        plt.plot(result.mus, result.sigmas, label=f"{result.exp_label}, r = {result.r_target}")
    plt.xlabel(r"$\mu_v$ [mV]")
    plt.ylabel(r"$\sigma_v$ [mV]")
    plt.title(r"$\mu$ vs $\sigma_v$ dependency for constant firing rate "
              "predicted by first time passage formula")
    plt.tight_layout()
    plt.legend()
    show_plots_non_blocking(caller_test_case=caller_test_case)
    plot_line_computation_vs_fit(results, caller_test_case=caller_test_case, descriptor="linear_fit")