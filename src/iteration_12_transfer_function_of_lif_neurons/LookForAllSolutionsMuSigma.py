import sys

from loguru import logger
from scipy.optimize import fsolve

from iteration_12_transfer_function_of_lif_neurons.config import DiffusionLIFConfig, default_diffusion_lif_config
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from src.Plotting import show_plots_non_blocking

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from brian2 import mV, Hz, Quantity, volt
from joblib import Parallel, delayed
from scipy.stats import stats

from BinarySeach import binary_search_for_target_value_precission_in_result_space
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import rate_LIF_whitenoise, SiegertGradients, newton_fsolve_find_mu_for_fixed_sigma


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

def compute_mu_to_sigma_curve_for_experiment(experiment: Experiment, r_target: Quantity):
    return compute_mu_to_sigma_curve(lif_config=DiffusionLIFConfig.from_experiment(experiment), r_target=r_target)


def compute_mu_to_sigma_curve(lif_config: DiffusionLIFConfig, r_target: Quantity):

    # scan over mu, keep sigma
    # first, look for max mu i.e. the mu for zero sigma that returns r_target
    look_for_mu = lambda mu: rate_LIF_whitenoise(mu,
                                                 tau_membrane=lif_config.tau_m,
                                                 sigma_v=0 * mV, theta=lif_config.theta,
                                                 V_reset=lif_config.V_r,
                                                 tau_ref=lif_config.tau_rp)
    _, max_mu = binary_search_for_target_value_precission_in_result_space(-55 * mV, upper_value=-35 * mV,
                                                                          func=look_for_mu, target_result=r_target,
                                                                          precision=1E-10 * Hz, max_iters=100)

    # Prepare mu values
    mu_s = np.linspace(lif_config.theta - 20 * mV, max_mu-0.01*mV, num=1000)

    def compute_sigma(mu, r_target, lif_config: DiffusionLIFConfig):
        """
        Find sigma for a given mu using binary search to hit r_target.
        """
        look_for_sigma = lambda s: rate_LIF_whitenoise(
            mu,
            tau_membrane=lif_config.tau_m,
            sigma_v=s,
            theta=lif_config.theta,
            V_reset=lif_config.V_r,
            tau_ref=lif_config.tau_rp
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
        delayed(compute_sigma)(mu, r_target, lif_config) for mu in mu_s
    )
    # Convert to numpy array
    logger.debug("sigma[0] = {}", sigmas[0])
    sigmas = np.array(sigmas/mV) * mV
    logger.debug("sigma[0] = {}. Attention! np.array removes units! This is why I need to re-add units!! Otherwise, bug", sigmas[0])
    return MuToSigmaResult(mus=mu_s, sigmas=sigmas, r_target=r_target, exp_label=lif_config.label)


def plot_line_computation_vs_fit(results: list[MuToSigmaResult], caller_test_case=None, descriptor="linear_fit", axs=None, colors = ("orange", "black"),
                                 config: DiffusionLIFConfig = default_diffusion_lif_config):

    should_create_figure = axs is None
    if should_create_figure:
        fig2, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    r2_s = np.zeros_like(results)

    for index, (result, color) in enumerate(zip(results, colors)):

        slope, intercept, r_value, p_value, std_err = \
            stats.linregress(result.mus, result.sigmas)

        x = result.mus
        y = result.sigmas

        # Calculate R² to show goodness of fit
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        residuals = y - y_pred

        r2_s[index] = r2
        axs[0].plot(x, y, alpha=0.5, label=f'{result.exp_label} data', linewidth=10, color=color)

        # Plot fitted lines across a common range
        x_plot = np.linspace(-65, -35, 100)
        y_plot = slope * x_plot + intercept

        axs[0].plot(x_plot, y_plot, linewidth=2,
                label=f'{result.exp_label} linear fit: \n $\sigma_v$={slope:.3f}$\mu${intercept:.2f}', color=color)

        axs[1].scatter(x, residuals, color=color, s=5, alpha=0.5, label=f'{result.exp_label}')
        axs[1].axhline(0, color='black', lw=1, linestyle='--')
        axs[1].set_xlabel('$mu_v$')
        axs[1].set_ylabel('Residuals')

        residuals = y - y_pred
        max_error = np.max(np.abs(residuals))
        e_infinity = max_error / (np.max(y) - np.min(y))
        subtitle = (
                r"$\mathrm{res}_i = \sigma_{v, i} - \hat{\sigma}_{v, i}$" + "\n" +
                rf"$E_{{\max}} = \max_i |\mathrm{{res}}_i| = {max_error:.4f}$" + "\n" +
                rf"$E_{{\infty}} = \frac{{E_{{\max}}}}{{\max_i \sigma_{{v, i}} - \min_i \sigma_{{v, i}}}} = {e_infinity:.4f}$"
        )

        axs[1].set_title(
            fr"Plot of residual error ($\sigma_{{v, i, \mathrm{{found}}}} - \sigma_{{v, i, \mathrm{{linear\ est}}}}$) of linear fit"
            + "\n" + subtitle
        )

    labels = [r.exp_label for r in results]
    if len(labels) == 1:
        cond = f"{labels[0]} condition"
        r2_label = f"{labels[0]}: {r2_s[0]}"
    elif len(labels) == 2:
        cond = f"{labels[0]} and {labels[1]} conditions"
        r2_label = f"{labels[0]}: {r2_s[0]} and {labels[1]}: {r2_s[1]}"
    else:
        cond = f"{', '.join(labels[:-1])} and {labels[-1]} conditions"
        r2_labels = [f"{label}: {r_2:.4f}" for label, r_2 in zip(labels, r2_s)]
        r2_label = f"{', '.join(r2_labels[:-1])} and {r2_labels[-1]}"

    axs[0].set_xlabel('$\mu_v$ (mV)', fontsize=12)
    axs[0].set_ylabel('$\sigma_v$ (mV)', fontsize=12)

    axs[0].set_title(f'Verify linear fits for {cond} \n $R^2$ values: {r2_label}', fontsize=14)

    for index, ax in enumerate(axs):
        ax.axvline(x=default_diffusion_lif_config.theta / mV, color='dimgray', linestyle='-.',
                   label=r'Threshold $\theta$')
        ax.text(
            0.02, 1.17, f"({chr(ord("A") + index)})",
            transform=ax.transAxes,
            fontsize=20,
            fontweight=1000,
            va="top",
            ha="left"
        )
        ax.legend(fontsize=10)

    if should_create_figure:
        fig2.tight_layout()
        show_plots_non_blocking(caller_test_case=caller_test_case, descriptor=descriptor)


def plot_mus_vs_sigmas(results: list[MuToSigmaResult], caller_test_case=None):
    """Plot μ vs σ curves and linear fit. Used by script runners with caller_test_case=self for figure naming."""
    for result, color in zip(results, ["orange", "black"]):
        plt.plot(result.mus, result.sigmas, color=color, label=f"{result.exp_label}, r = {result.r_target / Hz} Hz", lw=2)
    plt.xlabel(r"$\mu_v$ [mV]")
    plt.ylabel(r"$\sigma_v$ [mV]")
    plt.title(r"$\mu$ vs $\sigma_v$ dependency for constant firing rate "
              "predicted by first time passage formula")
    plt.tight_layout()
    plt.legend()
    show_plots_non_blocking(caller_test_case=caller_test_case)
    plot_line_computation_vs_fit(results, caller_test_case=caller_test_case, descriptor="linear_fit")