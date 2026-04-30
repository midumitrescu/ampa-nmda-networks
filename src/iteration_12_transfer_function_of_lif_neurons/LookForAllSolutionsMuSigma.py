import sys

from loguru import logger
from scipy.optimize import fsolve
import numpy as np

from Plotting import prepare_bigger_fonts
from iteration_12_transfer_function_of_lif_neurons.config import DiffusionLIFConfig, default_diffusion_lif_config
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from src.Plotting import show_plots_non_blocking

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from brian2 import mV, Hz, Quantity, volt, mvolt, is_dimensionless
from joblib import Parallel, delayed
from scipy.stats import stats

from BinarySeach import binary_search_for_target_value_precission_in_result_space
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import rate_LIF_whitenoise, SiegertGradients, newton_fsolve_find_mu_for_fixed_sigma


def compute_sigma_necessary_for_given_rate_and_mean(mu, r_target, lif_config: DiffusionLIFConfig):
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

def compute_sigma_necessary_for_given_rate_derivative_and_mean(mu, target_gain, lif_config: DiffusionLIFConfig):
    """
    Find sigma for a given mu using binary search to hit r_t
    """
    sg = SiegertGradients.for_lif_config(lif_config)
    look_for_sigma = lambda s: sg.d_rate_d_mu(mu, s)
    try:
        sigma, _ = binary_search_for_target_value_precission_in_result_space(
            lower_value=0.1 * mV,
            upper_value=20 * mV,
            func=look_for_sigma,
            target_result=target_gain,
            precision=1e-10 * Hz / mV,
            max_iters=100
        )
        return sigma
    except ValueError as e:
        print(f"mu={mu}: {e}")
        return np.nan  # fallback if binary search fails

def binary_search_sigma_at_mu_for_firing_rate(mu, r_target, lif_config: DiffusionLIFConfig):
    return compute_sigma_necessary_for_given_rate_and_mean(mu, r_target, lif_config)

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

    ax1.set_xlabel(r"$\mu$"' [mV]')
    ax1.set_ylabel(r"$\sigma$"' [mV]')
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

    def __init__(self, mus, sigmas, r_target, delta_mu=0 * mV, exp_label=""):
        self.mus = mus / mV
        self.sigmas = sigmas / mV
        self.r_target = r_target
        self.exp_label = exp_label
        self.delta_mu = delta_mu

    def mus_to_sigmas(self):
        return zip(self.mus, self.sigmas)

    def linear_fit(self):
        return stats.linregress(self.mus, self.sigmas)

    def __str__(self):
        return f"{self.__class__}, {self.exp_label}, mus = {len(self.mus)}, sigmas = {len(self.sigmas)}"

    def with_delta_mu(self, delta_mu: Quantity):
        new_result = MuToSigmaResult(mus=self.mus * mV - delta_mu,
                                                    sigmas=self.sigmas * mV,
                                                    r_target=self.r_target,
                                                    exp_label=self.exp_label)
        new_result.delta_mu = delta_mu
        return new_result

    def firing_rates_no_units(self, sg: SiegertGradients):
        return np.array([sg.firing_rate(mu * mV, sigma * mV) / Hz for mu, sigma in zip(self.mus, self.sigmas)])

    def firing_rates(self, sg: SiegertGradients):
        return self.firing_rates_no_units(sg) * Hz

    def d_rate_d_mus_no_units(self, sg: SiegertGradients):
        return np.array([sg.d_rate_d_mu(mu * mV, sigma * mV) / Hz * mV for mu, sigma in zip(self.mus, self.sigmas)])

    def d_rate_d_mus(self, sg: SiegertGradients):
        return self.d_rate_d_mus_no_units(sg) * Hz / mV


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
    return mu_to_sigma_for_constant_rate(lif_config=DiffusionLIFConfig.from_experiment(experiment), r_target=r_target)


def mu_to_sigma_for_constant_rate(lif_config: DiffusionLIFConfig, r_target: Quantity, mu_lims=None):

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
    mu_s = get_mu_linspace(lif_config=lif_config, mu_lims=mu_lims)

    # Run in parallel on all mu values
    sigmas = Parallel(n_jobs=-1, backend="loky")(
        delayed(compute_sigma_necessary_for_given_rate_and_mean)(mu, r_target, lif_config) for mu in mu_s
    )
    # Convert to numpy array
    logger.debug("sigma[0] = {}", sigmas[0])
    sigmas = np.array(sigmas/mV) * mV
    logger.debug("sigma[0] = {}. Attention! np.array removes units! This is why I need to re-add units!! Otherwise, bug", sigmas[0])
    return MuToSigmaResult(mus=mu_s, sigmas=sigmas, r_target=r_target, exp_label=lif_config.label)

def mu_to_sigma_for_constant_gain(lif_config: DiffusionLIFConfig, gain: Quantity, mu_lims= None):
    mu_s = get_mu_linspace(lif_config, mu_lims)
    sigmas = Parallel(n_jobs=-1, backend="loky")(
        delayed(compute_sigma_necessary_for_given_rate_derivative_and_mean)(mu, gain, lif_config) for mu in mu_s
    )
    # Convert to numpy array
    sigmas = np.array(sigmas)
    mask = np.isfinite(sigmas)
    sigmas = np.array(sigmas[mask]) * volt
    mu_s = mu_s[mask]
    logger.debug(
        "sigma[0] = {}. Attention! np.array removes units! This is why I need to re-add units!! Otherwise, bug",
        sigmas[0])
    return MuToSigmaResult(mus=mu_s, sigmas=sigmas, r_target=gain, exp_label=lif_config.label)


def get_mu_linspace(lif_config, mu_lims):
    if mu_lims is None:
        return np.linspace(lif_config.theta - 20 * mV, lif_config.theta - 0.01 * mV, num=1001)

    if is_dimensionless(mu_lims[0]):
            mu_lims = (mu_lims[0] * mV, mu_lims[1] * mV)

    return np.linspace(mu_lims[0], mu_lims[1], num=1001)

def plot_line_computation_vs_fit(results: list[MuToSigmaResult], caller_test_case=None, descriptor="linear_fit", axs=None, colors = ("orange", "black"),
                                 config: DiffusionLIFConfig = default_diffusion_lif_config):

    should_create_figure = axs is None

    prepare_bigger_fonts(zoom=1)

    if should_create_figure:
        fig = plt.figure(figsize=(10, 16))

        # Outer grid: 2 rows
        outer_gs = fig.add_gridspec(
            nrows=2, ncols=1,
            height_ratios=[2, 3],
            hspace=0.15
        )

        # --- Top: linear fit only ---
        ax_linear_fit = fig.add_subplot(outer_gs[0])

        # --- Bottom: grouped table + residuals ---
        inner_gs_bottom = outer_gs[1].subgridspec(
            nrows=2, ncols=1,
            height_ratios=[1.5, 2],
            hspace=0.05
        )

        ax_table = fig.add_subplot(inner_gs_bottom[0])
        ax_table.axis('off')

        ax_residuals = fig.add_subplot(inner_gs_bottom[1], sharex=ax_linear_fit)


    r2_s = np.zeros_like(results)
    rows = []

    rmse_s = np.zeros_like(results)
    mae_s = np.zeros_like(results)

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
        rmse_s[index] = np.sqrt(np.mean((y - y_pred)**2))
        mae_s[index] = np.mean(np.abs(y - y_pred))

        ax_linear_fit.plot(x, y, alpha=0.5, label=f'{result.exp_label} data', linewidth=10, color=color)

        # Plot fitted lines across a common range
        x_plot = np.linspace(-65, -35, 100)
        y_plot = slope * x_plot + intercept

        ax_linear_fit.plot(x_plot, y_plot, linewidth=3.5,
                           label=f'{result.exp_label} linear fit: \n $\sigma_v$={slope:.3f}$\mu${intercept:.2f}', color=color)

        ax_residuals.scatter(x, residuals, color=color, s=5, alpha=0.5, label=f'{result.exp_label}')
        ax_residuals.axhline(0, color='black', lw=1, linestyle='--')


        residuals = y - y_pred
        max_error = np.max(np.abs(residuals))
        e_infinity = max_error / (np.max(y) - np.min(y))
        rows.append([result.exp_label, f"{max_error:.4f}", f"{e_infinity:.4f}"])

    ## labels and titles for ax linear fit ##
    labels = [r.exp_label for r in results]
    if len(labels) == 1:
        rmse_label = f"{labels[0]}: {rmse_s[0] :.4f}"
        mae_label = f"{labels[0]}: {mae_s[0]:.4f}"
    elif len(labels) == 2:
        rmse_label = f"{labels[0]}: {rmse_s[0] :.4f} and {labels[1]}: {rmse_s[1]:.4f}"
        mae_label = f"{labels[0]}: {mae_s[0] :.4f} and {labels[1]}: {mae_s[1]:.4f}"
    else:
        rmse_labels =  [f"{label}: {rmse:.4f}" for label, rmse in zip(labels, rmse_s)]
        rmse_label = f"{', '.join(rmse_labels[:-1])} and {rmse_labels[-1]}"
        mae_labels =  [f"{label}: {mae:.4f}" for label, mae in zip(labels, mae_s)]
        mae_label =  f"{', '.join(mae_labels[:-1])} and {mae_labels[-1]}"

    ax_linear_fit.set_xlabel('$\mu_v$ [mV]')
    ax_linear_fit.set_ylabel('$\sigma_v$ [mV]')

    ax_linear_fit.text(
        0.5, 1.3, f'Verify errors of linear fits \n Root Mean Square Error [mV] \n {rmse_label} \n Mean Absolute Error [mV] \n {mae_label}',
        ha='center', va='top',
        transform=ax_linear_fit.transAxes,
        fontsize=18,
        clip_on = False
    )

    ## labels and title for table ##

    col_labels = ["", r"$E_{\max}$" + "\n" + "$\max |\mathrm{err}|$ [mV]", r"$E_{\infty}$" + "\n" +  r"$\frac{E_{\max}}{\max |\mathrm{err}| - \min |\mathrm{err}|}$"]
    table_title = r"Error ($\sigma_{v, i, \mathrm{found}} - \sigma_{v, i, \mathrm{linear\ est}}$) of linear fit" + "\n" + r"$\mathrm{err}_i = \sigma_{v, i} - \hat{\sigma}_{v, i}$"
    ax_table.text(
        0.5, 0.8, table_title,
        ha='center', va='bottom'
    )

    table =ax_table.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        bbox=[0.1, 0, 0.8, 0.8]
    )

    table.scale(0.8, 3)  # adjust size

    table.auto_set_font_size(False)
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # header row
            cell.set_height(0.6)
            cell.get_text().set_fontsize(cell.get_text().get_fontsize() + 2)
            cell.set_text_props(weight='bold')
        else:  # body rows
            cell.set_height(0.2)

    ## labels for residuals axis ##
    ax_residuals.set_xlabel('$\mu_v$ [mV]')
    ax_residuals.set_ylabel('Error [mV]')

    for index, ax in enumerate([ax_linear_fit, ax_residuals]):
        ax.axvline(x=default_diffusion_lif_config.theta / mV, color='dimgray', linestyle='-.',
                   label=r'Threshold $\theta$')
        ax.text(
            0.02, 1.1, f"({chr(ord("A") + index)})",
            transform=ax.transAxes,
            fontsize=20,
            fontweight=1000,
            va="top",
            ha="left"
        )
    ax_linear_fit.legend(
        loc='upper right',
        bbox_to_anchor=(1.1, 1),
        borderaxespad=0.,
        fontsize=16,
    )
    ax_residuals.legend()

    print(len(ax_linear_fit.texts))

    if should_create_figure:
        fig.tight_layout()
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