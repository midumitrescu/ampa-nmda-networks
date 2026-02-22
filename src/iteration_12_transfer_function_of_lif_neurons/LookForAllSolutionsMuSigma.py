import sys
from loguru import logger
logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from _pytest import unittest
from brian2 import mV, Hz
from joblib import Parallel, delayed
from scipy.stats import stats

from BinarySeach import binary_search_for_target_value_precission_in_result_space
from build.lib.src.Plotting import NeuronModelParams
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradientDescent, \
    rate_LIF_whitenoise
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
                                   resolution=50, target_rate_Hz=None):
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
    plt.show()

    return fig, (ax1, ax2)


class MyTestCase(unittest.TestCase):

    def test_look_for_all_solutions(self):
        experiment = palmer_control

        r_target = 0.3 * Hz
        solver = SiegertGradientDescent(tau_m=experiment.effective_time_constant_up_state.tau_eff(),
                                        theta=experiment.neuron_params.theta,
                                        v_reset=experiment.neuron_params.V_r,
                                        tau_ref=experiment.neuron_params.tau_rp, unit='mV')

        # Initial guesses (in mV) - adjusted for normalized form
        mu_0, sigma_0 = -56 * mV, 2.5 * mV

        fig, axes = plot_loss_landscape_with_curve(
            solver, r_target, (-60, -35), (0, 6),
            resolution=50, target_rate_Hz=0.3
        )

        fig.show()

    def test_look_for_all_solutions_using_binary_search(self):
        experiment = palmer_control
        palmer_control.with_property(NeuronModelParams.KEY_NEURON_V_R, -65)

        mu_s_nmda_block, sigmas_nmda_block = self.compute_mu_to_sigma_curve(experiment, r_target = 0.05 * Hz)
        mu_s_control, sigmas_control = self.compute_mu_to_sigma_curve(experiment, r_target = 0.3 * Hz)

        plt.plot(mu_s_nmda_block / mV, sigmas_nmda_block / mV, label="NMDA Block r = 0.05 Hz")
        plt.plot(mu_s_control / mV, sigmas_control / mV, label="Control r = 0.3 Hz")

        plt.xlabel("$\mu_v$ [mV]")
        plt.ylabel("$\sigma_v [mV]^2$")
        plt.title("$\mu$ vs $\sigma_v$ dependency for constant firing rate "
                  "predicted by first time passage formula")

        plt.tight_layout()
        plt.legend()
        plt.show()

        slope_nmda, intercept_nmda, r_value_nmda, p_value_nmda, std_err_nmda = \
            stats.linregress(mu_s_nmda_block / mV, sigmas_nmda_block / mV)

        slope_ctrl, intercept_ctrl, r_value_ctrl, p_value_ctrl, std_err_ctrl = \
            stats.linregress(mu_s_control / mV, sigmas_control / mV)

        print(f"NMDA Block (0.05 Hz): σ = {slope_nmda:.4f}·μ + {intercept_nmda:.4f}")
        print(f"Control (0.3 Hz): σ = {slope_ctrl:.4f}·μ + {intercept_ctrl:.4f}")

        # Check your data first
        print("NMDA Block data:")
        print(f"mu_s_nmda_block type: {type(mu_s_nmda_block)}")
        print(f"mu_s_nmda_block shape: {np.shape(mu_s_nmda_block)}")
        print(f"mu_s_nmda_block values: {mu_s_nmda_block}")
        print(f"mu_s_nmda_block / mV: {mu_s_nmda_block / mV}")

        print("\nControl data:")
        print(f"mu_s_control shape: {np.shape(mu_s_control)}")
        print(f"mu_s_control values: {mu_s_control}")

        # Check for NaN or Inf
        print(f"\nAny NaN in NMDA mu? {np.any(np.isnan(mu_s_nmda_block / mV))}")
        print(f"Any NaN in NMDA sigma? {np.any(np.isnan(sigmas_nmda_block / mV))}")
        print(f"Any Inf in NMDA mu? {np.any(np.isinf(mu_s_nmda_block / mV))}")

    def compute_mu_to_sigma_curve(self, experiment, r_target):

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
        mu_s = np.linspace(experiment.neuron_params.E_leak, max_mu, num=1000)

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
        sigmas = np.array(sigmas)
        return mu_s, sigmas

    def test_look_for_N_nmda_nu_nmda_such_that_d_rate_desired_value(self):


        pass

