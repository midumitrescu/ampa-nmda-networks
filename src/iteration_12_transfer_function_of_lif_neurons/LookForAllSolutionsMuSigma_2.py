from _pytest import unittest
from brian2 import mV, Hz

from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradientDescent
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


def plot_loss_landscape_3d_with_valley(solver, r_target, mu_range, sigma_range,
                                       resolution=50):
    """
    3D plot with log scaling and clipping to reveal the solution valley
    """
    # Create grid
    mu_grid = np.linspace(mu_range[0], mu_range[1], resolution)
    sigma_grid = np.linspace(sigma_range[0], sigma_range[1], resolution)
    Mu, Sigma = np.meshgrid(mu_grid, sigma_grid)

    # Compute loss
    loss_grid = np.zeros_like(Mu)
    rate_grid = np.zeros_like(Mu)

    for i in range(resolution):
        for j in range(resolution):
            F = solver.firing_rate(Mu[j, i] * mV, Sigma[j, i] * mV)
            if F is None or np.isnan(F/Hz):
                F = old_F
            else:
                old_F = F

            rate_grid[j, i] = F / Hz
            loss_grid[j, i] = 0.5 * ((F - r_target) / Hz) ** 2

    target_rate = r_target / Hz

    print("XXXXXXXX dtype:", loss_grid.dtype)
    print("NaN count:", np.isnan(loss_grid).sum())
    print("Inf count:", np.isinf(loss_grid).sum())
    print("Min value:", np.nanmin(loss_grid))
    print("Max value:", np.nanmax(loss_grid))
    # Create figure with two subplots
    fig = plt.figure(figsize=(18, 8))

    # Plot 1: Log scale with clipping to see valley
    ax1 = fig.add_subplot(121, projection='3d')

    # Clip extremely high values to reveal the valley
    clip_percentile = 90  # Clip top 10%
    vmax = np.percentile(loss_grid, clip_percentile)
    loss_clipped = np.clip(loss_grid, 0, vmax)

    # Use log scale for better contrast
    loss_log = np.log10(loss_clipped + 1e-10)

    # Plot surface
    surf1 = ax1.plot_surface(Mu, Sigma, loss_log,
                             cmap=cm.viridis, alpha=0.9,
                             linewidth=0, antialiased=True)

    # Find and plot solution curve
    tolerance = 0.05 * target_rate
    mask = np.abs(rate_grid - target_rate) < tolerance

    if np.any(mask):
        curve_mu = Mu[mask]
        curve_sigma = Sigma[mask]
        curve_loss_log = loss_log[mask]

        ax1.scatter(curve_mu, curve_sigma, curve_loss_log,
                    c='red', s=30, alpha=1, label=f'F = {target_rate} Hz')

    ax1.set_xlabel('$\mu$ (mV)')
    ax1.set_ylabel('$\sigma$ (mV)')
    ax1.set_zlabel('log10(Loss)')
    ax1.set_title(f'3D Loss Landscape (log10 scale, clipped top {100 - clip_percentile}%)')
    ax1.legend()

    # Plot 2: Zoomed view around the valley
    ax2 = fig.add_subplot(122, projection='3d')

    # Find region around solution curve
    if np.any(mask):
        mu_curve_avg = np.mean(curve_mu)
        sigma_curve_avg = np.mean(curve_sigma)

        # Zoom to ±20% around the curve
        mu_width = (mu_range[1] - mu_range[0]) * 0.3
        sigma_width = (sigma_range[1] - sigma_range[0]) * 0.3

        mu_zoom = (mu_curve_avg - mu_width, mu_curve_avg + mu_width)
        sigma_zoom = (sigma_curve_avg - sigma_width, sigma_curve_avg + sigma_width)

        # Create zoom mask
        zoom_mask = (Mu >= mu_zoom[0]) & (Mu <= mu_zoom[1]) & \
                    (Sigma >= sigma_zoom[0]) & (Sigma <= sigma_zoom[1])

        # Apply zoom mask
        Mu_zoom = np.ma.masked_where(~zoom_mask, Mu)
        Sigma_zoom = np.ma.masked_where(~zoom_mask, Sigma)
        Loss_zoom = np.ma.masked_where(~zoom_mask, loss_log)

        surf2 = ax2.plot_surface(Mu_zoom, Sigma_zoom, Loss_zoom,
                                 cmap=cm.plasma, alpha=0.9,
                                 linewidth=0, antialiased=True)

        # Plot curve in zoomed region
        mask_zoom = mask & zoom_mask
        if np.any(mask_zoom):
            ax2.scatter(Mu[mask_zoom], Sigma[mask_zoom], loss_log[mask_zoom],
                        c='red', s=50, alpha=1)

    ax2.set_xlabel('$\mu$ (mV)')
    ax2.set_ylabel('$\sigma$ (mV)')
    ax2.set_zlabel('log10(Loss)')
    ax2.set_title('Zoomed view around solution curve')

    plt.tight_layout()
    plt.show()

    return fig


# Alternative: 2D heatmap with 3D inset
def plot_loss_with_profile(solver, r_target, mu_range, sigma_range, resolution=50):
    """
    Combined 2D heatmap and 3D profile to see the valley
    """
    # Compute grid (same as before)
    mu_grid = np.linspace(mu_range[0], mu_range[1], resolution)
    sigma_grid = np.linspace(sigma_range[0], sigma_range[1], resolution)
    Mu, Sigma = np.meshgrid(mu_grid, sigma_grid)

    loss_grid = np.zeros_like(Mu)
    rate_grid = np.zeros_like(Mu)

    for i in range(resolution):
        for j in range(resolution):
            F = solver.firing_rate(Mu[j, i] * mV, Sigma[j, i] * mV)
            rate_grid[j, i] = F / Hz
            loss_grid[j, i] = 0.5 * ((F - r_target) / Hz) ** 2

    target_rate = r_target / Hz

    fig = plt.figure(figsize=(16, 8))

    # 2D heatmap with log scale
    ax1 = fig.add_subplot(121)

    # Use log scale with masking for zeros
    loss_log = np.log10(loss_grid + 1e-10)

    im = ax1.imshow(loss_log, extent=[mu_range[0], mu_range[1],
                                      sigma_range[0], sigma_range[1]],
                    origin='lower', aspect='auto', cmap='viridis',
                    vmin=np.percentile(loss_log, 5),
                    vmax=np.percentile(loss_log, 95))
    plt.colorbar(im, ax=ax1, label='log₁₀(Loss)')

    # Solution curve
    ax1.contour(Mu, Sigma, rate_grid, levels=[target_rate],
                colors='red', linewidths=3)

    ax1.set_xlabel('$\mu$ (mV)')
    ax1.set_ylabel('$\sigma$ (mV)')
    ax1.set_title('2D Loss Map with Solution Curve')

    # 3D profile along a line through the valley
    ax2 = fig.add_subplot(122, projection='3d')

    # Find a point on the curve
    mask = np.abs(rate_grid - target_rate) < 0.05 * target_rate
    if np.any(mask):
        mu_curve = np.mean(Mu[mask])
        sigma_curve = np.mean(Sigma[mask])

        # Take cross-sections
        mu_idx = np.argmin(np.abs(mu_grid - mu_curve))
        sigma_idx = np.argmin(np.abs(sigma_grid - sigma_curve))

        # Plot loss along mu at fixed sigma
        mu_line = mu_grid
        loss_mu = loss_grid[sigma_idx, :]

        # Plot loss along sigma at fixed mu
        sigma_line = sigma_grid
        loss_sigma = loss_grid[:, mu_idx]

        # Create 3D lines
        ax2.plot(mu_line, [sigma_curve] * len(mu_line), loss_mu,
                 'b-', linewidth=2, label=f'σ = {sigma_curve:.1f} mV')
        ax2.plot([mu_curve] * len(sigma_line), sigma_line, loss_sigma,
                 'g-', linewidth=2, label=f'μ = {mu_curve:.1f} mV')

        # Mark the minimum
        ax2.scatter([mu_curve], [sigma_curve], [loss_grid[sigma_idx, mu_idx]],
                    c='red', s=100, marker='o', label='Solution point')

    ax2.set_xlabel('$\mu$ (mV)')
    ax2.set_ylabel('$\sigma$ (mV)')
    ax2.set_zlabel('Loss')
    ax2.set_title('Loss Profiles through Solution')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return fig


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_loss_landscape_3d_with_limits(solver, r_target, mu_range, sigma_range,
                                       resolution=50, z_max=1000):
    """
    3D plot with vertical axis limits to reveal the valley

    Parameters:
    -----------
    z_max : float, maximum loss value to display (clip everything above this)
    """
    # Create grid
    mu_grid = np.linspace(mu_range[0], mu_range[1], resolution)
    sigma_grid = np.linspace(sigma_range[0], sigma_range[1], resolution)
    Mu, Sigma = np.meshgrid(mu_grid, sigma_grid)

    # Compute loss
    loss_grid = np.zeros_like(Mu)
    rate_grid = np.zeros_like(Mu)

    print("Computing loss landscape...")
    for i in range(resolution):
        for j in range(resolution):
            F = solver.firing_rate(Mu[j, i] * mV, Sigma[j, i] * mV)
            rate_grid[j, i] = F / Hz
            loss_grid[j, i] = 0.5 * ((F - r_target) / Hz) ** 2

    target_rate = r_target / Hz

    # Clip loss values above z_max for visualization
    loss_clipped = np.clip(loss_grid, 0, z_max)

    # Create figure
    fig = plt.figure(figsize=(14, 6))

    # Plot 1: 3D with clipped z-axis
    ax1 = fig.add_subplot(121, projection='3d')

    surf1 = ax1.plot_surface(Mu, Sigma, loss_clipped,
                             cmap=cm.viridis, alpha=0.9,
                             linewidth=0, antialiased=True)

    # Set z-axis limits
    ax1.set_zlim(0, z_max)

    # Find and plot solution curve (where loss is minimal)
    tolerance = 0.05 * target_rate
    mask = np.abs(rate_grid - target_rate) < tolerance

    if np.any(mask):
        curve_mu = Mu[mask]
        curve_sigma = Sigma[mask]
        curve_loss = loss_clipped[mask]  # Use clipped values for plotting

        ax1.scatter(curve_mu, curve_sigma, curve_loss,
                    c='red', s=20, alpha=0.8, label=f'F = {target_rate} Hz')

    ax1.set_xlabel('μ (mV)')
    ax1.set_ylabel('σ (mV)')
    ax1.set_zlabel('Loss')
    ax1.set_title(f'3D Loss Landscape (clipped at z={z_max})')
    ax1.legend()

    # Plot 2: Same but with even lower z_max to see valley better
    ax2 = fig.add_subplot(122, projection='3d')

    # Use even lower limit for the valley view
    z_valley = z_max / 10
    loss_valley = np.clip(loss_grid, 0, z_valley)

    surf2 = ax2.plot_surface(Mu, Sigma, loss_valley,
                             cmap=cm.plasma, alpha=0.9,
                             linewidth=0, antialiased=True)

    ax2.set_zlim(0, z_valley)

    if np.any(mask):
        curve_loss_valley = loss_valley[mask]
        ax2.scatter(curve_mu, curve_sigma, curve_loss_valley,
                    c='red', s=20, alpha=0.8)

    ax2.set_xlabel('μ (mV)')
    ax2.set_ylabel('σ (mV)')
    ax2.set_zlabel('Loss')
    ax2.set_title(f'Zoomed to valley (clipped at z={z_valley})')

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"Loss statistics:")
    print(f"  Min loss: {loss_grid.min():.6e}")
    print(f"  Max loss: {loss_grid.max():.6f}")
    print(f"  Loss at solution: {loss_grid[mask].min() if np.any(mask) else 'N/A'}")

    return fig


# Even simpler: Just one plot with proper z-limit
def simple_3d_with_limits(solver, r_target, mu_range, sigma_range,
                          resolution=50, z_max=None):
    """
    Simple 3D plot with automatic or manual z-axis limits
    """
    # Compute grid (same as above)
    mu_grid = np.linspace(mu_range[0], mu_range[1], resolution)
    sigma_grid = np.linspace(sigma_range[0], sigma_range[1], resolution)
    Mu, Sigma = np.meshgrid(mu_grid, sigma_grid)

    loss_grid = np.zeros_like(Mu)
    rate_grid = np.zeros_like(Mu)

    for i in range(resolution):
        for j in range(resolution):
            F = solver.firing_rate(Mu[j, i] * mV, Sigma[j, i] * mV)
            rate_grid[j, i] = F / Hz
            loss_grid[j, i] = 0.5 * ((F - r_target) / Hz) ** 2

    # If z_max not provided, use 10x the minimum loss around solution
    if z_max is None:
        target_rate = r_target / Hz
        mask = np.abs(rate_grid - target_rate) < 0.1 * target_rate
        if np.any(mask):
            z_max = 10 * loss_grid[mask].min()
        else:
            z_max = 100  # fallback

    # Clip
    loss_clipped = np.clip(loss_grid, 0, z_max)

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(Mu, Sigma, loss_clipped,
                           cmap=cm.viridis, alpha=0.9,
                           linewidth=0, antialiased=True)

    ax.set_zlim(0, z_max)

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Loss')

    ax.set_xlabel('μ (mV)')
    ax.set_ylabel('σ (mV)')
    ax.set_zlabel('Loss')
    ax.set_title(f'3D Loss Landscape (z-axis clipped at {z_max:.2f})')

    plt.show()

    return fig


# Usage:
# simple_3d_with_limits(solver, r_target=0.3*Hz,
#                       mu_range=(-60, -35), sigma_range=(0, 10),
#                       z_max=500)  # Adjust this value until you see the valley!

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

        fig = plot_loss_landscape_3d_with_valley(
            solver, r_target, (-60, -45), (0, 6),
            resolution=200, target_rate_Hz=0.3
        )

        fig.show()