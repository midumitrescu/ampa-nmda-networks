import matplotlib.pyplot as plt
import numpy as np
from brian2 import mV, ms, Hz, second, have_same_dimensions
from loguru import logger
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy import special
from scipy.integrate import quad

from build.lib.src.Plotting import show_plots_non_blocking

mHz = 1e-3 * Hz

def create_anneal_decay_schedule(n_steps,
                                 no_anneal=500, decay_every=100,
                                 decay_factor=0.9):
    steps = np.arange(n_steps)

    # Calculate decay exponent
    decay_exponent = np.floor(np.ceil(np.maximum(0, steps - no_anneal)) / decay_every)

    # Calculate learning rates


    return decay_factor ** decay_exponent


def erfcx(x):
    """Scaled complementary error function: exp(x^2)*erfc(x)"""
    return special.erfcx(x)


def rate_LIF_whitenoise(mu, tau_membrane, sigma_v, theta, V_reset, tau_ref):
    """
    Compute firing rate of LIF neuron with white noise input

    Parameters:
    -----------
    V_mean : Brian2 quantity (mV) - mean membrane potential
    tau : Brian2 quantity (ms) - membrane time constant
    sigmaV : Brian2 quantity (mV) - noise standard deviation
    Vth : Brian2 quantity (mV) - threshold voltage
    Vreset : Brian2 quantity (mV) - reset voltage
    tref : Brian2 quantity (ms) - refractory period

    Returns:
    --------
    Brian2 quantity (Hz) - firing rate
    """

    # Integration bounds
    lower_limit, upper_limit = integration_limits(mu, V_reset, sigma_v, theta)

    # Ensure to < upper_limit for integration
    if lower_limit > upper_limit:
        lower_limit, upper_limit = upper_limit, lower_limit

    # Numerical integration of erfcx (which is e^{x^2}erfc(x))
    I_mu_sigma, _ = quad(erfcx, lower_limit, upper_limit, epsabs=1e-18, epsrel=1e-13)

    # Compute firing rate
    rate = 1.0 / (tau_ref + tau_membrane * np.sqrt(np.pi) * I_mu_sigma)

    return rate


def integration_limits(V_mean, V_reset, sigma_v, theta):
    lower_limit = (V_mean - theta) / (np.sqrt(2) * sigma_v)
    upper_limit = (V_mean - V_reset) / (np.sqrt(2) * sigma_v)
    return lower_limit, upper_limit


class SiegertGradientDescent:
    def __init__(self, tau_m, theta, v_reset, tau_ref, unit='mV'):
        """
        Initialize with Brian2 units

        Parameters:
        -----------
        tau : Brian2 quantity (time) - membrane time constant
        Vth : Brian2 quantity (voltage) - threshold voltage
        Vreset : Brian2 quantity (voltage) - reset voltage
        tref : Brian2 quantity (time) - refractory period
        unit : str, unit system ('mV' or 'V')
        """
        self.theta = theta
        self.v_reset = v_reset
        self.unit = unit
        self.tau_m = tau_m
        self.tau_ref = tau_ref

    def firing_rate(self, V_mean, sigmaV):
        """
        Compute firing rate using the normalized form

        Parameters:
        -----------
        V_mean : Brian2 quantity (voltage) - mean membrane potential μ
        sigmaV : Brian2 quantity (voltage) - noise standard deviation σ

        Returns:
        --------
        Brian2 quantity (Hz) - firing rate
        """
        return rate_LIF_whitenoise(V_mean, self.tau_m, sigmaV,
                                   self.theta, self.v_reset, self.tau_ref)

    def phi(self, z):
        """Φ(z) = erfcx(z) = exp(z^2)*erfc(z)"""
        return erfcx(z)

    def gradient_loss(self, mu_v, sigma_v, r_target):

        # (V_mean, V_reset, sigma_v, theta)
        lower_limit, upper_limit = integration_limits(V_mean=mu_v, V_reset=self.v_reset, sigma_v=sigma_v,
                                                      theta=self.theta)

        phi_vect = np.array([self.phi(upper_limit), self.phi(lower_limit)]).T
        matrix = np.array([[1, -1],
                           [- upper_limit * np.sqrt(2), lower_limit * np.sqrt(2)]])

        # rate_LIF_whitenoise(mu, tau_membrane, sigma_v, theta, V_reset, tau_ref):
        f_lif = rate_LIF_whitenoise(mu=mu_v, tau_membrane=self.tau_m, sigma_v=sigma_v, theta=self.theta,
                                    V_reset=self.v_reset, tau_ref=self.tau_ref)

        return self.tau_m / sigma_v * (f_lif - r_target) * f_lif ** 2 * matrix @ phi_vect

    def update_step(self, V_mean, sigmaV, r_target, learning_rate):
        """
        Perform one gradient descent update

        Parameters:
        -----------
        V_mean : Brian2 quantity (voltage)
        sigmaV : Brian2 quantity (voltage)
        r_target : Brian2 quantity (firing rate)
        learning_rate : float (dimensionless)

        Returns:
        --------
        tuple : (V_mean_new, sigmaV_new) as Brian2 quantities
        """
        dL_dmu, dL_dsigma = self.gradient_loss(V_mean, sigmaV, r_target)

        # Update: p_new = p + η·∇L
        V_mean_new = V_mean + learning_rate * dL_dmu
        sigmaV_new = sigmaV + learning_rate * dL_dsigma

        # Ensure sigma stays positive
        if sigmaV_new <= 0 * mV:
            sigmaV_new = 1e-6 * mV

        return V_mean_new, sigmaV_new

    def find_parameters(self, r_target, mu_0, sigma_0,
                        learning_rate=1e-3 * (mV * second) ** 2, n_steps=5000, anneal_schedule=None,
                        tolerance=1e-8 * Hz ** 2):
        """
        Find μ and σ that give target firing rate

        Parameters:
        -----------
        r_target : target firing rate [Hz]
        mu_0 : initial μ guess [mV]
        sigma_0 : initial σ [mV]
        learning_rate : float, learning rate. Attention: unit must be [volt**2 * second**2]. Why?
        tolerance : in MSE space => Hz**2
        Returns:
        --------
        tuple : (mu_final, sigma_final, history)
        """
        logger.debug(f"\n{'=' * 60}")
        logger.debug(f"Finding parameters for r_target = {r_target}")
        logger.debug(f"{'=' * 60}")

        assert have_same_dimensions(1 * Hz, r_target)
        assert have_same_dimensions(1 * mV, mu_0)
        assert have_same_dimensions(1 * mV, sigma_0)
        assert have_same_dimensions(1 * mV ** 2 * second ** 2, learning_rate)
        assert have_same_dimensions(1 * Hz ** 2, tolerance)

        mu = mu_0
        sigma = sigma_0

        history = {
            'mu': [], 'sigma': [], 'f_rate': [], 'loss': []
        }

        if anneal_schedule is None:
            anneal_schedule = np.ones(n_steps)
        elif anneal_schedule:
            anneal_schedule = create_anneal_decay_schedule(n_steps=n_steps)

        for step, anneal in zip(range(n_steps), anneal_schedule):
            # Compute current firing rate and loss
            F = self.firing_rate(mu, sigma)
            loss = 0.5 * ((F - r_target)) ** 2  # dimensionless

            # Store history (store numerical values for plotting)
            history['mu'].append(mu)
            history['sigma'].append(sigma)
            history['f_rate'].append(F)
            history['loss'].append(loss)

            # Check convergence
            if loss < tolerance:
                logger.info(f"Converged at step {step} with loss = {loss:.7f}")
                break

            # Update parameters
            mu, sigma = self.update_step(mu, sigma, r_target, learning_rate*anneal)

            if step % 100 == 0:
                logger.debug(f"Step {step:3d}: μ={mu:.6f}, σ={sigma:.6f}, "
                      f"F={F:.6f}, loss={loss:.6e}, lr={learning_rate/((mV * second) ** 2) * anneal}")

        logger.info("Final results in {} steps", step)
        logger.info(f"μ = {mu / mV:.6f} mV")
        logger.info(f"σ = {sigma / mV:.6f} mV")
        logger.info(f"Firing rate = {F / mHz:.6f} mHz")
        logger.info(f"Target = {r_target / mHz : .2f} mHz")
        logger.info(f"Error = {abs(F - r_target) / mHz:.6f}")

        return mu, sigma, history

def plot_grad_descent(history: dict, r_target: float):
    # Plot convergence
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    steps = range(len(history['mu']))

    # --- μ convergence ---
    axes[0, 0].plot(steps, history['mu'] / mV)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel(r'$\mu$ (mV)')
    axes[0, 0].set_title(r'$\mu$'f' convergence (target {r_target})')
    axes[0, 0].grid(True, alpha=0.3)

    # --- σ convergence ---
    axes[0, 1].plot(steps, history['sigma'] / mV)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel(r'$\sigma$ (mV)')
    axes[0, 1].set_title(r'$\sigma$'f' convergence (target {r_target})')
    axes[0, 1].grid(True, alpha=0.3)

    # --- Loss convergence ---
    axes[1, 0].semilogy(steps, history['loss'])
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Loss convergence')
    axes[1, 0].grid(True, alpha=0.3)

    # --- Firing rate convergence ---
    axes[1, 1].plot(steps, history['f_rate'] / Hz)
    axes[1, 1].axhline(y=r_target / Hz, color='r', linestyle='--',
                       label=f'Target {r_target / Hz:.4f} Hz')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Firing rate (Hz)')
    axes[1, 1].set_title('Firing rate convergence')
    axes[1, 1].legend(fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    ax_phase = axes[2, 0]

    mu_vals = np.array(history['mu']) / mV
    sigma_vals = np.array(history['sigma']) / mV

    # Distance to target rate
    dist = np.abs(history["f_rate"] - r_target)/Hz

    # Create colored line segments
    points = np.array([mu_vals, sigma_vals]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = Normalize(vmin=dist.min(), vmax=dist.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(dist[:-1])
    lc.set_linewidth(2)

    ax_phase.add_collection(lc)
    ax_phase.scatter(mu_vals[0], sigma_vals[0], color='red', label='Start', zorder=3)
    ax_phase.scatter(mu_vals[-1], sigma_vals[-1], color='black', label='End', zorder=3)

    ax_phase.set_xlabel(r'$\mu$ (mV)')
    ax_phase.set_ylabel(r'$\sigma$ (mV)')
    ax_phase.set_title(r'Gradient descent in $\mu$–$\sigma$ space')
    ax_phase.legend(fontsize=12)
    ax_phase.grid(True, alpha=0.3)

    # Colorbar
    cbar = fig.colorbar(lc, ax=ax_phase)
    cbar.set_label('Distance to target rate (Hz)')

    # Hide unused subplot (bottom-right corner)
    axes[2, 1].axis('off')

    fig.suptitle(f'''
    Looking for {r"$\mu$"}, {r"$\sigma$"} resulting in {r_target / Hz:.4f} Hz
    {r"$\mu_\mathrm{{sol}}=$"} {history['mu'][-1] / mV :.2f} mV,
    {r"$\sigma_\mathrm{{sol}}=$"} {history['sigma'][-1] / mV :.2f} mV
    resulting in {r"$r_\mathrm{{sol}}=$"} {history['f_rate'][-1] / Hz :.4f} Hz
    ''', fontsize=18)
    fig.tight_layout()

    for ax in fig.get_axes():
        ax.title.set_fontsize(16)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.tick_params(axis='both', labelsize=12)

    fig.canvas.draw_idle()

    show_plots_non_blocking()

# Example usage with Brian2 units
if __name__ == "__main__":
    # Set up Brian2 unit system
    # start_scope()

    # Neuron parameters (typical values for cortical neuron)
    tau = 10 * ms  # membrane time constant
    Vth = -40 * mV  # threshold voltage
    Vreset = -65 * mV  # reset voltage
    tref = 2 * ms  # refractory period

    # Create solver instance
    solver = SiegertGradientDescent(tau, Vth, Vreset, tref, unit='mV')

    # Target firing rates
    r_targets = [0.05 * Hz, 0.3 * Hz]

    # Initial guesses (in mV) - adjusted for normalized form
    initial_guesses = [
        (-55 * mV, 5 * mV),  # for 0.05 Hz: moderate mu, moderate sigma
        (-45 * mV, 8 * mV)  # for 0.3 Hz: higher mu, higher sigma
    ]

    # Find parameters for each target
    for i, (r_target, (mu_init, sigma_init)) in enumerate(zip(r_targets, initial_guesses)):

        mu_sol, sigma_sol, history = solver.find_parameters(
            r_target, mu_init, sigma_init,
            anneal_schedule=True,
            n_steps = 10_000
        )

        plot_grad_descent(history=history, r_target=r_target)
