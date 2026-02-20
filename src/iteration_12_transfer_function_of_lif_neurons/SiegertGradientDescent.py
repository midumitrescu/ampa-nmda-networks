import unittest

import numpy as np
from brian2 import mV, ms, Hz, second, have_same_dimensions
from scipy import special
from scipy.integrate import quad
import matplotlib.pyplot as plt

mHz = 1e-3 * Hz

def create_anneal_decay_schedule(n_steps, base_lr=1e-3,
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
    I_mu_sigma, _ = quad(erfcx, lower_limit, upper_limit, epsabs=1e-13, epsrel=1e-13)

    # Compute firing rate
    rate = 1.0 / (tau_ref + tau_membrane * np.sqrt(np.pi) * I_mu_sigma)

    return rate


def integration_limits(V_mean, V_reset, sigma_v, theta):
    lower_limit = (V_mean - theta) / (np.sqrt(2) * sigma_v)
    upper_limit = (V_mean - V_reset) / (np.sqrt(2) * sigma_v)
    return lower_limit, upper_limit


class SiegertGradientDescent:
    def __init__(self, tau_m, theta, Vreset, tau_ref, unit='mV'):
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
        self.V_reset = Vreset
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
                                   self.theta, self.V_reset, self.tau_ref)

    def phi(self, z):
        """Φ(z) = erfcx(z) = exp(z^2)*erfc(z)"""
        return erfcx(z)

    def gradient_loss(self, mu_v, sigma_v, r_target):

        # (V_mean, V_reset, sigma_v, theta)
        lower_limit, upper_limit = integration_limits(V_mean=mu_v, V_reset=self.V_reset, sigma_v=sigma_v,
                                                      theta=self.theta)

        phi_vect = np.array([self.phi(upper_limit), self.phi(lower_limit)]).T
        matrix = np.array([[1, -1],
                           [- upper_limit * np.sqrt(2), lower_limit * np.sqrt(2)]])

        # rate_LIF_whitenoise(mu, tau_membrane, sigma_v, theta, V_reset, tau_ref):
        f_lif = rate_LIF_whitenoise(mu=mu_v, tau_membrane=self.tau_m, sigma_v=sigma_v, theta=self.theta,
                                    V_reset=self.V_reset, tau_ref=self.tau_ref)

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
                        tolerance=1e-8 * Hz ** 2, verbose=True):
        """
        Find μ and σ that give target firing rate

        Parameters:
        -----------
        r_target : target firing rate [Hz]
        mu_0 : initial μ guess [mV]
        sigma_0 : initial σ [mV]
        learning_rate : float, learning rate. Attention: unit must be [volt**2 * second**2]. Why?
        tolerance : in MSE space => Hz**2
        verbose : bool, print progress
        Returns:
        --------
        tuple : (mu_final, sigma_final, history)
        """

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
                if verbose:
                    print(f"Converged at step {step} with loss = {loss:.7f}")
                break

            # Update parameters
            mu, sigma = self.update_step(mu, sigma, r_target, learning_rate*anneal)

            if verbose and step % 100 == 0:
                print(f"Step {step:3d}: μ={mu:.6f}, σ={sigma:.6f}, "
                      f"F={F:.6f}, loss={loss:.6e}, lr={learning_rate/((mV * second) ** 2) * anneal}")

        return mu, sigma, history


# Example usage with Brian2 units
if __name__ == "__main__":
    # Set up Brian2 unit system
    # start_scope()

    # Neuron parameters (typical values for cortical neuron)
    tau = 10 * ms  # membrane time constant
    Vth = -50 * mV  # threshold voltage
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
        print(f"\n{'=' * 60}")
        print(f"Finding parameters for r_target = {r_target}")
        print(f"{'=' * 60}")

        mu_final, sigma_final, history = solver.find_parameters(
            r_target, mu_init, sigma_init,
            verbose=True,
            anneal_schedule=True,
            n_steps = 10_000
        )

        # Verify
        F_final = solver.firing_rate(mu_final, sigma_final)

        print(f"\nFinal results:")
        print(f"μ = {mu_final / mV:.6f} mV")
        print(f"σ = {sigma_final / mV:.6f} mV")
        print(f"Firing rate = {F_final / mHz:.6f} mHz")
        print(f"Target = {r_target / mHz : .2f} mHz")
        print(f"Error = {abs(F_final - r_target) / mHz:.6f}")

        # Plot convergence
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        steps = range(len(history['mu']))

        axes[0, 0].plot(steps, history['mu'] / mV)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('μ (mV)')
        axes[0, 0].set_title(f'μ convergence (target {r_target})')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(steps, history['sigma'] / mV)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('σ (mV)')
        axes[0, 1].set_title(f'σ convergence (target {r_target})')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].semilogy(steps, history['loss'])
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Loss convergence')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(steps, history['f_rate'])
        axes[1, 1].axhline(y=r_target / Hz, color='r', linestyle='--',
                           label=f'Target {r_target}')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Firing rate (Hz)')
        axes[1, 1].set_title('Firing rate convergence')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
