import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import mV, second, ms, Hz, have_same_dimensions, is_dimensionless

from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import rate_LIF_whitenoise, \
    integration_limits, SiegertGradientDescent, create_anneal_decay_schedule, plot_grad_descent
from iteration_8_compute_mean_steady_state.models_and_configs import palmer_experiment_0_1_Hz_with_NMDA_block
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control

tau_m = 20 * ms
mu = -55 * mV
v_reset = -65 * mV
theta = -50 * mV
sigma = 5 * mV
tau_ref = 2 * ms

r_target = 0.3 * Hz

learning_rate=1e-3 * (mV * second)**2

class GradientDescentTestCases(unittest.TestCase):

    def test_rate_LIF_plots_correctly_tilos_version(self):
        L = 1001  # #datapoints
        mu = np.linspace(0, 20, L) * mV
        sigmaV = np.array([0.5, 1., 2., 4., 6.]) * mV
        rate = np.zeros((len(sigmaV), L))

        taum = 20 * ms
        Vth = 15.0 * mV
        Vreset = 0. * mV
        tref = 2 * ms

        for i in range(len(sigmaV)):
            print(sigmaV[i])
            for j in range(L):
                rate[i, j] = rate_LIF_whitenoise(mu[j], taum, sigmaV[i], Vth, Vreset, tref)

        # firing rate for sigma=0 (no noise)
        rate_determ = np.zeros(L)
        for j in range(L):
            if mu[j] > Vth:
                T = taum * np.log((mu[j] - Vreset) / (mu[j] - Vth))
                rate_determ[j] = 1. / (T + tref)

        plt.plot(mu / mV, rate_determ / Hz, ls='--', color='k', label=r'$\sigma_V=0$mV')
        for i in range(len(sigmaV)):
            plt.plot(mu / mV, rate[i] / Hz, label=r'$\sigma_V=%g$mV' % (sigmaV[i] / mV,))
        plt.xlabel(r'input $\mu$ [mV]')
        plt.ylabel('firing rate [Hz]')
        plt.legend(loc=0)
        plt.show()

    def test_rate_LIF_white_noise_can_handle_zero_noise(self):

        mu = -35 * mV

        T = tau_m * np.log((mu - v_reset) / (mu - theta))
        rate_determ = 1. / (T + tau_ref)

        self.assertEqual(rate_determ / Hz, rate_LIF_whitenoise(mu, tau_membrane=tau_m, sigma_v=0*mV, theta=theta, tau_ref=tau_ref, V_reset=v_reset) / Hz)

    def test_rate_LIF_whitenoise_for_zero_noise_and_subthreshold_mean(self):
        mean = -50.00000000034059 * mV

        print(rate_LIF_whitenoise(mean, tau_membrane=20*ms, sigma_v=0*mV, theta=-40*mV, tau_ref=2*ms, V_reset=-55*mV) / Hz)


    def test_rate_LIF_plots_correctly(self):
        L = 1001  # #datapoints
        mu = np.linspace(-55, -35, L) * mV
        sigmaV = np.array([0.5, 1., 2., 4., 6.]) * mV
        rate = np.zeros((len(sigmaV), L))

        taum = 20 * ms
        Vth = -40 * mV
        Vreset = -50. * mV
        tref = 2 * ms

        for i in range(len(sigmaV)):
            print(sigmaV[i])
            for j in range(L):
                rate[i, j] = rate_LIF_whitenoise(mu[j], taum, sigmaV[i], Vth, Vreset, tref)

        # firing rate for sigma=0 (no noise)
        rate_determ = np.zeros(L)
        for j in range(L):
            if mu[j] > Vth:
                T = taum * np.log((mu[j] - Vreset) / (mu[j] - Vth))
                rate_determ[j] = 1. / (T + tref)

        plt.plot(mu / mV, rate_determ / Hz, ls='--', color='k', label=r'$\sigma_V=0$mV')
        for i in range(len(sigmaV)):
            plt.plot(mu / mV, rate[i] / Hz, label=r'$\sigma_V=%g$mV' % (sigmaV[i] / mV,))
        plt.xlabel(r'input $\mu$ [mV]')
        plt.ylabel('firing rate [Hz]')

        plt.axhline(y=0.05, color='orange', linestyle='-', label='Mk801')
        plt.axhline(y=0.3, color='black', linestyle='-', label='Control')
        # Set y-axis limits
        plt.ylim(0, 0.5)

        plt.legend(loc=0)

        plt.show()

    def test_plot_sigma_vs_rate(self):
        mu = np.linspace(-55, -40, 5) * mV
        sigmaV = np.linspace(0, 10, num=1000) * mV

        taum = 20 * ms
        Vth = -40 * mV
        Vreset = -50. * mV
        tref = 2 * ms

        rate_sigma = np.zeros((len(mu), len(sigmaV)))

        for i in range(len(mu)):
            for j in range(len(sigmaV)):
                rate_sigma[i, j] = rate_LIF_whitenoise(
                    mu[i], taum, sigmaV[j], Vth, Vreset, tref
                )

        # Plot
        for i in range(len(mu)):
            plt.plot(sigmaV / mV,
                     rate_sigma[i] / Hz,
                     label=r'$\mu=%g$ mV' % (mu[i] / mV,))

        # Horizontal reference lines
        plt.axhline(y=0.05, color='orange', linestyle='-', label='Mk801')
        plt.axhline(y=0.3, color='black', linestyle='-', label='Control')

        plt.xlabel(r'noise $\sigma_V$ [mV]')
        plt.ylabel('firing rate [Hz]')
        #plt.ylim(0, 0.5)
        plt.legend(loc=0)
        plt.show()

    def test_limit_units(self):

        object_under_test = SiegertGradientDescent(tau_m=tau_m, theta = theta, v_reset=v_reset, tau_ref = tau_ref)

        self.assertTrue(have_same_dimensions(1*mV, mu))

        lower_limit, upper_limit = integration_limits(V_mean=mu, V_reset=v_reset, sigma_v=sigma, theta=theta)
        self.assertTrue(is_dimensionless(lower_limit))
        self.assertTrue(is_dimensionless(upper_limit))

        self.assertTrue(is_dimensionless(object_under_test.phi(lower_limit)))
        self.assertTrue(is_dimensionless(object_under_test.phi(upper_limit)))

    def test_units_of_gradient_Loss(self):
        object_under_test = SiegertGradientDescent(tau_m=tau_m, theta=theta, v_reset=v_reset, tau_ref=tau_ref)

        self.assertTrue(have_same_dimensions(1*Hz, rate_LIF_whitenoise(mu = mu, tau_membrane=tau_m, sigma_v=sigma, theta=theta, V_reset=v_reset, tau_ref=tau_ref)))
        self.assertTrue(have_same_dimensions(1*Hz, rate_LIF_whitenoise(mu = mu, tau_membrane=tau_m, sigma_v=sigma, theta=theta, V_reset=v_reset, tau_ref=tau_ref) - 0.05 * Hz))

        # mu_v, sigma_v, r_target
        grad_L_mu, grad_L_sigma = object_under_test.gradient_loss(mu_v = mu, sigma_v = sigma, r_target = r_target)
        self.assertTrue(have_same_dimensions(1 * Hz**2 / mV, grad_L_mu))
        self.assertTrue(have_same_dimensions(1 * Hz**2 / mV, grad_L_sigma))

        self.assertTrue(have_same_dimensions(1 * mV, learning_rate * grad_L_mu))
        self.assertTrue(have_same_dimensions(1 * mV, learning_rate * grad_L_sigma))

    def test_try_annealing_schedule(self):
        num_steps = 5000
        decay_factor = create_anneal_decay_schedule(num_steps)
        lr = 1E-3
        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Learning rate (linear scale)
        axes[0].plot(np.arange(0, num_steps), lr * decay_factor, 'g-', linewidth=2)
        axes[0].axvline(x=500, color='r', linestyle='--', alpha=0.7, label='Annealing starts')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Learning Rate')
        axes[0].set_title(f'Learning Rate Schedule: base_lr=1e-3, decay_factor=0.999')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Plot 3: Learning rate (log scale) - better for seeing decay
        axes[1].semilogy(np.arange(0, num_steps), lr * decay_factor, 'r-', linewidth=2)
        axes[1].axvline(x=500, color='b', linestyle='--', alpha=0.7, label='Annealing starts')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Learning Rate (log scale)')
        axes[1].set_title('Learning Rate Schedule (log scale)')
        axes[1].grid(True, alpha=0.3, which='both')
        axes[1].legend()


        plt.tight_layout()
        plt.show()


    def test_grad_descent_0_05_Hz(self):
        experiment = palmer_experiment_0_1_Hz_with_NMDA_block

        r_target = 0.05 * Hz
        solver = SiegertGradientDescent(tau_m=experiment.effective_time_constant_up_state.tau_eff(),
                                        theta=experiment.neuron_params.theta,
                                        v_reset=experiment.neuron_params.V_r,
                                        tau_ref=experiment.neuron_params.tau_rp, unit='mV')

        # Initial guesses (in mV) - adjusted for normalized form
        mu_0, sigma_0 = -56 * mV, 2.5 * mV

        ''' def find_parameters(self, r_target, mu_0, sigma_0,
                        learning_rate=1e-3 * (mV * second) ** 2, n_steps=5000, anneal_schedule=None,
                        tolerance=1e-8 * Hz ** 2):'''
        mu_sol, sigma_sol, history = solver.find_parameters(
            r_target=r_target,
            mu_0=mu_0,
            sigma_0=sigma_0,
            learning_rate=1e-2 * (mV * second) ** 2,
            anneal_schedule=None,
            n_steps=10_000
        )

        plot_grad_descent(history=history, r_target=r_target)

    def test_grad_palmer_control(self):
        experiment = palmer_control

        r_target = 0.3 * Hz
        solver = SiegertGradientDescent(tau_m=experiment.effective_time_constant_up_state.tau_eff(),
                                        theta=experiment.neuron_params.theta,
                                        v_reset=experiment.neuron_params.V_r,
                                        tau_ref=experiment.neuron_params.tau_rp, unit='mV')

        # Initial guesses (in mV) - adjusted for normalized form
        mu_0, sigma_0 = -56 * mV, 2.5 * mV

        ''' def find_parameters(self, r_target, mu_0, sigma_0,
                        learning_rate=1e-3 * (mV * second) ** 2, n_steps=5000, anneal_schedule=None,
                        tolerance=1e-8 * Hz ** 2):'''
        mu_sol, sigma_sol, history = solver.find_parameters(
            r_target=r_target,
            mu_0=mu_0,
            sigma_0=sigma_0,
            anneal_schedule=None,
            n_steps=10_000
        )

        plot_grad_descent(history=history, r_target=r_target)


if __name__ == '__main__':
    unittest.main()
