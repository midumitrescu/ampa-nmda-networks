import unittest

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
from brian2 import mV, Hz, have_same_dimensions

from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients, I_mu_sigma
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control


class SiegertTestCase(unittest.TestCase):

    def test_check_derivative_d_rate_d_mu(self):

        siegert_gradient = SiegertGradients.for_experiment(palmer_control)

        sigma = 3 * mV
        mus = np.linspace(-65, -35, 500) * mV
        delta = 1e-5 * mV

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

        error = np.abs(analytical - numerical)

        self.assertLess(np.max(np.abs(rel_error)), 1E-8)

        assert_allclose(error, 0, atol=1E-8)
        assert_allclose(analytical, numerical, atol=1e-9)

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

        self.assertLess(np.max(np.abs(rel_error)), 1E-8)

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

    def test_check_second_derivative_d_sq_r_d_mu_sq(self):

        siegert_gradient = SiegertGradients.default()

        sigma = 1.9 * mV
        mus = np.linspace(-65, -35, 1001) * mV
        delta = 1e-5 * mV  # 1e-3 mV as requested

        analytical = np.zeros_like(mus / mV)
        numerical = np.zeros_like(mus / mV)
        rel_error = np.zeros_like(mus / mV)

        for index, mu in enumerate(mus):
            # --- analytical derivative ---
            d2r_dm2_analytical = siegert_gradient.d_squared_rate_d_mu_squared(
                mu_v=mu,
                sigma_v=sigma
            )
            assert have_same_dimensions(d2r_dm2_analytical, 1 * Hz / mV **2)

            analytical[index] = d2r_dm2_analytical / Hz * mV **2

            rp = siegert_gradient.firing_rate(mu + delta, sigma)
            r0 = siegert_gradient.firing_rate(mu, sigma)
            rm = siegert_gradient.firing_rate(mu - delta, sigma)

            numerical_d2_dr2 = (rp - 2 * r0 + rm) / (delta ** 2)
            numerical[index] = numerical_d2_dr2 / Hz * mV**2

            dr_a = float(d2r_dm2_analytical / Hz * mV **2)
            dr_n = float(numerical_d2_dr2 / Hz * mV **2)

            if abs(dr_n) > 1e-12:
                rel_error[index] = (dr_a - dr_n) / dr_n
            else:
                rel_error[index] = 0.0

        #self.assertLess(np.max(np.abs(rel_error)), 1E-8)

        # --- plotting ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

        ax1.plot(mus / mV, analytical, label="Analytical", linewidth=2)
        ax1.plot(mus / mV, numerical, "--", label=rf"Numerical ($\Delta$={delta})")
        ax1.set_ylabel(r"$\mu$ [mV]")
        ax1.set_title(r"$\frac{d^2 r}{d \mu^2}(\mu, \sigma$="f"{sigma / mV} mV)"" Second derivative")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(mus / mV, 100 * rel_error)
        ax2.set_ylabel("Relative error [%]")
        ax2.set_xlabel(r"$\mu$ (mV)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        #print("Max abs relative error (%):", np.max(np.abs(100 * rel_error)))


if __name__ == '__main__':
    unittest.main()
