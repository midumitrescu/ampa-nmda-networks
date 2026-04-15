import unittest

import numpy as np
from brian2 import Hz, mV
from scipy.optimize import fsolve

from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients

def eq_system_r_mk801_and_gain(x, siegert_gradients: SiegertGradients, r_mk801, target_gain):
    mu, sigma = x
    print(f"Look for solution of system: {mu}, {sigma}")
    return np.array([
        siegert_gradients.firing_rate(mu * mV, sigma * mV) - r_mk801,
        siegert_gradients.d_rate_d_mu(mu * mV, sigma * mV) - target_gain
    ])

# ---- JACOBIAN ----
def J(x, siegert_gradient: SiegertGradients):
    mu, sigma = x
    return np.array([
        [siegert_gradient.d_rate_d_mu(mu * mV, sigma * mV), siegert_gradient.d_rate_d_sigma(mu * mV, sigma * mV)],
        [
            siegert_gradient.d_squared_rate_d_mu_squared(mu * mV, sigma * mV),
            siegert_gradient.d_squared_rate_d_mu_d_sigma(mu * mV, sigma * mV)
        ]
    ])

class MyTestCase(unittest.TestCase):
    def test_scipy_fsolve(self):
        siegert_gradients = SiegertGradients.default()
        r_mk801 = 0.05 * Hz
        r_control = 0.18 * Hz
        delta_mu = 0.7 * mV

        target_gain = (r_control - r_mk801) / delta_mu

        sol = fsolve(func=lambda x: eq_system_r_mk801_and_gain(x, siegert_gradients, r_mk801=r_mk801, target_gain=target_gain),
                     x0 = np.array([-55, 5]),
                     fprime=lambda x: J(x, siegert_gradient=siegert_gradients))

        print(sol)
        mu, sigma = sol

        print("Rate MK801 is: ", siegert_gradients.firing_rate(mu * mV, sigma * mV))
        print("Rate MK801 is: ", siegert_gradients.firing_rate((mu - 0.7) * mV, sigma * mV))
        print("Rate Control is: ", siegert_gradients.firing_rate(mu * mV + delta_mu, sigma * mV))

    def test_scipy_least_squares(self):
        from scipy.optimize import least_squares
        siegert_gradients = SiegertGradients.default()
        r_mk801 = 0.05 * Hz
        r_control = 0.18 * Hz
        delta_mu = 0.7 * mV

        target_gain = (r_control - r_mk801) / delta_mu

        def residuals(x):
            return eq_system_r_mk801_and_gain(x, siegert_gradients, r_mk801=r_mk801, target_gain=target_gain)

        sol = least_squares(residuals,  np.array([-55, 5]))
        print(sol)
        mu, sigma = sol.x

        print("Rate MK801 is: ", siegert_gradients.firing_rate(mu * mV, sigma * mV))

        print("Rate Control is: ", siegert_gradients.firing_rate(mu * mV + delta_mu, sigma * mV))

if __name__ == '__main__':
    unittest.main()
