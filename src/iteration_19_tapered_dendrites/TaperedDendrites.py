import unittest

import numpy as np
from brian2 import Quantity
from scipy.integrate import quad

def G_inf(x, x0, t, tau,
          Phi,
          chi,
          r_a,
          beta,
          lambda_of_x):
    """
    Green's function (Eq. 32)

    Parameters
    ----------
    x, x0 : float
        Observation and injection locations.
    t : float
        Time (must be > 0).
    tau : float
        Membrane time constant.
    Phi : callable
        Phi(x, lambda).
    chi : callable
        Characteristic admittance χ(x).
    r_a : callable
        Axial resistance.
    beta : callable
        β(μ(x)).
    lambda_of_x : callable
        Space constant λ(x).
    """

    if t <= 0:
        return 0.0

    # Spatial integral
    L, _ = quad(lambda y: 1.0 / lambda_of_x(y), x0, x)

    prefactor = (
            Phi(x0, lambda_of_x) * chi(x0) * r_a
            / np.sqrt(4 * np.pi * tau * t)
    )

    exponent = (
        -beta(x) * t / tau
        -tau * L**2 / (4 * t)
    )

    return prefactor * np.exp(exponent)

lambda0 = 2.0
tau = 20.0

def lam(x, rho_of_x, r_of_x, r_a: Quantity, g_L:Quantity):
    # which units on quantities!?
    return np.sqrt(r_of_x(x)**2 / (2 * r_a * g_L * rho_of_x(x)))

def beta(x):
    return 1.0

def chi(x):
    return 1.0

def Phi(x, lam):
    return 1.0

class TaperedDendritesCase(unittest.TestCase):
    def test_one_pulse(self):

        r_a = 1 # which unit!? Axial resistance
        x0 = 0.0

        xs = np.linspace(-5, 5, 200)

        G = [G_inf(x, x0, 2.0, tau,
                   Phi, chi, r_a, beta, lam)
             for x in xs]


if __name__ == '__main__':
    unittest.main()
