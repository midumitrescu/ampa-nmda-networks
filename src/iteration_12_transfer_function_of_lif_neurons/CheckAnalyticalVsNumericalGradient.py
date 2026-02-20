import unittest

from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SigertGradientDescent


# Quick verification of the derivative sign
def numerical_gradient_mu(solver, mu, sigma_V, r_target, eps=1e-6):
    """Compute numerical gradient for verification"""
    F1 = solver.firing_rate(mu + eps, sigma_V)
    F2 = solver.firing_rate(mu - eps, sigma_V)
    return (F1 - F2) / (2 * eps)



class MyTestCase(unittest.TestCase):
    def test_something(self):
        # Test with sample values
        solver = SigertGradientDescent()
        mu, sigma = 0.02, 0.005

        # Analytical gradient (using corrected formula)
        grad_analytical = solver.gradient_mu(mu, sigma)

        # Numerical gradient
        grad_numerical = numerical_gradient_mu(solver, mu, sigma, None)

        print(f"Analytical: {grad_analytical:.6e}")
        print(f"Numerical:  {grad_numerical:.6e}")
        print(f"Relative error: {abs(grad_analytical - grad_numerical) / abs(grad_numerical):.2e}")


if __name__ == '__main__':
    unittest.main()
