import unittest
import numpy

import matplotlib.pyplot as plt
import numpy as np


class MyTestCase(unittest.TestCase):

    @staticmethod
    def test_simple_plotting_works():

        plt.plot(np.linspace(0, 1000), np.linspace(0, 1000))
        plt.title("Test")
        plt.show()

    @staticmethod
    def test_latex_plotting_works():
        plt.rcParams['text.usetex'] = True
        plt.plot(np.linspace(0, 1000), np.linspace(0, 1000))
        plt.title(r'$\gamma$, $\frac{\nu_1}{\mu_2}$')
        plt.show()


if __name__ == '__main__':
    unittest.main()
