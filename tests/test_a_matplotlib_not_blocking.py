import unittest

import matplotlib.pyplot as plt


class MatplotlibNotBlockingTestCase(unittest.TestCase):
    def test_matplotlib_graph_generation_is_non_blocking(self):
        plt.figure(figsize=(3, 3))
        plt.title("Nonblocking graph test")
        plt.show(block=False)
        plt.close()