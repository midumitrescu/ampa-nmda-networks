import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import NeuronGroup, StateMonitor
from brian2 import run, mV, SpikeMonitor, Hz
from brian2 import second, ms

from iteration_12_transfer_function_of_lif_neurons.config import default_diffusion_lif_config
from iteration_15_OU_process.lif_difussion_check import filter_spikes_in_time_window


class OUProcessTestCases(unittest.TestCase):

    @staticmethod
    def test_empty_array():
        spikes = np.array([])

        result = filter_spikes_in_time_window(spikes, 0.0, 1.0)

        np.testing.assert_array_equal(result, np.array([]))

    @staticmethod
    def test_full_range():
        spikes = np.array([0.1, 0.3, 0.5, 0.7])

        result = filter_spikes_in_time_window(spikes, 0.0, 1.0)

        np.testing.assert_array_equal(
            result,
            np.array([0.1, 0.3, 0.5, 0.7]),
        )

    @staticmethod
    def test_partial_overlap():
        spikes = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        result = filter_spikes_in_time_window(spikes, 0.25, 0.75)

        np.testing.assert_array_equal(
            result,
            np.array([0.3, 0.5, 0.7]),
        )

    @staticmethod
    def test_no_overlap():
        spikes = np.array([1.0, 2.0, 3.0])

        result = filter_spikes_in_time_window(spikes, 4.0, 5.0)

        np.testing.assert_array_equal(result, np.array([]))

    @staticmethod
    def test_boundary_inclusion():
        spikes = np.array([0.1, 0.2, 0.3, 0.4])

        result = filter_spikes_in_time_window(spikes, 0.2, 0.3)

        np.testing.assert_array_equal(
            result,
            np.array([0.2, 0.3]),
        )


if __name__ == '__main__':
    unittest.main()
