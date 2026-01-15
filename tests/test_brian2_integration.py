import unittest

import pytest
from loguru import logger

import numpy as np
from brian2 import run, NeuronGroup, mV, StateMonitor, ms, \
    ufarad, siemens, cm, devices, defaultclock
from numpy.testing import assert_array_almost_equal


class Brian2TestCases(unittest.TestCase):

    def setUp(self):
        logger.info("Brian2 seed to 0")
        devices.device.seed(0)

    def test_exponential_relaxation_example(self):
        devices.device.seed(0)
        defaultclock.dt = 0.05 * ms
        C = 1 * ufarad / cm ** 2
        g_L = 0.004 * siemens / cm ** 2

        E_Leak = -65 * mV
        V_r = E_Leak
        theta = -55 * mV

        neuron = NeuronGroup(1,
                             """
                             dv/dt = 1/C * (-g_L * (v-E_Leak)): volt                          """,
                             threshold="v > theta",
                             reset="v = V_r",
                             method="exact")

        neuron.v = -80 * mV
        v_monitor = StateMonitor(source=neuron, variables="v", record=True)
        run(200 * ms, report='text')

        assert_array_almost_equal(np.array(v_monitor.v[0][0:10]),
                                  [-0.08, -0.077281, -0.075055, -0.073232, -0.07174, -0.070518,
                                    - 0.069518, -0.068699, -0.068028, -0.067479])


if __name__ == '__main__':
    unittest.main()
