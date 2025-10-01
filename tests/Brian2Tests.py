import unittest

import numpy as np
from brian2 import run, NeuronGroup, mV, StateMonitor, ms, \
    ufarad, siemens, cm
from numpy.testing import assert_array_almost_equal


class Brian2TestCases(unittest.TestCase):

    def test_exponential_relaxation_example(self):
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
                                  [-0.08, -0.075055, -0.07174, -0.069518, -0.068028, -0.06703,
                                   -0.066361, -0.065912, -0.065611, -0.06541])


if __name__ == '__main__':
    unittest.main()
