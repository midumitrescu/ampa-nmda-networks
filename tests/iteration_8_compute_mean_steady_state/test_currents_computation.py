import unittest

from brian2 import ms
import numpy as np

from Configuration import Experiment
from iteration_7_one_compartment_step_input.one_compartment_with_up_only import simulate_with_up_state_and_nmda
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control


class CurrentComputationTestCases(unittest.TestCase):
    '''
    Q_ampa = -6669.919309200741, Q_gaba = 1913.5144157111763, Q_nmda = -53.078547562775945
        Ratio NMDA from exitatory: 0.007895071319913852
    '''

    def test_current_integral(self):
        t_range = [0, 1_000]
        result = simulate_with_up_state_and_nmda(palmer_control.with_properties({
            "t_range": t_range,
            # "theta": -40,
            Experiment.KEY_CURRENTS_TO_RECORD: ["I_nmda", "I_ampa", "I_gaba"]
        }))
        dt = palmer_control.sim_clock / ms
        q_ampa = np.sum(result.currents.I_ampa[0] * dt)
        q_gaba = np.sum(result.currents.I_gaba[0] * dt)
        q_nmda = np.sum(result.currents.I_nmda[0] * dt)
        print(f"Q_ampa = {q_ampa}, Q_gaba = {q_gaba}, Q_nmda = {q_nmda}")
        print(f"Ratio NMDA from exitatory: {q_nmda / (q_nmda + q_ampa)}")

        self.assertAlmostEqual(-667.921226738323, q_ampa, delta=0.00001)
        self.assertAlmostEqual(187.8352876776553, q_gaba, delta=0.0000001)
        self.assertAlmostEqual(-5.085106526062978, q_nmda, delta=0.0000001)

        self.assertAlmostEqual(-667.921226738323, result.currents.q_ampa, delta=0.00001)
        self.assertAlmostEqual(187.8352876776553, result.currents.q_gaba, delta=0.0000001)
        self.assertAlmostEqual(-5.085106526062978, result.currents.q_nmda, delta=0.0000001)


if __name__ == '__main__':
    unittest.main()
