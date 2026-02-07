import sys
import unittest

from loguru import logger

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_8_compute_mean_steady_state.models_and_configs import wang_recurrent_config
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_steady_state


class TestsForSiegertScripts(unittest.TestCase):
    def test_compute_E_0_using_steady_state(self):
        base = Experiment(wang_recurrent_config)
        steady_up_state_results = sim_steady_state(base, state=base.network_params.up_state)

        self.assertAlmostEqual(-48.75341763069946, steady_up_state_results.v_steady)
        self.assertAlmostEqual(15.999999999999934, steady_up_state_results.g_e_steady)
        self.assertAlmostEqual(7.999999999999916, steady_up_state_results.g_i_steady)
        self.assertAlmostEqual(0.02220431199227894, steady_up_state_results.g_nmda_steady)
        self.assertAlmostEqual(0.19999999999999946, steady_up_state_results.x_nmda_steady)
        self.assertAlmostEqual(0.9090909090908988, steady_up_state_results.s_nmda_steady)


if __name__ == '__main__':
    unittest.main()
