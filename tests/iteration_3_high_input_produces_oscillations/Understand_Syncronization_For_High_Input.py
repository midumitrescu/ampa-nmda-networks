import unittest

import numpy as np
from brian2 import ms, msiemens, cm, siemens

from Plotting import plot_non_blocking
from iteration_3_high_input_produces_oscillations.high_input_produces_bifurcation import sim, make_experiment_work_with_N_E_and_C_E_zero
import matplotlib.pyplot as plt


class Understand_Syncronization_For_High_Input(unittest.TestCase):

    def test_simulation_without_self_synapses_should_show_only_poisson_rates(self):
        for index, multiplication in enumerate(np.linspace(0.01, 0.5, num=3) * 100):
            np.random.seed(0)
            no_feedback_synapses_config = {
                "panel": f"Iteration {index}: Testing neurons under high input, plot smoothened, sim clock = 0.05 ms",
                "g": 0,
                "sim_time": 1500,
                "sim_clock": 0.05 * ms,
                "nu_ext_over_nu_thr": 1 * multiplication,
                "t_range": [100, 120],
                "voltage_range": [-70, -30],
                "epsilon": 0,
                "C_ext": 1000,
                "smoothened_rate_width": 0.5*ms
            }

            no_feedback_synapses = make_experiment_work_with_N_E_and_C_E_zero(no_feedback_synapses_config)

            np.random.seed(0)
            sim(experiment=no_feedback_synapses)
            # make sure that the network fires, when no inhibition is present
            # this is achieved by taking gamma = 0
            plot_non_blocking()
            np.random.seed(0)

            no_feedback_plot_rate_raw = make_experiment_work_with_N_E_and_C_E_zero(no_feedback_synapses_config)
            no_feedback_plot_rate_raw.plot_params.plot_smoothened_rate = False
            no_feedback_plot_rate_raw.plot_params.panel = f"Iteration {index}: Testing neurons under high input, plot rate raw, sim clock = 0.05 ms"

            sim(experiment=no_feedback_plot_rate_raw)
            plot_non_blocking()

            no_feedback_plot_rate_raw_increased_clock = make_experiment_work_with_N_E_and_C_E_zero(no_feedback_synapses_config)
            no_feedback_plot_rate_raw_increased_clock.sim_clock = 0.01 * ms
            np.random.seed(0)
            no_feedback_plot_rate_raw_increased_clock.plot_params.panel = f"Iteration {index}: Testing neurons under high input, plot rate raw, sim clock = 0.01 ms"
            sim(experiment=no_feedback_plot_rate_raw_increased_clock)
            # make sure that the network fires, when no inhibition is present
            # this is achieved by taking gamma = 0
            plot_non_blocking()

    def test_with_different_g_l(self):
        for index, multiplication in enumerate(np.linspace(0.1, 2, num=3)):
            np.random.seed(0)
            no_feedback_synapses_config = {

                "sim_time": 1500,
                "sim_clock": 0.05 * ms,

                "g": 0,
                "nu_ext_over_nu_thr": 1 * multiplication,
                "epsilon": 0,
                "C_ext": 1000,
                "g_L":0.004,

                "panel": f"Iteration {index}: Testing neurons under high input, plot smoothened, sim clock = 0.05 ms",
                "t_range": [100, 120],
                "voltage_range": [-70, -30],
                "smoothened_rate_width": 0.5*ms
            }

            no_feedback_synapses = make_experiment_work_with_N_E_and_C_E_zero(no_feedback_synapses_config)

            np.random.seed(0)
            sim(experiment=no_feedback_synapses)
            # make sure that the network fires, when no inhibition is present
            # this is achieved by taking gamma = 0
            plot_non_blocking()

            np.random.seed(0)

            no_feedback_plot_rate_raw = make_experiment_work_with_N_E_and_C_E_zero(no_feedback_synapses_config)
            no_feedback_plot_rate_raw.plot_params.plot_smoothened_rate = False
            no_feedback_plot_rate_raw.plot_params.panel = f"Iteration {index}: Testing neurons under high input, plot rate raw, sim clock = 0.05 ms"

            sim(experiment=no_feedback_plot_rate_raw)
            plot_non_blocking()

            no_feedback_plot_rate_raw_increased_clock = make_experiment_work_with_N_E_and_C_E_zero(no_feedback_synapses_config)
            no_feedback_plot_rate_raw_increased_clock.sim_clock = 0.01 * ms
            np.random.seed(0)
            no_feedback_plot_rate_raw_increased_clock.plot_params.panel = f"Iteration {index}: Testing neurons under high input, plot rate raw, sim clock = 0.01 ms"
            sim(experiment=no_feedback_plot_rate_raw_increased_clock)
            # make sure that the network fires, when no inhibition is present
            # this is achieved by taking gamma = 0
            plot_non_blocking()





if __name__ == '__main__':
    unittest.main()
