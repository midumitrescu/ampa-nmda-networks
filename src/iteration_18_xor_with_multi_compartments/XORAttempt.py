import unittest

from brian2 import second

from iteration_16.SimulateFittedSolutionWithCompartments import run_simulations_in_parallel_and_compare
from iteration_16.model import config_with_weak_synapses
from iteration_16.simpy import load_solutions


class MyTestCase(unittest.TestCase):

    def test_simulate_with_fitted_config_first_solution(self):
        cfg = config_with_weak_synapses.with_property(N_E=20, simulation_time=0.5 * second, seed=200, k_comp=4)
        solutions = load_solutions(config=cfg, file_name="/home/md/Workspace/python/ampa-nmda-networks/src/iteration_16/solutions/solutions.txt")
        gr, gamma = solutions[0]

        low_inhibition_ratio = cfg.with_fitted_solution(gr=gr, gamma=gamma)

        k_s = [1, 10, 500, low_inhibition_ratio.N_E]
        run_simulations_in_parallel_and_compare(low_inhibition_ratio, k_s)


if __name__ == '__main__':
    unittest.main()
