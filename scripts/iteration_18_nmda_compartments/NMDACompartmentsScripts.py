import math
import unittest

from brian2 import Hz, mV, second
from joblib import Parallel, delayed

from iteration_16.Simulate_K_NMDA_Compartments import plot_k_sweep_results
from iteration_16.model import calibrated_configuration
from iteration_16.nmda_compartment_model import NMDASimulationWangCompartments


class NMDAWithCompartmentScripts(unittest.TestCase):

    def test_compare_many_compartments_to_only_one(self):

        k_s = [1, 2, 5, 10]
        config = calibrated_configuration.with_property(r_e=2 * Hz, r_i=2 * Hz, N_E=10, N_I=5, seed=200, simulation_time=2 * second, e_L=-45 * mV)
        run_one = lambda k: NMDASimulationWangCompartments.run(config, k=k)

        results = Parallel(n_jobs=len(k_s))(
            delayed(run_one)(k) for k in k_s
        )

        result_by_k = dict(zip(k_s, results))

        plot_k_sweep_results(result_by_k, k_s, config)

    def test_compare_many_compartments_to_only_one_with_meanfield_scaling(self, scaling = math.sqrt):

        k_s = [1, 2, 5, 10]
        config = calibrated_configuration.with_property(r_e=2 * Hz, r_i=2 * Hz, N_E=10, N_I=5, seed=200, simulation_time=2 * second, e_L=-45 * mV)
        w_x_1_compartment = config.w_x
        run_one = lambda k: NMDASimulationWangCompartments.run(config.with_property(w_x = w_x_1_compartment / scaling(k)), k=k)

        results = Parallel(n_jobs=len(k_s))(
            delayed(run_one)(k) for k in k_s
        )

        result_by_k = dict(zip(k_s, results))

        plot_k_sweep_results(result_by_k, k_s, config)


if __name__ == '__main__':
    unittest.main()
