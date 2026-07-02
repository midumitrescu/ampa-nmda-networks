import unittest

from brian2 import second, kHz, nS, ms, Hz, Quantity
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

import numpy as np

from iteration_16.SimulateFittedSolutionWithCompartments import run_simulations_in_parallel_and_compare
from iteration_16.model import config_with_weak_synapses, config_with_intermediate_synapses, \
    ConductanceDiffusionSimulationConfig
from iteration_16.nmda_compartment_model import NMDASimulationWangCompartments
from iteration_16.simpy import load_solutions

def run_simulations_in_parallel_and_compare(base_config: ConductanceDiffusionSimulationConfig, k_s, plot_comparrison=True):
    #run_one = lambda k: NMDASimulationWangCompartments.run_and_plot(base_config.with_property(k_comp=k), title=gen_plot_title(base_config.with_property(k_comp=k)))
    run_one = lambda k: NMDASimulationWangCompartments.run(base_config.with_property(k_comp=k), detailed_statistics=plot_comparrison, title=f"Testing XOR for {k}", testing=False)

    results = Parallel(n_jobs=1)(
        delayed(run_one)(k) for k in k_s
    )

    result_by_k = dict(zip(k_s, results))

    if plot_comparrison:
        pass
    # think of a comparrison

    return result_by_k

def r_of_t(cfg:ConductanceDiffusionSimulationConfig, delta_peak: Quantity, time_peak: Quantity):

def generate_inhom_poisson_spike_trains(config: ConductanceDiffusionSimulationConfig, r):

    # Simulation parameters
    T = config.simulation_time / second  # seconds
    dt = config.simulation_time / ms

    # Time-dependent rate (Hz)
    def rate(t):
        return 20 + 15 * np.sin(2 * np.pi * t)

    # Maximum rate
    lambda_max = 35.0

    # Expected number of candidate spikes
    N = int(3 * lambda_max * T)

    isi = np.random.exponential(1 / lambda_max, size=N)
    candidate_times = np.cumsum(isi)
    candidate_times = candidate_times[candidate_times < T]

    # Thin the candidates
    bernoullis = np.random.rand(len(candidate_times))
    accepted =  bernoullis < (
            rate(candidate_times) / lambda_max
    )

    spike_times = candidate_times[accepted]

    print(spike_times)

class MyTestCase(unittest.TestCase):

    def test_simulate_with_fitted_config_first_solution(self):
        cfg = config_with_intermediate_synapses.with_property(N_E=2000, simulation_time=0.5 * second, seed=201, alpha_nmda = 0.25 * kHz, g_nmda_max=4 * 4.074575871229172 * nS)
        solutions = load_solutions(config=cfg, file_name="/home/md/Workspace/python/ampa-nmda-networks/src/iteration_16/solutions/solutions.txt")
        gr, gamma = solutions[0]

        low_inhibition_ratio = cfg.with_fitted_solution(gr=gr, gamma=gamma)

        k_s = [1, 2, 4, 8]
        run_simulations_in_parallel_and_compare(low_inhibition_ratio, k_s)

    def test_generation_of_inhomogenous_poisson_rate(self):
        cfg = config_with_intermediate_synapses

        T = cfg.simulation_time
        dt = cfg.dt

        t = np.arange(0, T, dt) * second

        r0 = 0.3 * Hz
        rmax = 15.0 * Hz

        t_peak = 200 * ms
        sigma = 50 * ms

        r_of_t = r0 + (rmax - r0) * np.exp(
            -(t - t_peak) ** 2 / (2 * sigma ** 2)
        )

        plt.plot(t / ms, r_of_t)
        plt.title(r"$r(t) = r_0 + (r_{max} - r_0)\cdot\exp{ - \frac{(t-t_{peak})^2}{\sigma^2}} $""\n"
                  r"$r_0=$"f"{r0 / Hz} Hz, "r"$r_{max}=$"f"{rmax / Hz} Hz, "r"$t_{peak}=$"f"{t_peak/ms} ms, "r"$\sigma=$"f"{sigma / ms} ms")
        plt.show()

        self.assertAlmostEqual(0.30493130063016693, r_of_t[0] / Hz)
        self.assertAlmostEqual(0.30000022522791175, r_of_t[-1] / Hz)

    def test_generation_of_one_inhomogenous_poisson_spiek_train(self):





if __name__ == '__main__':
    unittest.main()
