import unittest

from brian2 import second, kHz, nS, ms, Hz, Quantity
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from numpy.testing import assert_almost_equal

import numpy as np

from iteration_16.SimulateFittedSolutionWithCompartments import run_simulations_in_parallel_and_compare
from iteration_16.model import config_with_weak_synapses, config_with_intermediate_synapses, \
    ConductanceDiffusionSimulationConfig
from iteration_16.nmda_compartment_model import NMDASimulationWangCompartments
from iteration_16.simpy import load_solutions
from iteration_18_xor_with_multi_compartments import sequences


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
    pass

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

def pop_rate(spikes, neuron_ids, cfg: ConductanceDiffusionSimulationConfig, sigma=10.0 * ms):
    # -------------------------
    # 2. Population rates
    # -------------------------
    T = cfg.simulation_time
    dt = cfg.dt
    bins = np.arange(0, T + dt, dt)
    all_spikes = []
    for i in neuron_ids:
        s = spikes[i]
        s = s[np.isfinite(s)]
        all_spikes.append(s)

    all_spikes = np.concatenate(all_spikes) if len(all_spikes) else np.array([])
    counts, _ = np.histogram(all_spikes, bins=bins)

    # Gaussian kernel
    width = int(5 * sigma / dt)
    x = np.arange(-width, width + 1)
    kernel = np.exp(-0.5 * (x * dt / sigma) ** 2)
    kernel /= kernel.sum()

    rate = np.convolve(counts, kernel, mode='same')
    rate = rate / (len(neuron_ids) * dt) # Hz
    return bins[:-1], rate

def plot_raster_and_rates(spikes, cfg: ConductanceDiffusionSimulationConfig, sigma=10.0 * ms):

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True
    )

    if len(spikes) > 100:
        # -------------------------
        # 1. Raster plot (E + I)
        # -------------------------
        e_show = 80
        i_show = 20
    else:
        e_show = 8
        i_show = 2

    # Excitatory (RED)
    for i in range(e_show):
        t = spikes[i]
        t = t[np.isfinite(t)]
        ax1.scatter(t, np.ones_like(t) * i, s=1, color='red')

    # Inhibitory (BLUE)
    offset = -i_show
    for j in range(i_show):
        idx = cfg.N_E + j
        t = spikes[idx]
        t = t[np.isfinite(t)]
        ax1.scatter(t, np.ones_like(t) * (offset + j), s=1, color='blue')

    ax1.set_ylabel("Neuron index")
    ax1.set_title("Raster plot (Excitatory = red, Inhibitory = blue)")
    ax1.axhline(0, color='gray', linewidth=0.5)


    e_ids = np.arange(0, cfg.N_E)
    i_ids = np.arange(cfg.N_E, spikes.shape[0])

    t_e, r_e = pop_rate(spikes, e_ids, cfg=cfg, sigma=sigma)
    t_i, r_i = pop_rate(spikes, i_ids, cfg=cfg, sigma=sigma)

    # Excitatory (RED)
    ax2.plot(t_e, r_e, color='red', label='Excitatory')

    # Inhibitory (BLUE)
    ax2.plot(t_i, r_i, color='blue', label='Inhibitory')

    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Rate (Hz)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

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

    def test_plot_poisson_trains(self):
        np.random.seed(200)
        cfg = config_with_intermediate_synapses.with_property(N_E=1600, N_I=400, simulation_time=0.5 * second, seed=201,
                                                              alpha_nmda=0.25 * kHz,
                                                              g_nmda_max=4 * 4.074575871229172 * nS)
        solutions = load_solutions(config=cfg,
                                   file_name="/home/md/Workspace/python/ampa-nmda-networks/src/iteration_16/solutions/solutions.txt")
        gr, gamma = solutions[0]

        high_inhibition_ration = cfg.with_fitted_solution(gr=gr, gamma=gamma)
        print(f"gr = {gr} nS/Hz, gamma = {gamma}")
        print(f"E rate: {high_inhibition_ration.r_e}, I rate: {high_inhibition_ration.r_i}")
        rates_e = np.repeat(high_inhibition_ration.r_e, high_inhibition_ration.N_E)
        rates_i = np.repeat(high_inhibition_ration.r_i, high_inhibition_ration.N_I)

        spikes = sequences.build_rate_seq_slow(np.concatenate([rates_e, rates_i]), 0, cfg.simulation_time / second)
        self.assertEqual((2000, 19067), spikes.shape)

        plot_raster_and_rates(spikes,  T=cfg.simulation_time, n_e = cfg.N_E, dt = cfg.dt)

    def test_rate_should_be_approximately_correct(self):
        np.random.seed(200)
        T_test = 1E6
        cfg = config_with_intermediate_synapses.with_property(N_E=3, N_I=2, simulation_time=T_test * second, seed=201, r_e=50 * Hz, r_i = 20 * Hz)

        rates_e = np.repeat(cfg.r_e, cfg.N_E)
        rates_i = np.repeat(cfg.r_i, cfg.N_I)

        spikes = sequences.build_rate_seq_parallel(np.concatenate([rates_e, rates_i]), cfg=cfg)

        number_of_spikes = np.isfinite(spikes).sum(axis=1)
        rates = number_of_spikes / T_test

        assert_almost_equal(rates, [50, 50, 50, 20, 20], decimal=2)

    def test_rate_which_method_is_faster_for_poisson_train(self):
        np.random.seed(200)
        T_test = 1E6
        cfg = config_with_intermediate_synapses.with_property(N_E=3, N_I=2, simulation_time=T_test * second, seed=201, r_e=50 * Hz, r_i = 20 * Hz)

        rates_e = np.repeat(cfg.r_e, cfg.N_E)
        rates_i = np.repeat(cfg.r_i, cfg.N_I)

        import time

        rates = np.concatenate([rates_e, rates_i])

        t0 = time.perf_counter()
        spikes_0 = sequences.build_rate_seq_slow(rates, T= cfg.simulation_time)
        slower = time.perf_counter() - t0

        t0 = time.perf_counter()
        spikes_1 = sequences.build_rate_seq_fast(rates, T=cfg.simulation_time)
        faster = time.perf_counter() - t0

        t0 = time.perf_counter()
        spikes_parallel = sequences.build_rate_seq_parallel(rates, cfg=cfg)
        parallel = time.perf_counter() - t0

        print(f"slow:     {slower:.3f} s")
        print(f"fast: {faster:.3f} s")
        print(f"parallel: {parallel:.3f} s")
        print(f"Speedup:  {slower / faster:.2f}×")
        print(f"Speedup parallel:  {faster / parallel:.2f}×")
        self.assertGreater(faster/parallel, 1.3, "parallel has to bring a speedup")

    def test_plotting_in_dev(self):
        cfg = config_with_intermediate_synapses.with_property(N_E=16, N_I=4, simulation_time=0.5 * second, seed=201,
                                                              alpha_nmda=0.25 * kHz,
                                                              g_nmda_max=4 * 4.074575871229172 * nS,
                                                              r_e=50 * Hz, r_i = 20 * Hz)
        rates_e = np.repeat(cfg.r_e, cfg.N_E)
        rates_i = np.repeat(cfg.r_i, cfg.N_I)

        spikes = sequences.build_rate_seq_parallel(np.concatenate([rates_e, rates_i]), cfg=cfg)

        plot_raster_and_rates(spikes, cfg=cfg, sigma=20 * ms)





if __name__ == '__main__':
    unittest.main()
