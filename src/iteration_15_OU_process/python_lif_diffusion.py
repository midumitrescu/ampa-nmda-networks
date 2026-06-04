import brian2
import matplotlib.pyplot as plt
import numpy as np
from brian2 import NeuronGroup, StateMonitor, Quantity
from brian2 import run, mV, SpikeMonitor, Hz
from brian2 import second, ms
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d

from Plotting import show_plots_non_blocking
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients
from iteration_12_transfer_function_of_lif_neurons.config import default_diffusion_lif_config, DiffusionLIFConfig
from iteration_15_OU_process.serializer import save_on_disk, exp_label_to_folder_name


def simulate_diffusion_process(seed: None, T: Quantity, mu=-47.61595645 * mV, sigma=1.90531046 * mV,
                               delta_v=0.7 * mV, target_rates=[0.05 * Hz, 0.18 * Hz],
                               lif_config: DiffusionLIFConfig = default_diffusion_lif_config, dt=0.1 * ms, sim_method="heun"):
    if seed is not None:
        brian2.devices.device.seed(seed)

    return simulate_two_diffussion_approx(T, mu=mu, sigma=sigma, delta_v=delta_v, dt=dt, target_rates=target_rates,
                                          lif_config=lif_config, sim_method=sim_method)


def simulate_two_diffussion_approx(T, dt=0.1 * ms, mu=-47.61595645 * mV, sigma=1.90531046 * mV, delta_v=0.7 * mV,
                                   lif_config=default_diffusion_lif_config,
                                   target_rates=[0.05 * Hz, 0.18 * Hz], exp_labels=["MK801", "Control"], sim_method="heun"):
    V_r = lif_config.V_r
    theta = lif_config.theta
    mean = [mu, mu + delta_v]
    mean_no_mv_units = np.array([mean[0] / mV, mean[1] / mV])
    sigma_no_mv_units = sigma / mV
    N = NeuronGroup(
        2,
        """
        tau : second
        mean: volt
        dv/dt = -(v-mean)/tau + sqrt(2*tau**-1)*sigma*xi : volt
        """,
        method=sim_method,
        threshold="v >= theta",
        reset="v = V_r",
        refractory=lif_config.tau_rp,
        dt=dt)
    N.mean = mean
    N.tau = lif_config.tau_m
    N.v[:] = lif_config.V_r
    M = StateMonitor(N, ["v"], record=True)
    spikemon = SpikeMonitor(N)
    run(T)

    v_s = M.v / mV
    t = M.t / second
    spike_times = [None] * len(target_rates)
    for neuron_id in [0, 1]:
        spike_times[neuron_id] = spikemon.all_values()['t'][neuron_id] / second
    return t, v_s, spike_times
