import unittest

from brian2 import mV, Hz
from brian2 import second, ms
from joblib import Parallel, delayed

from iteration_12_transfer_function_of_lif_neurons.config import default_diffusion_lif_config, DiffusionLIFConfig
from iteration_15_OU_process.brian_lif_diffusion import Brian2DiffusionSimulation, NativeDiffusionSimulation
from iteration_15_OU_process.serializer import save_on_disk


def simulate_and_plot(dt=0.1 * ms, testing: bool = True,
                      mu=-47.61595645 * mV, sigma=1.90531046 * mV,
                      delta_v=0.7 * mV, target_rates=[0.05 * Hz, 0.18 * Hz],
                      lif_config: DiffusionLIFConfig = default_diffusion_lif_config,
                      seed=None, exp_label: str = "", sim_method="heun"):

    diffusion_simulation = Brian2DiffusionSimulation(seed=seed, dt=dt, mu=mu, sigma=sigma, delta_v=delta_v,
                                           target_rates=target_rates, lif_config=lif_config, sim_method=sim_method, testing=testing)
    t, v_s, spike_times = diffusion_simulation.simulate_two_diffusion_approx()
    file_names = save_on_disk(t, v_s, spike_times, diffusion_simulation=diffusion_simulation, lif_config=lif_config, exp_label=exp_label, testing=testing)

    if testing:
        plot_start = 0 * second
        plot_end = diffusion_simulation.T
    else:
        plot_start = 10 * second
        plot_end = 15 * second

    diffusion_simulation.plot_diffusion_approx(t=t, v_s=v_s, spike_times=spike_times,
                          plot_start=plot_start, plot_end=plot_end,
                          exp_label=exp_label)

    return file_names

def simulate_native_and_plot(dt=0.1 * ms, testing: bool = True,
                      mu=-47.61595645 * mV, sigma=1.90531046 * mV,
                      delta_v=0.7 * mV, target_rates=[0.05 * Hz, 0.18 * Hz],
                      lif_config: DiffusionLIFConfig = default_diffusion_lif_config,
                      seed=None, exp_label: str = ""):

    diffusion_simulation = NativeDiffusionSimulation (seed=seed, dt=dt, mu=mu, sigma=sigma, delta_v=delta_v,
                                           target_rates=target_rates, lif_config=lif_config, testing=testing)
    t, v_s, spike_times = diffusion_simulation.simulate_two_diffusion_approx()
    file_names = save_on_disk(t, v_s, spike_times, diffusion_simulation=diffusion_simulation, lif_config=lif_config, exp_label=exp_label, testing=testing)

    if testing:
        plot_start = 0 * second
        plot_end = diffusion_simulation.T
    else:
        plot_start = 10 * second
        plot_end = 15 * second

    diffusion_simulation.plot_diffusion_approx(t=t, v_s=v_s, spike_times=spike_times,
                          plot_start=plot_start, plot_end=plot_end,
                          exp_label=exp_label)

    return file_names


class Brian2LIFDifussion(unittest.TestCase):

    def test_run_lif_diffusion_approximation_in_separate_methods(self):
        simulate_and_plot(testing=True, lif_config=default_diffusion_lif_config,
                          mu=-47.61595645 * mV, sigma=1.90531046 * mV, delta_v=0.7 * mV,
                          exp_label="Just a test")

    def test_diffusion_approx(self):

        testing = False
        for sim_method in ["heun"]:
            Parallel(n_jobs=-3, prefer="processes")(
                delayed(simulate_and_plot)(testing=testing, lif_config=default_diffusion_lif_config,
                                           mu=-47.61595645 * mV, sigma=1.90531046 * mV, delta_v=0.7 * mV,
                                           exp_label=f"Brian2 Simulate diffusion process {sim_method}", seed=seed, sim_method=sim_method,
                                           )
                for seed in range(1, 20)
            )

            # def test_run_no_firing(self):

            not_firing_config = default_diffusion_lif_config.with_property(DiffusionLIFConfig.KEY_NEURON_THRESHOLD, 100)
            Parallel(n_jobs=-3, prefer="processes")(
                delayed(simulate_and_plot)(testing=testing, lif_config=not_firing_config,
                                           mu=-47.61595645 * mV, sigma=1.90531046 * mV, delta_v=0.7 * mV,
                                           exp_label=f"Brian2 Check OU process μ, σ {sim_method}", seed=seed, sim_method=sim_method,
                                           )
                for seed in range(1, 20)
            )

    def test_diffusion_process_native_implementation(self):

        testing = False
        Parallel(n_jobs=-3, prefer="processes")(
            delayed(simulate_native_and_plot)(testing=testing, lif_config=default_diffusion_lif_config,
                                       mu=-47.61595645 * mV, sigma=1.90531046 * mV, delta_v=0.7 * mV,
                                       exp_label="Simulate diffusion process native", seed=seed
                                       )
            for seed in range(1, 20)
        )


        not_firing_config = default_diffusion_lif_config.with_property(DiffusionLIFConfig.KEY_NEURON_THRESHOLD, 100)
        Parallel(n_jobs=-3, prefer="processes")(
            delayed(simulate_native_and_plot)(testing=testing, lif_config=not_firing_config,
                                       mu=-47.61595645 * mV, sigma=1.90531046 * mV, delta_v=0.7 * mV,
                                       exp_label="Check OU process μ, σ native", seed=seed,
                                       )
            for seed in range(1, 20)
        )



if __name__ == '__main__':
    unittest.main()
