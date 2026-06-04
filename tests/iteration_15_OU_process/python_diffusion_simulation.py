import unittest

from brian2 import ms

from iteration_12_transfer_function_of_lif_neurons.config import default_diffusion_lif_config
from iteration_15_OU_process.brian_lif_diffusion import NativeDiffusionSimulation


class MyTestCase(unittest.TestCase):
    def test_simulate_one_brian_diffusion_process(self):
        lif_config = default_diffusion_lif_config
        seed = 9

        dt = 0.1 * ms
        diffusion_simulation = NativeDiffusionSimulation(dt=dt, seed=seed, testing=True, lif_config=lif_config)
        t, v_s, spike_times = diffusion_simulation.simulate_two_diffusion_approx()

        diffusion_simulation.plot_diffusion_approx(v_s=v_s, t=t, spike_times=spike_times,
                                                   plot_start=0 * ms, plot_end=diffusion_simulation.T,
                                                   exp_label="Native test")

        self.assertEqual(100_000, len(t))
        means = v_s.mean(axis=1)
        print(means)
        self.assertAlmostEqual(-47.69705304, means[0])
        self.assertAlmostEqual(-47.01609717, means[1])

        self.assertEqual(0, len(spike_times[0]))
        self.assertEqual(1, len(spike_times[1]))


if __name__ == '__main__':
    unittest.main()
