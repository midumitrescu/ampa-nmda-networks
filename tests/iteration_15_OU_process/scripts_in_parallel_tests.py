import unittest
from pathlib import Path

from brian2 import mV
from joblib import Parallel, delayed

from Plotting import get_or_create_file_base_dir
from iteration_12_transfer_function_of_lif_neurons.config import default_diffusion_lif_config, DiffusionLIFConfig
from iteration_15_OU_process.lif_difussion_check import simulate_and_plot
from iteration_15_OU_process.serializer import load_from_disk, load_aggregate


class RunOUProcessInParallelTests(unittest.TestCase):
    def test_diffusion_process_can_run_in_parallel(self):
        files = Parallel(n_jobs=2, prefer="processes")(
            delayed(simulate_and_plot)(testing=True, lif_config=default_diffusion_lif_config,
                                       mu=-47.61595645 * mV, sigma=1.90531046 * mV, delta_v=0.7 * mV,
                                       exp_label="Simulate diffusion process test", seed=seed)
            for seed in [8, 9]
        )

        data = [load_from_disk(file)for file in files]
        data_seed_8 = data[0]
        data_seed_9 = data[1]

        self.assertEqual(10_000, len(data_seed_8['t']))
        self.assertEqual((2, 10_000), data_seed_8['voltages'].shape)
        self.assertEqual(0, len(data_seed_8['spikes'][0]))
        self.assertEqual(1, len(data_seed_8['spikes'][1]))

        self.assertEqual(10_000, len(data_seed_9['t']))
        self.assertEqual((2, 10_000), data_seed_9['voltages'].shape)
        self.assertEqual(0, len(data_seed_9['spikes'][0]))
        self.assertEqual(0, len(data_seed_9['spikes'][1]))

        print(data_seed_8['voltages'].mean(axis=1))

    def test_can_load_aggregate(self):
        exp_label = "Simulate diffusion process test"
        data = load_aggregate(exp_label=exp_label, testing=True)

        number_files_available = len(list((Path.cwd().resolve() / "test" / "simulate_diffusion_process_test").glob("*.h5")))
        self.assertEqual(number_files_available, len(data))

    def test_difussion_approximation_no_firing_runs_in_parallel(self):
        not_firing_config = default_diffusion_lif_config.with_property(DiffusionLIFConfig.KEY_NEURON_THRESHOLD, 100)
        files = Parallel(n_jobs=2, prefer="processes")(
            delayed(simulate_and_plot)(testing=True, lif_config=not_firing_config,
                                       mu=-47.61595645 * mV, sigma=1.90531046 * mV, delta_v=0.7 * mV,
                                       exp_label="Check OU process μ, σ", seed=seed,
                                       )
            for seed in [8, 9]
        )
        data = [load_from_disk(file) for file in files]
        data_seed_8 = data[0]
        data_seed_9 = data[1]

        self.assertEqual(10_000, len(data_seed_8['t']))
        self.assertEqual((2, 10_000), data_seed_8['voltages'].shape)
        self.assertEqual(0, len(data_seed_8['spikes'][0]))
        self.assertEqual(0, len(data_seed_8['spikes'][1]))

        self.assertEqual(10_000, len(data_seed_9['t']))
        self.assertEqual((2, 10_000), data_seed_9['voltages'].shape)
        self.assertEqual(0, len(data_seed_9['spikes'][0]))
        self.assertEqual(0, len(data_seed_9['spikes'][1]))

    def test_check_name_of_local_folder(self):
        exp_label = "Test diffusion process"
        script_name = exp_label.replace(".", "_").replace(" ", "_").lower()
        file_location = get_or_create_file_base_dir(save_name=script_name, out_dir="test/")
        print(file_location.resolve())

