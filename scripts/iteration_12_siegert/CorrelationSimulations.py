import itertools
import sys
import unittest

import matplotlib.pyplot as plt
from brian2 import ms, clear_cache, Hz
from loguru import logger

from Plotting import show_plots_non_blocking
from iteration_12_siegert.df_utils import save_metadata_header, \
    find_last_index
from iteration_12_siegert.one_compartment_with_up_only import simulate_and_record_essential_variables

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams, State
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_steady_state
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control, palmer_nmda_block

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

INDEX_COLUMN_NAME = "index"

def label_for_float(nu: float):
    return f"{nu: .5f}".replace(".", "_")

def filename_for_experiment(experiment: Experiment, N_nmda: int, nu_nmda: float, output_dir="simulations_2"):
    return f"{output_dir}/{experiment.plot_params.panel.replace(" ", "_")}_N_nmda_max_{N_nmda}_nu_nmda_max_{label_for_float(nu=nu_nmda)}_T_{int(experiment.sim_time / ms)}.csv"

def prepare_experiment_with_N_and_nu_nmda(n_nmda: int, nu_nmda: float, up_state_base: dict,
                                          experiment_base: Experiment) -> Experiment:
    up_state_base_local = up_state_base.copy()
    up_state_base_local[State.KEY_N] = 0
    up_state_base_local[State.KEY_NU] = 0
    up_state_base_local[State.KEY_N_NMDA] = int(n_nmda)
    up_state_base_local[State.KEY_NU_NMDA] = int(nu_nmda)

    exp = experiment_base.with_property("up_state", up_state_base_local)
    return exp


def x_s_correlation(n_nmda: int, nu_nmda: float, up_state_base: dict, experiment_base: Experiment):

    exp = prepare_experiment_with_N_and_nu_nmda(n_nmda=n_nmda, nu_nmda=nu_nmda, up_state_base=up_state_base,
                                                experiment_base=experiment_base)

    steady_state_results = sim_steady_state(exp, state=exp.network_params.up_state)
    simulation_results = simulate_and_record_essential_variables(exp)

    x_s_corr_score = np.corrcoef(x=simulation_results.internal_states_monitor.x_nmda,
                                 y=simulation_results.internal_states_monitor.s_nmda)

    spikes = simulation_results.spikes.all_values['t'][0] / ms
    if len(spikes) < 3:
        mean_isi = np.nan
        std_isi = np.nan
    else:
        isis = np.diff(spikes)
        mean_isi = np.mean(isis)
        std_isi = np.std(isis, ddof=0)
    res = {
        "n_nmda": n_nmda,
        "nu_nmda": nu_nmda,
        "v_steady": steady_state_results.v_steady,
        "g_nmda_steady": steady_state_results.g_nmda_steady,
        "x_nmda_steady": steady_state_results.x_nmda_steady,
        "s_nmda_steady": steady_state_results.s_nmda_steady,
        "v_mean": np.mean(simulation_results.voltages.v),
        "v_var": np.var(simulation_results.voltages.v),
        "g_nmda_mean": np.mean(simulation_results.internal_states_monitor.g_nmda),
        "g_nmda_var": np.var(simulation_results.internal_states_monitor.g_nmda),
        "x_nmda_mean": np.mean(simulation_results.internal_states_monitor.x_nmda),
        "x_nmda_var": np.var(simulation_results.internal_states_monitor.x_nmda),
        "s_nmda_mean": np.mean(simulation_results.internal_states_monitor.s_nmda),
        "s_nmda_var": np.var(simulation_results.internal_states_monitor.s_nmda),
        "x_s_correlation": x_s_corr_score[0][1],
        "mean_rate": simulation_results.spikes.mean_rate,
        "num_spikes": simulation_results.spikes.num_spikes,
        "mean_isi": mean_isi,
        "std_isi": std_isi,
        "cv_isi": std_isi / mean_isi,
    }
    logger.info("Done compute [x, s] correlation for [N_nmda = {}, nu_nmda = {}]", n_nmda, nu_nmda)
    return res


'''
generate elems = itertools.product(n_nmda array, nu_nmda array)
'''
def run_one_batch(elems, up_state_base: dict, experiment_base: Experiment):
    results = Parallel(n_jobs=-2)(
        delayed(lambda n_nmda, nu_nmda: x_s_correlation(n_nmda=n_nmda, nu_nmda=nu_nmda, up_state_base=up_state_base,
                                          experiment_base=experiment_base))(n_nmda, nu_nmda) for n_nmda, nu_nmda in elems
    )

    return pd.DataFrame(results)


def scan_n_nmda_and_nu_nmda_simulation(base: Experiment, output_dir="correlations", N_nmda_max=100, nu_nmda_max=10,
                                       batch_size=100):
    clear_cache("cython")
    experiment = base.with_properties({
        Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "g_nmda"],
        #"t_range": [0, 100],
        "t_range": [0, 60 * 1000],
        "in_testing": False
    })
    up_state_base = experiment.network_params.up_state.params

    file_name = filename_for_experiment(experiment=experiment, output_dir=output_dir, N_nmda=N_nmda_max, nu_nmda=nu_nmda_max)

    metadata = experiment.params
    os.makedirs(output_dir, exist_ok=True)

    n_scan = np.arange(1, N_nmda_max)
    nu_scan = np.arange(0, nu_nmda_max + 0.1, step=0.1)

    all_combos = list(itertools.product(n_scan, nu_scan))
    number_of_simulations = len(n_scan) * len(nu_scan)

    if not os.path.exists(file_name):
        # Fresh run
        save_metadata_header(file_name, metadata)
        start_idx = 0
        write_header = True
    else:
        # Resume run
        start_idx = find_last_index(file_name, INDEX_COLUMN_NAME)
        write_header = False

    if start_idx >= number_of_simulations:
        print("All runs already completed.")
        return

    print(f"Simulating product of {n_scan} with {nu_scan}")
    num_batches = int(np.ceil(number_of_simulations / batch_size))

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start = batch_idx * batch_size
        if start_idx >  start:
            print(f"Batch {batch_idx} was already processed")
            continue

        end = min(start + batch_size, number_of_simulations)
        batch_elements = all_combos[start:end]

        batch_df = run_one_batch(elems = batch_elements, up_state_base = up_state_base, experiment_base = experiment)
        batch_df[INDEX_COLUMN_NAME] = np.arange(start, end)
        batch_df = batch_df[["index", *batch_df.columns.drop("index")]]

        batch_df.to_csv(
            file_name,
            mode="a",
            header=write_header,
            index=False,
            float_format="%.20f",
        )

        write_header = False
        print()


class SimulationsWithWangNumbers(unittest.TestCase):

    def test_experiment_preparation(self):
        experiment = palmer_control.with_properties({
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "g_nmda"],
            "t_range": [0, 1000],
            # "t_range": [0, 60 * 1000],
            "in_testing": False
        })
        up_state_base = experiment.network_params.up_state.params
        object_under_test = prepare_experiment_with_N_and_nu_nmda(n_nmda=10, nu_nmda=10, up_state_base=up_state_base,
                                                                  experiment_base=experiment)

        self.assertEqual(10, object_under_test.network_params.up_state.N_NMDA)
        self.assertEqual(10, object_under_test.network_params.up_state.nu_nmda / Hz)

    def test_run_one_correlation_simulation(self):
        experiment = palmer_control.with_properties({
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "g_nmda"],
            "t_range": [0, 1000],
            # "t_range": [0, 60 * 1000],
            "in_testing": False
        })
        up_state_base = experiment.network_params.up_state.params

        print(x_s_correlation(n_nmda=10, nu_nmda=10, up_state_base=up_state_base, experiment_base=experiment))

    def test_plot_for_increasing_n_nmda(self):
        experiment = palmer_control.with_properties({
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "g_nmda"],
            "t_range": [0, 1000],
            # "t_range": [0, 60 * 1000],
            "in_testing": False
        })
        up_state_base = experiment.network_params.up_state.params

        df = run_one_batch(elems=itertools.product(np.arange(1, 100), np.array([10])), up_state_base=up_state_base, experiment_base=experiment)

        plt.plot(df.n_nmda, df.x_s_correlation)
        show_plots_non_blocking()

    def test_run_simulation(self):
        scan_n_nmda_and_nu_nmda_simulation(base=palmer_nmda_block.with_property("panel", "x_s_correlation"), batch_size=50)

    def plot_simulation(self):
        pass

