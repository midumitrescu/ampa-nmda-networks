import sys
import unittest

from brian2 import ms, clear_cache
from loguru import logger

from iteration_12_siegert.df_utils import prepare_experiment_with_N_tot, filename_for_N_scan_experiment, \
    save_metadata_header, \
    find_last_index, load_df_without_metadata, without_elements_after_n_max
from iteration_12_siegert.one_compartment_with_up_only import simulate_and_record_essential_variables

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_steady_state
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control, palmer_nmda_block

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def mean_and_sigma(n, up_state_base: dict, base: Experiment):
    base = base.with_properties({
        Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["g_e", "g_i", "x_nmda", "s_nmda", "g_nmda"],
        Experiment.KEY_CURRENTS_TO_RECORD: ["I_nmda"]
    })

    exp = prepare_experiment_with_N_tot(n, up_state_base, base)

    steady_state_results = sim_steady_state(exp, state=exp.network_params.up_state)
    simulation_results = simulate_and_record_essential_variables(exp)

    spikes = simulation_results.spikes.all_values['t'][0] / ms
    if len(spikes) < 3:
        mean_isi = np.nan
        std_isi = np.nan
    else:
        isis = np.diff(spikes)
        mean_isi = np.mean(isis)
        std_isi = np.std(isis, ddof=0)
    if int(n % 10) == 0:
        print(n, " done")
    return {
        "N": n,
        "v_steady": steady_state_results.v_steady,
        "g_e_steady": steady_state_results.g_e_steady,
        "g_i_steady": steady_state_results.g_i_steady,
        "g_nmda_steady": steady_state_results.g_nmda_steady,
        "x_nmda_steady": steady_state_results.x_nmda_steady,
        "s_nmda_steady": steady_state_results.s_nmda_steady,
        "v_mean": np.mean(simulation_results.voltages.v),
        "v_var": np.var(simulation_results.voltages.v),
        "g_e_mean": np.mean(simulation_results.internal_states_monitor.g_e),
        "g_e_var": np.var(simulation_results.internal_states_monitor.g_e),
        "g_i_mean": np.mean(simulation_results.internal_states_monitor.g_i),
        "g_i_var": np.var(simulation_results.internal_states_monitor.g_i),
        "g_nmda_mean": np.mean(simulation_results.internal_states_monitor.g_nmda),
        "g_nmda_var": np.var(simulation_results.internal_states_monitor.g_nmda),
        "x_nmda_mean": np.mean(simulation_results.internal_states_monitor.x_nmda),
        "x_nmda_var": np.var(simulation_results.internal_states_monitor.x_nmda),
        "s_nmda_mean": np.mean(simulation_results.internal_states_monitor.s_nmda),
        "s_nmda_var": np.var(simulation_results.internal_states_monitor.s_nmda),
        "corr_coef_x_s": np.corrcoef(x=simulation_results.internal_states_monitor.x_nmda, y=simulation_results.internal_states_monitor.s_nmda)[0, 1],

        "mean_rate": simulation_results.spikes.mean_rate,
        "num_spikes": simulation_results.spikes.num_spikes,
        "mean_isi": mean_isi,
        "std_isi": std_isi,
        "cv_isi": std_isi / mean_isi,
    }


def scan_mean_sigma_from_simulation(base: Experiment, output_dir="simulations_2", N_max=10_000,
                                    batch_size=100):
    clear_cache("cython")
    experiment = base.with_properties({
        Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "g_e", "g_i", "g_nmda"],
        # "t_range": [0, 1000],
        "t_range": [0, 60 * 1000],
        "in_testing": False
    })
    up_state_base = experiment.network_params.up_state.params

    file_name = filename_for_N_scan_experiment(experiment=experiment, output_dir=output_dir, N_max=N_max)

    """
    Parameters
    ----------
    func : callable
        A function or lambda that takes a single integer argument and returns
        either a dict, Series, or something convertible to a DataFrame row.
    N_max : int
        Upper bound for np.arange(0, N_max).
    batch_size : int, default=100
        Number of samples per batch.
    n_jobs : int, default=-1
        Number of parallel jobs for joblib.
    output_dir : str
        Directory where CSV files are saved.
    file_prefix : str
        Prefix for saved batch files.
    """

    metadata = experiment.params
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(file_name):
        # Fresh run
        save_metadata_header(file_name, metadata)
        start_idx = 1
        write_header = True
    else:
        # Resume run
        start_idx = find_last_index(file_name, "N") + 1
        write_header = False

    if start_idx >= N_max:
        print("All runs already completed.")
        return

    n_s = np.arange(start_idx, N_max)

    print("Simulating ", n_s)
    num_batches = int(np.ceil(len(n_s) / batch_size))

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):

        start = batch_idx * batch_size
        end = min(start + batch_size, len(n_s))
        batch_elements = n_s[start:end]

        results = Parallel(n_jobs=-2)(
            delayed(lambda n: mean_and_sigma(n, up_state_base, experiment))(n) for n in batch_elements
        )

        batch_df = pd.DataFrame(results)

        batch_df.to_csv(
            file_name,
            mode="a",
            header=write_header,
            index=False,
            float_format="%.20f",
        )

        write_header = False
        print()

def compute_theoretical_mean_sigma_and_rate(max_n, experiment: Experiment):
    up_state_base = {
        "N": 2000,
        "nu": 82,
        "N_nmda": 10,
        "nu_nmda": 10,
    }
    N = np.arange(1, max_n)
    results = Parallel(n_jobs=-1)(delayed(mean_and_sigma)(n, up_state_base, base) for n in N)
    return pd.DataFrame.from_records(results)

class SimulationsWithWangNumbers(unittest.TestCase):

    def test_call_one_mean_sigma_control_nmda(self):
        scan_mean_sigma_from_simulation(palmer_control, N_max=10_000)
        scan_mean_sigma_from_simulation(palmer_nmda_block, N_max=10_000)


    def test_simulate_without_firing(self):
        palmer_control_no_firing = palmer_control.with_properties({
            "panel": "Control_no_firing",
            "theta": 100
        })
        palmer_nmda_block_no_firing = palmer_nmda_block.with_properties({
            "panel": "NMDA_block_no_firing",
            "theta": 100
        })
        scan_mean_sigma_from_simulation(palmer_control_no_firing, N_max=10_000)
        scan_mean_sigma_from_simulation(palmer_nmda_block_no_firing, N_max=10_000)

    def test_TODO_move_from_here_plot_no_firing(self):
        palmer_control_no_firing = palmer_control.with_properties({
            "panel": "Control_no_firing",
            "theta": 100
        })
        for max_n in [500, 2000, 2500, 3000, 4000]:
            df = self.compute_theoretical_mean_sigma_and_rate(max_n, base=experiment)
            file_control_simulation = "../simulations_2/Control_N_10000_T_60000.csv"
            file_nmda_block_simulation = "../simulations_2/NMDA_block_N_10000_T_60000.csv"
            df_control_simulation = load_df_without_metadata(file_control_simulation)
            df_control_simulation = without_elements_after_n_max(df_control_simulation, max_n=max_n)
            df_nmda_block_simulation = load_df_without_metadata(file_nmda_block_simulation)
            df_nmda_block_simulation = without_elements_after_n_max(df_nmda_block_simulation, max_n=max_n)

            plot_theory_vs_simulation(base=experiment, df_theory=df, df_control_simulation=df_control_simulation,
                                      df_nmda_block_simulation=df_nmda_block_simulation)
