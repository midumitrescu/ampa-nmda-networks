import sys

from brian2 import ms
from loguru import logger

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams

import pandas as pd
from io import StringIO


def prepare_experiment_with_N_tot(n, up_state_base, base) -> Experiment:
    up_state_base_local = up_state_base.copy()
    up_state_base_local["N"] = int(n)
    exp = base.with_property("up_state", up_state_base_local)
    return exp


def filename_for_experiment(experiment: Experiment, N_max: int, output_dir="simulations"):
    return f"{output_dir}/{experiment.plot_params.panel.replace(" ", "_")}_N_{N_max}_T_{int(experiment.sim_time / ms)}.csv"


drop_keys = [Experiment.KEY_SELECTED_MODEL, Experiment.KEY_STEADY_MODEL, Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD,
             Experiment.KEY_CURRENTS_TO_RECORD, "t_range", PlotParams.KEY_WHAT_PLOTS_TO_SHOW]


def save_metadata_header(path, metadata: dict):
    filtered = {k: v for k, v in metadata.items() if k not in drop_keys}
    with open(path, "w") as f:
        for k, v in filtered.items():
            if isinstance(v, float):
                f.write(f"# {k}: {v :.20f}\n")
            else:
                f.write(f"# {k}: {v}\n")


def load_df_without_metadata(csv_path):
    return pd.read_csv(csv_path, comment="#")

def find_last_index(csv_path, index_col):
    with open(csv_path, "r") as f:
        lines = f.readlines()

    # Keep non-comment, non-empty lines
    data_lines = [l for l in lines if l.strip() and not l.startswith("#")]

    if not data_lines:
        # Should not really happen, but be safe
        return 0

    if len(data_lines) == 1:
        # Only CSV header present, no simulation data yet
        return 0

    # Parse only the last data row (plus header)
    header = data_lines[0]
    last_row = data_lines[-1]

    df = pd.read_csv(StringIO(header + last_row))

    return int(df[index_col].iloc[0])


def without_elements_after_n_max(df: pd.DataFrame, max_n: int) -> pd.DataFrame:
    return df[df["N"].astype(int) <= max_n]

def with_elements_between(df: pd.DataFrame, lower_bound: int = -1, upper_bound:int = -1) -> pd.DataFrame:
    result = df.copy()

    if lower_bound is not None and lower_bound >= 0:
        result = result[result["N"].astype(int) >= lower_bound]
    if upper_bound is not None and upper_bound >= 0 and upper_bound >= lower_bound:
        result = result[result["N"].astype(int) <= upper_bound]

    return result
