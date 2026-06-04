from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
from brian2 import mV
from brian2 import ms
from brian2 import usecond

from Plotting import get_or_create_file_base_dir
from iteration_15_OU_process.brian_lif_diffusion import DiffusionSimulation
from iteration_15_OU_process.plot_utils import exp_label_to_folder_name


def load_from_disk(file_path):
    """
    Reads a simulation .h5 file created by save_on_disk()
    and returns all stored data in a dictionary.
    """

    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = {}

    with h5py.File(file_path, "r") as f:

        exp_data = dict(f["exp_data"].attrs)
        t = np.array(f["t"])
        voltages = np.array([
            np.array(f["voltages"]["MK801"]),
            np.array(f["voltages"]["Control"])
        ])

        spikes = [
            np.array(f["spikes"]["MK801"]),
            np.array(f["spikes"]["Control"])
        ]

        data = {
            "file_path": str(file_path),
            "exp_data": exp_data,
            "t": t,
            "voltages": voltages,
            "spikes": spikes,
        }

    return data


def close(a, b, tol=1e-12):
    return abs(a - b) < tol

def assert_same_experiment(reference, current, check_exp_label=True):
    same_exp_keys = ("T", "dt", "mu", "sigma", "delta_v", "theta", "v_r", "tau_m", "tau_ref")
    for key in same_exp_keys:
        if not close(current[key], reference[key]):
            raise ValueError(
                f"Experiment mismatch in file {current['file_path']}\n"
                f"Parameter '{key}': {current[key]} != {reference[key]}"
            )
    if check_exp_label:
        if not current["exp_label"].equals(reference["exp_label"]):
            raise ValueError(
                f"Experiment mismatch in file {current['file_path']}\n"
                f"Exp label current {current["exp_label"]} != reference {reference["exp_label"]}"
            )
    return reference

def save_on_disk(t, v_s, spike_times,
                 diffusion_simulation: DiffusionSimulation,
                lif_config, exp_label,
                 run_id=None, compression="gzip", testing: bool=False):

    script_name = exp_label_to_folder_name(exp_label)

    out_dir = "test/" if testing else "simulation/"
    base_dir = get_or_create_file_base_dir(save_name=script_name, out_dir=out_dir)

    # Important for parallel writes:
    # each worker writes its own file
    if run_id is None:
        run_id = uuid4().hex

    file_path = base_dir / f"{run_id}.h5"

    exp_data = {
        "T": diffusion_simulation.T / ms,
        "dt": int(diffusion_simulation.dt / usecond),
        "mu": diffusion_simulation.mu / mV,
        "sigma": diffusion_simulation.sigma / mV,
        "delta_v": diffusion_simulation.delta_v / mV,
        "seed": diffusion_simulation.seed,
        "exp_label": exp_label,
        "theta": lif_config.theta / mV,
        "v_r": lif_config.V_r / mV,
        "tau_m": lif_config.tau_m / ms,
        "tau_ref": lif_config.tau_rp / ms
    }

    with h5py.File(file_path, "w") as f:
        # store metadata
        meta_grp = f.create_group("exp_data")

        for key, value in exp_data.items():
            if value is None:
                meta_grp.attrs[key] = "__NONE__"
            else:
                meta_grp.attrs[key] = value
        f.create_dataset(
            "t",
            data=t,
            compression=compression,
            shuffle=True,
        )

        # Voltage traces
        grp_v = f.create_group("voltages")

        grp_v.create_dataset(
            "MK801",
            data=v_s[0],
            compression=compression,
            shuffle=True)

        grp_v.create_dataset(
            "Control",
            data=v_s[1],
            compression=compression,
            shuffle=True,
        )
        grp_spikes = f.create_group("spikes")
        grp_spikes.create_dataset(
            "MK801",
            data=spike_times[0],
            compression=compression,
            shuffle=True)

        grp_spikes.create_dataset(
            "Control",
            data=spike_times[1],
            compression=compression,
            shuffle=True,
        )

    return file_path

def load_aggregate(exp_label: str, testing: bool=False, same_exp=False):
    script_name = exp_label_to_folder_name(exp_label)
    out_dir = "test/" if testing else "simulation/"
    folder_path = get_or_create_file_base_dir(save_name=script_name, out_dir=out_dir).resolve()
    files = sorted(folder_path.glob("*.h5"))
    if len(files) == 0:
        raise ValueError(f"No .h5 files found in {folder_path}")

    first_file_data = load_from_disk(files[0])

    reference_config = first_file_data['exp_data']

    all_runs = []

    for file_path in files:
        run_data = load_from_disk(file_path)
        assert_same_experiment(reference_config, run_data["exp_data"], check_exp_label=same_exp)
        all_runs.append(run_data)

    return all_runs

