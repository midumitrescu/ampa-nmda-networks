import sys

import brian2
from brian2 import PoissonInput, SpikeMonitor, msecond
from loguru import logger

from iteration_12_siegert.df_utils import prepare_experiment_with_N_nmda, filename_for_nu_scan_experiment

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

import sys

from brian2 import clear_cache
from loguru import logger

from iteration_12_siegert.df_utils import filename_for_N_scan_experiment, save_metadata_header, \
    find_last_index

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_steady_state

import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from brian2 import plt, mpl, StateMonitor, mV, start_scope, defaultclock, kHz, NeuronGroup, run, \
    second

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    SimulationResults

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

def simulate_and_record_with_only_nmda_input(experiment: Experiment) -> SimulationResults:
    if experiment.in_testing:
        np.random.seed(0)
        brian2.devices.device.seed(0)
        brian2.seed(0)
        np.random.default_rng(0)

    start_scope()

    defaultclock.dt = experiment.sim_clock

    C = experiment.neuron_params.C

    theta = experiment.neuron_params.theta
    g_L = experiment.neuron_params.g_L
    E_leak = experiment.neuron_params.E_leak
    V_r = experiment.neuron_params.V_r

    g_nmda_max = experiment.synaptic_params.g_nmda
    g_x = experiment.synaptic_params.g_x_nmda

    E_nmda = experiment.synaptic_params.e_ampa

    MG_C = experiment.synaptic_params.MG_C  # extracellular magnesium concentration

    tau_nmda_rise = experiment.synaptic_params.tau_nmda_rise
    tau_nmda_decay = experiment.synaptic_params.tau_nmda_decay

    alpha = 0.5 * kHz  # saturation of NMDA channels at high presynaptic firing rates

    model = experiment.model
    single_neuron = NeuronGroup(1,
                                model=model,
                                threshold="v >= theta",
                                reset="v = V_r",
                                refractory=experiment.neuron_params.tau_rp,
                                method=experiment.integration_method)
    single_neuron.v[:] = E_leak

    order = [0, 1, 2, 3, 4, 5] if experiment.in_testing else [0] * 5

    P_upstate_nmda = PoissonInput(target=single_neuron, target_var="x_nmda",
                                  N=experiment.network_params.up_state.N_NMDA,
                                  rate=experiment.network_params.up_state.nu_nmda, weight=g_x, order=order[4])

    spike_monitor = SpikeMonitor(single_neuron)
    v_monitor = StateMonitor(source=single_neuron,
                             variables="v", record=True)

    internal_states_monitor = StateMonitor(source=single_neuron, variables=["x_nmda", "s_nmda", "g_nmda"],
                                           record=True)
    currents_monitor = StateMonitor(source=single_neuron, variables=["I_nmda"],
                                    record=True)

    reporting = "text" if experiment.in_testing else None
    run(experiment.sim_time, report=reporting, report_period=60 * second)

    return SimulationResults(experiment, None, spike_monitor, v_monitor, g_monitor=None, internal_states_monitor=internal_states_monitor,
                             currents_monitor=currents_monitor)

def run_nu_nmda_input_simulation_and_compute_statistics(nu, index, up_state_base: dict, base: Experiment, skip_ms = 100*msecond):
    exp = prepare_experiment_with_N_nmda(nu, up_state_base, base)

    skip_start_simulation = int(skip_ms / exp.sim_clock)

    steady_state_results = sim_steady_state(exp, state=exp.network_params.up_state)
    simulation_results = simulate_and_record_with_only_nmda_input(exp)

    spikes = simulation_results.spikes.all_values['t'][0] / mV
    if len(spikes) < 3:
        mean_isi = np.nan
        std_isi = np.nan
    else:
        isis = np.diff(spikes)
        mean_isi = np.mean(isis)
        std_isi = np.std(isis, ddof=0)
    if int(nu % 1) == 0:
        print(nu, " done")
    x_nmda_simulation = simulation_results.internal_states_monitor.x_nmda[:, skip_start_simulation:]
    s_nmda_simulation = simulation_results.internal_states_monitor.s_nmda[:, skip_start_simulation:]
    return {
        "index": index,
        "nu_nmda": nu,
        "n_nmda": exp.network_params.up_state.N_NMDA,
        "v_steady": steady_state_results.v_steady,
        "g_nmda_steady": steady_state_results.g_nmda_steady,
        "x_nmda_steady": steady_state_results.x_nmda_steady,
        "s_nmda_steady": steady_state_results.s_nmda_steady,

        "v_mean": np.mean(simulation_results.voltages.v[:, skip_start_simulation:]),
        "v_var": np.var(simulation_results.voltages.v[:, skip_start_simulation:]),
        "g_nmda_mean": np.mean(simulation_results.internal_states_monitor.g_nmda[:, skip_start_simulation:]),
        "g_nmda_var": np.var(simulation_results.internal_states_monitor.g_nmda[:, skip_start_simulation:]),
        "x_nmda_mean": np.mean(x_nmda_simulation),
        "x_nmda_var": np.var(x_nmda_simulation),
        "s_nmda_mean": np.mean(s_nmda_simulation),
        "s_nmda_var": np.var(s_nmda_simulation),

        "corr_coef_x_s": np.corrcoef(x=x_nmda_simulation, y=s_nmda_simulation)[0, 1],

        "i_nmda_mean": np.mean(simulation_results.currents.I_nmda[:, skip_start_simulation:]),
        "i_nmda_var":  np.var(simulation_results.currents.I_nmda[:, skip_start_simulation:]),
        "mean_rate": simulation_results.spikes.mean_rate,
        "num_spikes": simulation_results.spikes.num_spikes,
        "mean_isi": mean_isi,
        "std_isi": std_isi,
        "cv_isi": std_isi / mean_isi,
    }


def run_nmda_input_simulation_and_compute_statistics(n, up_state_base: dict, base: Experiment, skip_ms = 100*msecond):
    exp = prepare_experiment_with_N_nmda(n, up_state_base, base)

    steady_state_results = sim_steady_state(exp, state=exp.network_params.up_state)
    simulation_results = simulate_and_record_with_only_nmda_input(exp)

    spikes = simulation_results.spikes.all_values['t'][0] / mV
    if len(spikes) < 3:
        mean_isi = np.nan
        std_isi = np.nan
    else:
        isis = np.diff(spikes)
        mean_isi = np.mean(isis)
        std_isi = np.std(isis, ddof=0)
    if int(n % 10) == 0:
        print(n, " done")

    skip_start_simulation = int(skip_ms / exp.sim_clock)
    x_nmda_simulation = simulation_results.internal_states_monitor.x_nmda[:, skip_start_simulation:]
    s_nmda_simulation = simulation_results.internal_states_monitor.s_nmda[:, skip_start_simulation:]
    return {
        "N": n,
        "v_steady": steady_state_results.v_steady,
        "g_nmda_steady": steady_state_results.g_nmda_steady,
        "x_nmda_steady": steady_state_results.x_nmda_steady,
        "s_nmda_steady": steady_state_results.s_nmda_steady,

        "v_mean": np.mean(simulation_results.voltages.v[:, skip_start_simulation:]),
        "v_var": np.var(simulation_results.voltages.v[:, skip_start_simulation:]),
        "g_nmda_mean": np.mean(simulation_results.internal_states_monitor.g_nmda[:, skip_start_simulation:]),
        "g_nmda_var": np.var(simulation_results.internal_states_monitor.g_nmda[:, skip_start_simulation:]),

        "x_nmda_mean": np.mean(x_nmda_simulation),
        "x_nmda_var": np.var(x_nmda_simulation),
        "s_nmda_mean": np.mean(s_nmda_simulation),
        "s_nmda_var": np.var(s_nmda_simulation),

        "corr_coef_x_s": np.corrcoef(x=x_nmda_simulation, y=s_nmda_simulation)[0, 1],

        "i_nmda_mean": np.mean(simulation_results.currents.I_nmda[:, skip_start_simulation:]),
        "i_nmda_var":  np.var(simulation_results.currents.I_nmda[:, skip_start_simulation:]),
        "mean_rate": simulation_results.spikes.mean_rate,
        "num_spikes": simulation_results.spikes.num_spikes,
        "mean_isi": mean_isi,
        "std_isi": std_isi,
        "cv_isi": std_isi / mean_isi,
    }


sigle_compartment_with_nmda_only = '''
dv/dt = 1/C * (- I_L - I_nmda): volt (unless refractory)

I_L = g_L * (v-E_leak): amp
I_nmda = g_nmda * (v - E_nmda): amp

g_nmda = g_nmda_max * sigmoid_v * s_nmda: siemens
ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha * x_nmda * (1 - s_nmda) : 1
dx_nmda/dt = - x_nmda / tau_nmda_rise : 1
sigmoid_v = 1/(1 + (MG_C/mmole)/3.57 * exp(-0.062*(v/mvolt))): 1
'''


def scan_for_nu_nmda_variables(base: Experiment, output_dir="simulations_2", nu_max=600, batch_size=50, test=True):
    clear_cache("cython")
    experiment = base.with_properties({
        Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "g_nmda"],
        Experiment.KEY_CURRENTS_TO_RECORD: ["I_nmda"],
        "t_range": [0, 1000] if test else [0, 30 * 1000],
        "in_testing": False,
    })
    up_state_base = experiment.network_params.up_state.params

    file_name = filename_for_nu_scan_experiment(experiment=experiment, output_dir=output_dir, nu_max=nu_max)

    metadata = experiment.params
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(file_name):
        # Fresh run
        save_metadata_header(file_name, metadata)
        start_idx = 0
    else:
        # Resume run
        start_idx = find_last_index(file_name, "index") + 1

    step = 0.1
    last_simulated_nu = int(start_idx * step)
    nu_s = np.arange(last_simulated_nu, nu_max + 0.05, step=step)
    if last_simulated_nu >= nu_s[-1]:
        print("All runs already completed.")
        return file_name

    print("Simulating  ", nu_s)
    num_batches = int(np.ceil(len(nu_s) / batch_size))
    skip_elems = 100 * msecond if test else 300 * msecond


    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(nu_s))
        batch_elements = nu_s[start:end]
        print("Processing batch", batch_elements)

        results = Parallel(n_jobs=1)(
            delayed(lambda nu, index: run_nu_nmda_input_simulation_and_compute_statistics(nu, index, up_state_base, experiment,
                                                                               skip_ms=skip_elems))(nu, index) for
            nu, index in zip(batch_elements, np.arange(start, end))
        )

        batch_df = pd.DataFrame(results)

        batch_df.to_csv(
            file_name,
            mode="a",
            header=start == 0,
            index=False,
            float_format="%.20f",
        )

        print()

    return file_name


def scan_N_for_nmda_variables(base: Experiment, output_dir="simulations_2", N_max=10_000,
                              batch_size=100, test=True):
    clear_cache("cython")
    experiment = base.with_properties({
        Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "g_nmda"],
        Experiment.KEY_CURRENTS_TO_RECORD: ["I_nmda"],
        "t_range": [0, 1000] if test else [0, 60 * 1000],
        "in_testing": False,
    })
    up_state_base = experiment.network_params.up_state.params

    file_name = filename_for_N_scan_experiment(experiment=experiment, output_dir=output_dir, N_max=N_max)

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
        return file_name

    n_s = np.arange(start_idx, N_max)

    print("Simulating ", n_s)
    num_batches = int(np.ceil(len(n_s) / batch_size))
    skip_ms = 100 * msecond if test else 500 * msecond

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(n_s))
        batch_elements = n_s[start:end]

        results = Parallel(n_jobs=-1)(
            delayed(lambda n: run_nmda_input_simulation_and_compute_statistics(n, up_state_base, experiment, skip_ms=skip_ms))(n) for n in batch_elements
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

    return file_name
