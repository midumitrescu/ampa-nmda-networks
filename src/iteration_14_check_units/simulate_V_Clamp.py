import brian2.devices.device
import matplotlib.pyplot as plt
import numpy as np
from brian2 import PopulationRateMonitor, SpikeMonitor, StateMonitor, Hz, ms, nsiemens, seed, mpl, start_scope, \
    defaultclock, kHz, mmole, NeuronGroup, PoissonGroup, Synapses, network_operation, second, mV, run, stop, Mohm
from brian2.units.allunits import nampere
from joblib import delayed, Parallel
from loguru import logger
from matplotlib import gridspec
from matplotlib.gridspec import SubplotSpec
from mpl_toolkits.axes_grid1.mpl_axes import Axes

from Plotting import show_plots_non_blocking
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, State, \
    CurrentClampParams, NeuronModelParams
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_steady_state, \
    no_presynaptic_input
from utils import ExtendedDict

import pandas as pd
plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

current_clamp_model = '''

'''

steady_state_model_with_current_clamp = '''
dv/dt = 1/C * (- I_L - I_ampa - I_gaba - I_nmda + I_inj): volt
I_L = g_L * (v-E_leak): amp
I_ampa = g_e * (v - E_ampa): amp
I_gaba = g_i * (v - E_gaba): amp
I_nmda = g_nmda * (v - E_nmda): amp

dg_e/dt = -g_e / tau_ampa + g_ampa * N_E * r_e : siemens
dg_i/dt = -g_i / tau_gaba + g_gaba * N_I * r_i : siemens

dx_nmda/dt = - x_nmda / tau_nmda_rise + g_x * N_N * r_nmda: 1
ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha * x_nmda * (1 - s_nmda) : 1
sigmoid_v = 1/(1 + (MG_C/mmole)/3.57 * exp(-0.062*(v/mvolt))): 1

g_nmda = g_nmda_max * sigmoid_v * s_nmda: siemens
'''


def run_current_injection_simulation(experiment: Experiment, injected_currents: list[float]):
    steady_state_results_no_current = sim_steady_state(experiment.with_property(CurrentClampParams.KEY_I_INJECTED, 0),
                                                       state=experiment.network_params.up_state)

    current_clamp_experiments_no_up_state = [experiment.with_properties({
        CurrentClampParams.KEY_I_INJECTED: current,
        State.KEY_STATE_UP: no_presynaptic_input.params,
    }) for current in injected_currents]

    current_clamp_for_up_state = [experiment.with_properties({
        CurrentClampParams.KEY_I_INJECTED: current,
    }) for current in injected_currents]

    current_simulations = Parallel(n_jobs=1)(
        delayed(sim_steady_state)(current_experiment, current_experiment.network_params.up_state) for current_experiment in current_clamp_experiments_no_up_state + current_clamp_for_up_state
    )

    no_up_state = current_simulations[:len(current_clamp_experiments_no_up_state)]
    with_up_state = current_simulations[len(current_clamp_experiments_no_up_state):]

    with_up_state = [result.recompute_r_in(steady_state_results_no_current) for result in with_up_state]

    df_no_up_state = pd.DataFrame([res.to_dict() for res in no_up_state])
    df_in_up_state = pd.DataFrame([res.to_dict() for res in with_up_state])

    return df_no_up_state, df_in_up_state

def simulate_current_injection_2(experiment: Experiment, state: State = no_presynaptic_input):
    sim_results_no_steady_input = sim_steady_state(experiment, state=state)

    sim_results_state_no_current = sim_steady_state(experiment.with_property(CurrentClampParams.KEY_I_INJECTED, 0),
                                                    state=experiment.network_params.up_state)
    sim_results_steady_input = sim_steady_state(experiment, state=experiment.network_params.up_state)

    if sim_results_steady_input.experiment.current_clamp_params.i_inj != 0:
        sim_results_steady_input.r_in = (
                                                    sim_results_steady_input.v_steady - sim_results_state_no_current.v_steady) * mV / sim_results_steady_input.experiment.current_clamp_params.i_inj / Mohm

    return sim_results_no_steady_input, sim_results_steady_input

def simulate_current_injection(experiment: Experiment, state: State = no_presynaptic_input):
    sim_results_no_steady_input = sim_steady_state(experiment, state=state)

    sim_results_state_no_current = sim_steady_state(experiment.with_property(CurrentClampParams.KEY_I_INJECTED, 0),
                                                    state=experiment.network_params.up_state)
    sim_results_steady_input = sim_steady_state(experiment, state=experiment.network_params.up_state)

    if sim_results_steady_input.experiment.current_clamp_params.i_inj != 0:
        sim_results_steady_input.r_in = (
                                                    sim_results_steady_input.v_steady - sim_results_state_no_current.v_steady) * mV / sim_results_steady_input.experiment.current_clamp_params.i_inj / Mohm

    return sim_results_no_steady_input, sim_results_steady_input
