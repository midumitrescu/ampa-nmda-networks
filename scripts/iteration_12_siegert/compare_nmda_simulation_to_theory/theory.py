import sys

from loguru import logger

from iteration_12_siegert.SiegertScripts import compute_siegert_firing_rate
from utils import ExtendedDict

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

import sys

from loguru import logger

from iteration_12_siegert.df_utils import prepare_experiment_with_N_nmda

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

import pandas as pd
from joblib import Parallel, delayed
from brian2 import plt, mpl

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True
import numpy as np
from brian2 import plt, mpl, mV, nsiemens

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True



def nmda_variables(n, up_state_base: dict, base: Experiment):
    exp = prepare_experiment_with_N_nmda(n, up_state_base, base)

    mu_v_with_nmda = exp.effective_time_constant_up_state.E_0_with_nmda()
    sigma_v_with_nmda = exp.effective_time_constant_up_state.std_voltage_with_nmda()
    tau_0_with_nmda = exp.neuron_params.C / (exp.effective_time_constant_up_state.mean_total_conductance_with_nmda())
    siegert_firing_rate_with_nmda = compute_siegert_firing_rate(mu_v_with_nmda, sigma_v_with_nmda, tau_0_with_nmda, exp)

    crazy_nmda = exp.effective_time_constant_up_state.crazy_s_nmda_mean_and_variance()

    return ExtendedDict({
        "N": n,
        "mu_v_with_nmda": mu_v_with_nmda / mV,
        "sigma_v_with_nmda": sigma_v_with_nmda / mV,
        "firing_rate_with_nmda": siegert_firing_rate_with_nmda,

        "g_nmda_mean": exp.effective_time_constant_up_state.compute_mean_g_nmda() / nsiemens,
        "g_nmda_std": exp.effective_time_constant_up_state.std_g_nmda() / nsiemens,

        "x_mean": exp.effective_time_constant_up_state.mean_x_nmda(),
        "s_mean": exp.effective_time_constant_up_state.mean_s_nmda(),
        "x_std": exp.effective_time_constant_up_state.std_x_nmda(),
        "s_std":  exp.effective_time_constant_up_state.std_s_nmda(),

        "s_nmda_crazy_mean": crazy_nmda[0],
        "s_nmda_crazy_var": crazy_nmda[1],
    })


def compute_theoretical_nmda_mean_sigma_and_rate(max_n, base):
    up_state_base = {
        "N": 0,
        "nu": 0,
        "N_nmda": 10,
        "nu_nmda": 10,
    }
    N = np.arange(1, max_n)
    results = Parallel(
        n_jobs=-1,  # use all cores
        backend="loky")(delayed(lambda n: nmda_variables(n, up_state_base, base))(n) for n in N)
    return pd.DataFrame.from_records(results)