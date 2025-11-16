from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms, cm, uS
from joblib import Parallel, delayed
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.mpl_axes import Axes

from Plotting import show_plots_non_blocking
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    simulate_with_nmda_cut_off_in_down_state, \
    plot_voltages_and_g_s, SimulationResults, plot_simulation, plot_raster_and_rates, plot_currents, \
    sim_with_persistent_nmda_input


def sim_and_plot_experiment_grid_up_down_state_with_persistent_nmda_input(experiments: list[Experiment], title):
    return sim_and_plot_experiment_grid_with_lambda(experiments, title, run_simulate_with_persistant_nmda)

'''
Expectation is that all experiments share the same time windows
'''


def sim_and_plot_experiment_grid_with_nmda_cut_off_in_down_state(experiments: list[Experiment], title):
    return sim_and_plot_experiment_grid_with_lambda(experiments, title, run_simulate_with_cut_off_nmda_in_down_state)

def sim_and_plot_experiment_grid_with_lambda(experiments: list[Experiment], title,
                                             obtain_results_function: Callable[[list[Experiment]], list[SimulationResults]]):
    results = obtain_results_function(experiments)

    t_range = experiments[0].plot_params.t_range
    if t_range:
        params_t_range = t_range

        if isinstance(params_t_range[0], list):
            for time_slot in params_t_range:
                plot_results_grid(results, time_slot, title=title)
        else:
            plot_results_grid(results, t_range, title=title)

    for result in results:
        plot_simulation(result)
        show_plots_non_blocking()

    return results

def run_simulate_with_persistant_nmda(experiments: list[Experiment]):
    return parallelize(experiments, sim_with_persistent_nmda_input)

def run_simulate_with_cut_off_nmda_in_down_state(experiments: list[Experiment]):
    return parallelize(experiments, simulate_with_nmda_cut_off_in_down_state)

def parallelize(experiments: list[Experiment], function_simulate_one_experiment: Callable[[Experiment], SimulationResults]):
    def sim_unpickled(experiment: Experiment):
        simulation_results = function_simulate_one_experiment(experiment)
        simulation_results.internal_states_monitor = None
        return simulation_results

    return Parallel(n_jobs=16)(
        delayed(sim_unpickled)(current_experiment) for current_experiment in experiments
    )


def plot_results_grid(results: list[SimulationResults], time_range: tuple[int, int], title:str):
    plot_grid_raster_population_and_g_s(results, time_range, title=title)
    plot_grid_currents(results, time_range, title=title)


def plot_grid_raster_population_and_g_s(results: list[SimulationResults], time_range: tuple[int, int], title:str):
    if not results[0].experiment.plot_params.show_raster_and_rate():
        return

    fig = plt.figure(figsize=(35, 25))
    fig.suptitle(title, size=25)

    outer = gridspec.GridSpec(2, len(results), figure=fig, hspace=0.2,
                              wspace=0.1)

    for index, result in enumerate(results):
        ax_spikes, _ = plot_raster_and_rates(result, time_range, outer[0, index])
        plot_voltages_and_g_s(result, time_range, outer[1, index])

        ax_spikes.set_title(f"{gen_raster_and_rates_grid_title(result)} \n {ax_spikes.get_title()}")

    show_plots_non_blocking()

def gen_raster_and_rates_grid_title(results: SimulationResults):
    experiment = results.experiment
    return fr"""Synapse: [$g_{{\mathrm{{AMPA}}}}={experiment.synaptic_params.g_ampa * (cm ** 2) / uS:.2f}\,\mu\mathrm{{S}}$, $g_{{\mathrm{{GABA}}}}={experiment.synaptic_params.g_gaba * (cm ** 2) / uS:.2f}\,\mu\mathrm{{S}}$, $g={experiment.network_params.g}$, $g_{{\mathrm{{NMDA}}}}={experiment.synaptic_params.g_nmda * (cm ** 2) / uS:.2f}\,\mu\mathrm{{S}}$]"""


def plot_grid_currents(results, time_range, title:str):
    if not results[0].experiment.plot_params.show_currents_plots():
        return

    fig = plt.figure(figsize=(35, 25))
    fig.suptitle(title, size=25)

    outer = gridspec.GridSpec(1, len(results), figure=fig, hspace=0.2,
                              wspace=0.1)

    for index, result in enumerate(results):
        plot_currents(result, time_range, outer[index])

    show_plots_non_blocking()


    show_plots_non_blocking()
