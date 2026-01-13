from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from brian2 import cm
from joblib import Parallel, delayed
from matplotlib import gridspec

from Plotting import show_plots_non_blocking
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_7_one_compartment_step_input.grid_computations import sim_and_plot_experiment_grid_with_lambda, \
    grid_title, parallelize
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import \
    simulate_with_up_and_down_state_and_nmda_and_steady_state, SimulationResultsWithSteadyState, plot_raster_and_rates, \
    plot_voltages_and_g_s, plot_currents_graph, plot_simulation

def convert_to_experiment_list(experiment: Experiment, nmda_schedule: list[float]) -> list[Experiment]:
    return [experiment.with_property("g_nmda", nmda_strength) for nmda_strength in nmda_schedule]

def convert_to_experiment_matrix(experiment: Experiment, nmda_schedule):
    if type(nmda_schedule[0]) is list:
        return [convert_to_experiment_list(experiment, row) for row in nmda_schedule]
    else:
        return [convert_to_experiment_list(experiment, nmda_schedule)]


def sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state(experiment: Experiment, title, nmda_schedule: list[float], show_individual_plots=True):
    experiments = convert_to_experiment_matrix(experiment, nmda_schedule)
    title = grid_title(panel_title=title, experiment=experiment)
    return sim_and_plot_experiment_grid_with_lambda(experiments, title, run_simulate_with_steady_state, show_individual_plots=show_individual_plots)

def run_simulate_with_steady_state(experiments: list[Experiment]):
    return parallelize(experiments, simulate_with_up_and_down_state_and_nmda_and_steady_state)

def sim_and_plot_experiment_grid_with_lambda(experiments, title, obtain_results_function: Callable[
                                                 [list[Experiment]], list[SimulationResultsWithSteadyState]], show_individual_plots=True):

    experiments_as_numpy_array = np.array(experiments)
    experiments_shape = experiments_as_numpy_array.shape

    results = obtain_results_function(experiments_as_numpy_array.flatten())
    results_in_matrix_form = np.array(results).reshape(experiments_shape)

    t_range = experiments_as_numpy_array.flatten()[0].plot_params.t_range
    if t_range:
        params_t_range = t_range

        if isinstance(params_t_range[0], list):
            for time_slot in params_t_range:
                plot_results_grid(results_in_matrix_form, time_slot, title=title)
        else:
            plot_results_grid(results_in_matrix_form, t_range, title=title)
    if show_individual_plots:
        for result in results:
            plot_simulation(result)
            show_plots_non_blocking()

    return results_in_matrix_form

def parallelize(experiments: list[Experiment],
                function_simulate_one_experiment: Callable[[Experiment], SimulationResultsWithSteadyState]):
    def sim_unpickled(experiment: Experiment):
        simulation_results = function_simulate_one_experiment(experiment)
        simulation_results.internal_states_monitor = None
        return simulation_results

    return Parallel(n_jobs=16)(
        delayed(sim_unpickled)(current_experiment) for current_experiment in experiments
    )

def plot_results_grid(results: np.ndarray[SimulationResultsWithSteadyState, np.dtype[SimulationResultsWithSteadyState]], time_range: tuple[int, int], title: str):
    plot_grid_raster_population_and_g_s(results, time_range, title=title)
    plot_grid_currents(results, time_range, title=title)


def plot_grid_raster_population_and_g_s(results: np.ndarray[SimulationResultsWithSteadyState, np.dtype[SimulationResultsWithSteadyState]], time_range: tuple[int, int], title: str):
    if not results.flatten()[0].experiment.plot_params.show_raster_and_rate():
        return

    fig = plt.figure(figsize=(40, 30))
    fig.suptitle(title, size=25)

    rows, cols = results.shape
    outer = gridspec.GridSpec(2*rows, cols, figure=fig, hspace=0.2, wspace=0.1)

    for idx, result in np.ndenumerate(results):
        row, column = idx
        ax_spikes, _ = plot_raster_and_rates(result, time_range, outer[2*row, column])
        plot_voltages_and_g_s(result, time_range, outer[2*row + 1, column])

        ax_spikes.set_title(f"{gen_raster_and_rates_grid_subtitle(result)} \n {ax_spikes.get_title()}")

    show_plots_non_blocking()


def gen_raster_and_rates_grid_subtitle(results: SimulationResultsWithSteadyState):
    experiment = results.experiment
    return fr"""Synapse: [$g_{{\mathrm{{AMPA}}}}={experiment.synaptic_params.g_ampa:.2f}$, $g_{{\mathrm{{GABA}}}}={experiment.synaptic_params.g_gaba:.2f}$, $g={experiment.network_params.g}$, $g_{{\mathrm{{NMDA}}}}={experiment.synaptic_params.g_nmda:.2f}$]"""


def plot_grid_currents(results: np.ndarray[SimulationResultsWithSteadyState, np.dtype[SimulationResultsWithSteadyState]], time_range, title: str):
    if not results.flatten()[0].experiment.plot_params.show_currents_plots():
        return

    fig = plt.figure(figsize=(40, 30))
    fig.suptitle(title, size=25)

    rows, cols = results.shape
    outer = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.2, wspace=0.1)

    for idx, result in np.ndenumerate(results):
        plot_currents_graph(result, time_range, outer[idx])

    show_plots_non_blocking()