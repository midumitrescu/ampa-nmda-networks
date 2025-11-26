from typing import Callable

import matplotlib.pyplot as plt
from brian2 import cm, uS
from joblib import Parallel, delayed
from matplotlib import gridspec

from Plotting import show_plots_non_blocking
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_7_one_compartment_step_input.grid_computations import sim_and_plot_experiment_grid_with_lambda, \
    grid_title, parallelize

from iteration_8_compute_mean_steady_state import one_compartment_with_up_down_and_steady


def sim_and_plot_experiment_grid_with_increasing_nmda_input(experiment: Experiment, title, nmda_schedule: list[float]):
    experiments = [experiment.with_property("g_nmda", nmda_strength) for nmda_strength in nmda_schedule]
    title = grid_title(panel_title=title, experiment=experiment)
    return sim_and_plot_experiment_grid_with_lambda(experiments, title, )


def run_simulate_with_steady_state(experiments: list[Experiment]):
    return parallelize(experiments, sim)