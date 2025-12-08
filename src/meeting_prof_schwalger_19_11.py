import matplotlib.pyplot as plt

from Configuration import Experiment, NetworkParams, PlotParams
from iteration_5_nmda_refactored.network_with_nmda import sim_and_plot
from models import NMDAModels

plt.rcParams['text.usetex'] = True

experiment = Experiment({
    NetworkParams.KEY_NU_E_OVER_NU_THR: 0,
    NetworkParams.KEY_EPSILON: 0.1,
    "g": 4,
    "g_ampa": 2.5e-06,
    "g_gaba": 2.5e-06,
    "g_nmda": 5e-07,

    Experiment.KEY_SELECTED_MODEL: NMDAModels.model_with_detailed_hidden_variables.eq,
    Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda", "sigmoid_v",
                                                "g_nmda", "I_nmda"],
    "record_N": 10,

    Experiment.KEY_SIMULATION_CLOCK: 0.005,

    "t_range": [[0, 2000], [1500, 1520]],
    PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                        PlotParams.AvailablePlots.HIDDEN_VARIABLES]
})


def network_under_nmda_automatically_bifurcating():
    network_without_nmda()

    interesting_nus = [1.8722, 1.8723]
    for nu_ext_over_nu_thr in interesting_nus:
        sim_and_plot(experiment.with_property(NetworkParams.KEY_NU_E_OVER_NU_THR, nu_ext_over_nu_thr))

    detailed_view = experiment.with_property(NetworkParams.KEY_NU_E_OVER_NU_THR, interesting_nus[-1]).with_property(
        "t_range", [500, 550])
    sim_and_plot(detailed_view)


def network_without_nmda():
    sim_and_plot(
        experiment.with_property(NetworkParams.KEY_NU_E_OVER_NU_THR, 1.8723).with_property("g_nmda", 0).with_property(
            "t_range", [[0, 2000]]))




'''
10.5 -> no Firing
11 -> assynchronous irregular
12 -> synchronous regular
'''
def network_showing_synchronous_regular():
    interesting_nus = [5.2, 5.3]
    interesting_nus = [5.25]
    interesting_nus = [10.5, 11, 11.5]
    for nu_ext_over_nu_thr in interesting_nus:
        config = {
            NetworkParams.KEY_NU_E_OVER_NU_THR: nu_ext_over_nu_thr,
            NetworkParams.KEY_EPSILON: 0.1,
            "g": 4,
            #"g_ampa": 1e-06,
            #"g_gaba": 1e-06,
            #"g_nmda": 5e-06,
            "g_ampa": 0.5e-06,
            "g_gaba": 0.5e-06,
            #"g_nmda": 1e-06,
            "g_nmda": 0,
            Experiment.KEY_SELECTED_MODEL: NMDAModels.model_with_detailed_hidden_variables.eq,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda", "sigmoid_v",
                                                        "g_nmda", "I_nmda"],
            "record_N": 10,

            #Experiment.KEY_SIMULATION_CLOCK: 0.005,

            #"t_range": [[0, 500], [160, 180], [250, 270]],
            "t_range": [[0, 500]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.HIDDEN_VARIABLES]
        }
        sim_and_plot(Experiment(config))
    #sim_and_plot(Experiment(config).with_property("g_nmda", 0))
