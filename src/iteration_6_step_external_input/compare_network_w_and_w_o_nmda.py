from brian2 import ms

from Configuration import Experiment, PlotParams, SynapticParams, NetworkParams
from iteration_4_conductance_based_model.grid_computations import compare_g_ampa_vs_nu_ext_over_nu_thr as grid_compute_ampa_gaba
from iteration_5_nmda.grid_computations import compare_g_ampa_vs_nu_ext_over_nu_thr as grid_compute_nmda
from iteration_5_nmda.network_with_nmda import wang_model, translated_model

#g_ampas = [2e-06, 2.5e-06, 2.625e-6, 2.75e-06, 3e-06]
g_ampas = [2.625e-6, 2.75e-06, 2.875e-6, 3e-06]
nu_thresholds = [1.55, 1.5625, 1.575, 1.6, 1.65]
simulation = {
        "sim_time": 5_000,
        "sim_clock": 0.1 * ms,
        "epsilon": 0.1,
        "C_ext": 1000,

        "g_L": 0.00004,
        PlotParams.KEY_PANEL: "Investigate the stability of a network with only AMPA and GABA",
        "t_range": [[0, 500], [3000, 5000], [4500, 4550]],
        "voltage_range": [-70, -30],
        "smoothened_rate_width": 3 * ms,
        Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: []
    }

def question_1_network_ampa_gaba_only(g=0):

    experiment = Experiment(simulation).with_property(NetworkParams.KEY_G, g)
    grid_compute_ampa_gaba(experiment, g_ampas, nu_thresholds)

def question_1_network_with_nmda_wang_model(g=0):
    experiment = (Experiment(simulation)
                  .with_property(NetworkParams.KEY_G, g)
                  .with_property(PlotParams.KEY_PANEL,"Investigate the stability of a network with AMPA, GABA and NMDA synapses")
                  .with_property(Experiment.KEY_SELECTED_MODEL, wang_model))
    grid_compute_nmda(experiment, g_ampas, nu_thresholds)


def question_1_network_with_translated_nmda_model(g=0):
    experiment = (Experiment(simulation)
                  .with_property(NetworkParams.KEY_G, g)
                  .with_property(PlotParams.KEY_PANEL,
                                 "Investigate the stability of a network with AMPA, GABA and NMDA synapses and plausible NMDA 1/2")
                  .with_property(Experiment.KEY_SELECTED_MODEL, wang_model))
    grid_compute_nmda(experiment, g_ampas, nu_thresholds)