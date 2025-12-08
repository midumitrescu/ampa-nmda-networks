from brian2 import ms

from Configuration import Experiment, SynapticParams, NetworkParams, PlotParams
from iteration_4_conductance_based_model.conductance_based_model import sim_and_plot
from iteration_4_conductance_based_model.grid_computations import compare_g_ampa_vs_nu_ext_over_nu_thr


def question_q_network_instability_for_g_0(g=0, g_ampa_to_take=3e-06, nu_ext_over_nu_thr=1.5):
    g_ampas = [2e-06, 2.5e-06, 2.625e-6, 2.75e-06, 3e-06]
    nu_thresholds = [1.5, 1.7, 1.8]
    simulation = {
        "sim_time": 5_000,
        "sim_clock": 0.1 * ms,
        "g": g,
        "epsilon": 0.1,
        "C_ext": 1000,

        "g_L": 0.00004,
        PlotParams.KEY_PANEL: r"Compare $g_\mathrm{AMPA}$ vs $\frac{\nu_\mathrm{Ext}}{\nu_\mathrm{Thr}}$  such that network is stable\\",
        "t_range": [[0, 500], [4000, 5000]],
        "voltage_range": [-70, -30],
        "smoothened_rate_width": 0.5 * ms
    }
    experiment = Experiment(simulation)
    compare_g_ampa_vs_nu_ext_over_nu_thr(experiment, g_ampas, nu_thresholds)

    long_run_to_check_stability_and_cv = experiment.with_property(SynapticParams.KEY_G_AMPA, g_ampa_to_take) \
                .with_property(NetworkParams.KEY_NU_E_OVER_NU_THR, nu_ext_over_nu_thr) \
                .with_property(Experiment.KEY_SIM_TIME, 30_000) \
                .with_property(PlotParams.KEY_T_RANGE, [[10_000, 30_000], [20_000, 20_500]])\
                .with_property(PlotParams.KEY_PANEL, "Example of parameters when model is stable (SI slow)")

    sim_and_plot(long_run_to_check_stability_and_cv)

def question_q_network_instability_for_g_1():
    question_q_network_instability_for_g_0(g=1)

def question_q_network_instability_for_g_2():
    question_q_network_instability_for_g_0(g=2)



    # I am missing the plotting of the follow up question i.e.