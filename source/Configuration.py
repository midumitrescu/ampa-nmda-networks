from loguru import logger
from brian2 import ufarad, cm, siemens, mV, ms


class NetworkParams:
    KEY_G = "g"
    KEY_NU_THR = "nu_thr"
    KEY_NU_E_OVER_NU_THR = "nu_ext_over_nu_thr"

    KEY_N = "N"
    KEY_N_E = "N_E"
    KEY_N_I = "N_I"

    KEY_C_EXT = "C_ext"

    KEY_GAMMA = "GAMMA"
    KEY_EPSILON = "epsilon"

    def __init__(self, params):
        self.g = params.get(NetworkParams.KEY_G, 0)
        self.N_E = params.get(NetworkParams.KEY_N_E, 10_000)
        self.gamma = params.get(NetworkParams.KEY_GAMMA, 0.25)
        self.epsilon = params.get(NetworkParams.KEY_EPSILON, 0.1)

        self.N_I = round(self.gamma * self.N_E)
        self.N = self.N_E + self.N_I

        self.C_E = int(self.epsilon * self.N_E)

        self.C_ext = params.get(NetworkParams.KEY_C_EXT, self.C_E)


class NeuronModelParams:
    KEY_NEURON_C = "C"
    KEY_NEURON_G_L = "g_L"
    KEY_NEURON_THRESHOLD = "theta"
    KEY_NEURON_V_R = "v_reset"
    KEY_NEURON_V_L = "v_leak"
    KEY_TAU_REF = "tau_ref"

    # synapse parameters
    J = 0.5 * mV
    D = 1.5 * ms

    def __init__(self, params: dict, network_params: NetworkParams = NetworkParams):
        self.C = params.get(NeuronModelParams.KEY_NEURON_C, 1 * ufarad * (cm ** -2))
        self.g_L = params.get(NeuronModelParams.KEY_NEURON_G_L, 0.004 * siemens * (cm ** -2))
        self.theta = params.get(NeuronModelParams.KEY_NEURON_THRESHOLD, -40 * mV)
        self.V_r = params.get(NeuronModelParams.KEY_NEURON_V_R, -65 * mV)
        self.E_leak = params.get(NeuronModelParams.KEY_NEURON_G_L, -65 * mV)
        self.tau_rp = params.get(NeuronModelParams.KEY_TAU_REF, 2 * ms)


        self.nu_thr = (self.theta - self.E_leak) / (self.J * network_params.C_E * self.tau)
        self.tau = self.C / self.g_L



        logger.info("Computed nu threshold = {}", self.nu_thr)


class PlotParams:
    KEY_PANEL = "panel"

    KEY_T_RANGE = "t_range"
    KEY_RATE_RANGE = "rate_range"
    KEY_VOLTAGE_RANGE = "voltage_range"
    KEY_RATE_TICK_STEP = "rate_tick_step"

    KEY_PLOT_SMOOTH_WIDTH = "smoothened_rate_width"


    def __init__(self, params):
        self.panel = params.get(PlotParams.KEY_PANEL, "")
        self.t_range = params.get(PlotParams.KEY_T_RANGE, [0, 100])
        self.rate_range = params.get(PlotParams.KEY_RATE_RANGE, [0, 150])
        self.voltage_range = params.get(PlotParams.KEY_VOLTAGE_RANGE, None)

        self.rate_tick_step = params.get(PlotParams.KEY_RATE_TICK_STEP, 30)
        self.smoothened_rate_width = params.get(self.KEY_PLOT_SMOOTH_WIDTH, None)
        self.plot_smoothened_rate = self.smoothened_rate_width is not None


class Experiment:
    KEY_SIM_TIME = "sim_time"
    KEY_SIMULATION_CLOCK = "simulation_clock"

    def __init__(self, params):
        self.sim_time = params.get(Experiment.KEY_SIM_TIME, params.get(PlotParams.KEY_T_RANGE, (0, 200))[1]) * ms
        self.network_params = NetworkParams(params)
        self.neuron_params = NeuronModelParams(self.network_params)
        self.plot_params = PlotParams(params)

        self.nu_ext_over_nu_thr = params.get(NetworkParams.KEY_NU_E_OVER_NU_THR, 1)
        self.nu_thr = self.neuron_params.nu_thr
        self.nu_ext = self.nu_ext_over_nu_thr * self.nu_thr

        self.mean_excitatory_input = self.neuron_params.J * self.neuron_params.tau * self.network_params.C_E * self.nu_ext
        self.mean_inhibitory_input = - self.network_params.g * self.neuron_params.J * self.neuron_params.tau * self.network_params.C_E * self.nu_ext

        self.sim_clock = params.get(Experiment.KEY_SIMULATION_CLOCK, 0.05 * ms)
