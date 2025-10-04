from loguru import logger
from brian2 import ufarad, cm, siemens, mV, ms, msiemens


class SynapticParams:
    KEY_SYNAPTIC_STRENGTH = "J"
    KEY_SYNAPTIC_DELAY = "D"

    KEY_G_AMPA = "g_ampa"
    KEY_G_GABA = "g_gaba"

    KEY_TAU_AMPA = "tau_ampa_ms"
    KEY_TAU_GABA = "tau_gaba_ms"

    KEY_E_AMPA = "E_ampa"
    KEY_E_GABA = "E_gaba"

    def __init__(self, params: dict):

        self.J = params.get(SynapticParams.KEY_SYNAPTIC_STRENGTH, 0.5 * mV)
        self.D = params.get(SynapticParams.KEY_SYNAPTIC_DELAY, 1.5 * ms)

        self.g_ampa = params.get(SynapticParams.KEY_G_AMPA, 0) * siemens / cm ** 2
        self.g_gaba = params.get(SynapticParams.KEY_G_AMPA, 0) * siemens / cm ** 2

        self.tau_ampa = params.get(SynapticParams.KEY_TAU_AMPA, 2) * ms
        self.tau_gaba = params.get(SynapticParams.KEY_TAU_AMPA, 2) * ms

        self.e_ampa = params.get(SynapticParams.KEY_E_AMPA, 0 * mV)
        self.e_gaba = params.get(SynapticParams.KEY_E_AMPA, -80 * mV)

    def __str__(self):
        return f"{self.__class__}(J={self.J}, D={self.D})"

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

    def __init__(self, params: dict):

        self.synaptic_params = SynapticParams(params)

        self.g = params.get(NetworkParams.KEY_G, 0)
        self.N_E = params.get(NetworkParams.KEY_N_E, 10_000)
        self.gamma = params.get(NetworkParams.KEY_GAMMA, 0.25)
        self.epsilon = params.get(NetworkParams.KEY_EPSILON, 0.1)

        self.N_I = round(self.gamma * self.N_E)
        self.N = self.N_E + self.N_I

        self.C_E = int(self.epsilon * self.N_E)

        self.C_ext = params.get(NetworkParams.KEY_C_EXT, self.C_E)

    def __str__(self):
        return f"{self.__class__}({self.KEY_G}={self.g}, {self.KEY_GAMMA}={self.gamma}, {self.KEY_EPSILON}={self.epsilon}, {self.KEY_N_E}={self.N_E}, {self.KEY_N_I}={self.N_I}, \
                {self.KEY_N}={self.N}, C_E={self.C_E}, {self.KEY_C_EXT}={self.C_ext})"


class NeuronModelParams:
    KEY_NEURON_C = "C"
    KEY_NEURON_G_L = "g_L"
    KEY_NEURON_THRESHOLD = "theta"
    KEY_NEURON_V_R = "v_reset"
    KEY_NEURON_E_L = "E_leak"
    KEY_TAU_REF = "tau_ref"

    def __init__(self, params: dict, network_params: NetworkParams = NetworkParams):

        self.synaptic_params = SynapticParams(params)

        self.C = params.get(NeuronModelParams.KEY_NEURON_C, 1 * ufarad * (cm ** -2))
        self.g_L = params.get(NeuronModelParams.KEY_NEURON_G_L, 0.00004 * siemens * (cm ** -2))
        self.theta = params.get(NeuronModelParams.KEY_NEURON_THRESHOLD, -40 * mV)
        self.V_r = params.get(NeuronModelParams.KEY_NEURON_V_R, -65 * mV)
        self.E_leak = params.get(NeuronModelParams.KEY_NEURON_E_L, -65 * mV)
        self.tau_rp = params.get(NeuronModelParams.KEY_TAU_REF, 2 * ms)

        self.tau = self.C / self.g_L
        self.nu_thr = (self.theta - self.E_leak) / (self.synaptic_params.J * network_params.C_E * self.tau)

        logger.info("Computed tau membrane = {}, nu threshold = {}", self.tau, self.nu_thr)

    def __str__(self):
        return (f"{self.__class__}({NeuronModelParams.KEY_NEURON_C}={self.C}, {NeuronModelParams.KEY_NEURON_G_L}={self.g_L}, \
                {NeuronModelParams.KEY_NEURON_THRESHOLD}={self.theta}, {NeuronModelParams.KEY_NEURON_V_R}={self.V_r}, \
                {NeuronModelParams.KEY_NEURON_E_L}={self.E_leak}, {NeuronModelParams.KEY_TAU_REF}={self.tau_rp}, nu_thr(computed)={self.nu_thr},\
                tau membrane(computed)={self.tau})")


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

    def __init__(self, params: dict):
        self.sim_time = params.get(Experiment.KEY_SIM_TIME, params.get(PlotParams.KEY_T_RANGE, (0, 200))[1]) * ms
        self.network_params = NetworkParams(params)
        self.synaptic_params = SynapticParams(params)
        self.neuron_params = NeuronModelParams(params=params, network_params=self.network_params)
        self.plot_params = PlotParams(params)

        self.nu_ext_over_nu_thr = params.get(NetworkParams.KEY_NU_E_OVER_NU_THR, 1)
        self.nu_thr = self.neuron_params.nu_thr
        self.nu_ext = self.nu_ext_over_nu_thr * self.nu_thr

        self.mean_excitatory_input = self.synaptic_params.J * self.neuron_params.tau * self.network_params.C_E * self.nu_ext
        self.mean_inhibitory_input = - self.network_params.g * self.synaptic_params.J * self.neuron_params.tau * self.network_params.C_E * self.nu_ext

        self.sim_clock = params.get(Experiment.KEY_SIMULATION_CLOCK, 0.05 * ms)

    def gen_plot_title(self):
        return f''' {self.plot_params.panel}
        Network: [N={self.network_params.N}, $N_E={self.network_params.N_E}$, $N_I={self.network_params.N_I}$, $\gamma={self.network_params.gamma}$, $\\epsilon={self.network_params.epsilon}$]
    Input Params: [$\\nu_T={self.nu_thr}$, $\\frac{{\\nu_E}}{{\\nu_T}}={self.nu_ext_over_nu_thr: .2f}$, $\\nu_E={self.nu_ext: .2f}$ Hz]
    Neuron Params: [$C={self.neuron_params.C * cm**2}, g_L= {self.neuron_params.g_L * cm**2}, \\theta={self.neuron_params.theta}, V_R={self.neuron_params.V_r}, E_L={self.neuron_params.E_leak}, \\tau_M={self.neuron_params.tau}, \\tau_\u007b ref \u007d={self.neuron_params.tau_rp}$]
'''
    #\\frac\u007b\\mu\\text\u007b F \u007d\u007d\u007b \\text\u007b cm \u007d\u007d \u007d\u007d
